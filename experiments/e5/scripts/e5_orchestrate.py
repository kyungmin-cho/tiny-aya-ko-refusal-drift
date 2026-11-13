"""E5 오케스트레이터(포드, 표준 라이브러리): 학습 2 → 평가 7 → α 선택·공개 후보 묶기 → 집계·AuditKIT를 순서대로 하나씩 돈다(GPU를 겹쳐 쓰지 않음).

순서: preflight(스냅샷·입력 sha·학습 파일 재생성, DO_NOT_DELETE_POD 표시) → train_T100 → train_C100 → eval_base → eval_T100_ep3 → eval_C100_ep3
      → eval_T100_ep3_resta{0.25,0.5,0.75,1.0} → select_release(규칙으로 α 고르기, release/ 묶기, 포드 sha256, RELEASE_READY)
      → agg(PROJ/scripts/e4_aggregate.py: MultiJail ko·XSTest 짝 비교) → auditkit(e5_auditkit.py: 모든 평가셋)
이어 하기: 끝난 학습(DONE.json + ep1·ep3 어댑터)·평가(e5_eval.eval_done)는 건너뛴다. 평가는 건별 jsonl이라 끊긴 곳부터.
실패: 그 단계에 기대는 단계만 막고 나머지는 계속. 끝에 실패가 있으면 종료 코드 1, 로그 끝 줄 E5_DONE_WITH_FAILURES.
DO_NOT_DELETE_POD는 preflight에서 쓰고, 맥 e5_mac_fetch.py가 가중치 sha256을 맞춘 뒤에만 지운다(포드 쪽에서는 지우지 않는다).
사용(GPU 머신, 레포 루트): nohup uv run python -I -B experiments/e5/scripts/e5_orchestrate.py experiments/e5 > experiments/e5/logs/orchestrate.log 2>&1 &
       --dry-run [--plan-json p]  /  --skip <단계…>(시간이 모자랄 때 접기)  /  --only <단계…>
"""
import argparse, datetime, json, os, shlex, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(PROJ, "scripts")); sys.path.insert(0, HERE)
import e4_common as C  # noqa: E402
from e5_eval import eval_done  # noqa: E402

KST = datetime.timezone(datetime.timedelta(hours=9))
ALPHAS = (0.25, 0.5, 0.75, 1.0)
# 시간(초). 측정 = E4(agg/e4_timing.json): 스텝 1.3019초(T100), 학습 단계 34.7분(적재 포함), 평가 1모델 3.7~3.9분(765건), ReSta 평가 5.3~5.4분.
# 추정 = E5 평가는 1,280건 생성·판정 + MMLU 600 로짓(측정 없음) → 3.8분 × 1,280/765 + 0.5분, ReSta 병합 저장 1.0분(측정 없음)
T = dict(train_min=34.7, eval_min=3.8 * 1280 / 765 + 0.5, resta_extra_min=1.6, resta_save_min=1.0, select_min=1.0, agg_min=0.5, auditkit_min=1.0)


def plan():
    S = []
    add = lambda **k: S.append(dict(dict(deps=[]), **k))
    add(name="preflight", kind="preflight")
    for arm in ("T100", "C100"):
        add(name=f"train_{arm}", kind="train", arm=arm, deps=["preflight"])
    add(name="eval_base", kind="eval", tag="base", spec={"kind": "base"}, deps=["preflight"])
    for arm in ("T100", "C100"):
        add(name=f"eval_{arm}_ep3", kind="eval", tag=f"{arm}_ep3", spec={"kind": "adapter", "adapter": f"train/{arm}/ep3"}, deps=[f"train_{arm}"])
    for al in ALPHAS:
        add(name=f"eval_T100_ep3_resta{al}", kind="eval", tag=f"T100_ep3_resta{al}", spec={"kind": "resta", "adapter": "train/T100/ep3", "alpha": al}, deps=["train_T100"])
    add(name="select_release", kind="select", deps=["train_C100", "eval_base"])
    add(name="agg", kind="agg")
    add(name="auditkit", kind="auditkit")
    return S


def est_min(s):
    k = s["kind"]
    if k == "train":
        return T["train_min"]
    if k == "eval":
        return T["eval_min"] + (T["resta_extra_min"] + T["resta_save_min"] if s["spec"]["kind"] == "resta" else 0)
    return dict(select=T["select_min"], agg=T["agg_min"], auditkit=T["auditkit_min"]).get(k, 0.0)


def is_done(root, s):
    if s["kind"] == "train":
        d = os.path.join(root, "train", s["arm"])
        return os.path.exists(os.path.join(d, "DONE.json")) and all(os.path.exists(os.path.join(d, f"ep{e}", "adapter_model.safetensors")) for e in (1, 3))
    if s["kind"] == "eval":
        return eval_done(os.path.join(root, "eval", s["tag"]), need_saved=s["spec"]["kind"] == "resta")
    return False


def preflight(root):
    open(os.path.join(root, "DO_NOT_DELETE_POD"), "a").write(
        f"{datetime.datetime.now(KST).isoformat(timespec='seconds')} 가중치를 맥으로 받아 sha256을 맞추기 전에는 포드를 지우지 않는다(맥 e5_mac_fetch.py만 지움)\n")
    man = json.load(open(os.path.join(root, "data", "data_manifest.json")))
    probs = []
    for name, p in (("tiny-aya-global", C.AYA_SNAP), ("tiny-aya-base", C.AYA_BASE_SNAP), ("Qwen3Guard-Gen-4B", C.model_snap("Qwen/Qwen3Guard-Gen-4B", C.REV["guard"]))):
        if not os.path.isdir(p) or not any(f.endswith(".safetensors") for f in os.listdir(p)):
            probs.append(f"스냅샷 없음 또는 가중치 없음: {name}")
    from e5_eval import item_paths
    try:
        item_paths(root)
    except Exception as e:  # noqa: BLE001
        probs.append(str(e))
    if man["train_cfg"]["train_seed"] == 20261111:
        probs.append("E5 시드가 E4와 같음")
    if probs:
        raise RuntimeError("; ".join(probs))
    return C.materialize(man, os.path.join(root, "data"))


def log_event(root, **ev):
    os.makedirs(os.path.join(root, "state"), exist_ok=True)
    ev["kst"] = datetime.datetime.now(KST).isoformat(timespec="seconds")
    with open(os.path.join(root, "state", "events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    print(json.dumps(ev, ensure_ascii=False), flush=True)


def command(s, root, runner):
    sc = lambda n: os.path.join(HERE, n)
    k = s["kind"]
    if k == "train":
        return runner + [sc("e5_train.py"), s["arm"], root]
    if k == "eval":
        spec = dict(s["spec"])
        if "adapter" in spec:
            spec["adapter"] = os.path.join(root, spec["adapter"])
        return runner + [sc("e5_eval.py"), s["tag"], json.dumps(spec), root]
    if k == "select":
        return [sys.executable, "-I", "-B", sc("e5_select_release.py"), root]
    if k == "agg":
        return [sys.executable, "-I", "-B", os.path.join(PROJ, "scripts", "e4_aggregate.py"), root]
    if k == "auditkit":
        tags = ["base", "T100_ep3", "C100_ep3"] + [f"T100_ep3_resta{a}" for a in ALPHAS]
        pairs = ["base:T100_ep3", "base:C100_ep3", "C100_ep3:T100_ep3"] + [f"T100_ep3:T100_ep3_resta{a}" for a in ALPHAS] + [f"base:T100_ep3_resta{a}" for a in ALPHAS]
        return runner + [sc("e5_auditkit.py"), os.path.join(root, "eval"), os.path.join(root, "agg", "auditkit"), "--models", *tags, "--pairs", *pairs,
                         "--parts", "e1", "mjen", "xs_safe", "xs_unsafe", "sel_h", "sel_b", "mmlu_rep", "mmlu_sel"]
    raise ValueError(s)


def run(root, runner, dry=False, skip=(), only=None, plan_json=None):
    steps = plan(); t0 = time.time(); status = {}; rows = []; cum = 0.0
    for s in steps:
        done = is_done(root, s)
        if (only and s["name"] not in only) or s["name"] in skip:
            status[s["name"]] = "done" if done else "skipped"
            if not dry:
                log_event(root, step=s["name"], event="skip_user")
            rows.append(dict(name=s["name"], status=status[s["name"]], est_min=round(est_min(s), 1), cum_min=round(cum, 1)))
            continue
        blocked = [d for d in s["deps"] if status.get(d) in ("failed", "blocked", "skipped")]
        if dry:
            st = "done" if done else ("blocked" if blocked else "todo")
            if st == "todo":
                cum += est_min(s)
            status[s["name"]] = st
            rows.append(dict(name=s["name"], status=st, est_min=round(est_min(s), 1), cum_min=round(cum, 1),
                             cmd="(in-process)" if s["kind"] == "preflight" else " ".join(shlex.quote(x) for x in command(s, root, runner))))
            continue
        if done:
            status[s["name"]] = "done"; log_event(root, step=s["name"], event="skip_done"); continue
        if blocked:
            status[s["name"]] = "blocked"; log_event(root, step=s["name"], event="blocked", by=blocked); continue
        log_event(root, step=s["name"], event="start", est_min=round(est_min(s), 1), elapsed_min=round((time.time() - t0) / 60, 1))
        ts = time.time()
        try:
            if s["kind"] == "preflight":
                info = preflight(root); rc = 0
                log_event(root, step=s["name"], event="preflight", data=info)
            else:
                os.makedirs(os.path.join(root, "logs"), exist_ok=True)
                with open(os.path.join(root, "logs", f"{s['name']}.log"), "a", encoding="utf-8") as lf:
                    rc = subprocess.call(command(s, root, runner), stdout=lf, stderr=subprocess.STDOUT)
        except Exception as ex:  # noqa: BLE001
            rc = -1; log_event(root, step=s["name"], event="error", err=f"{type(ex).__name__}: {str(ex)[:300]}")
        ok = rc == 0 and (s["kind"] in ("preflight", "select", "agg", "auditkit") or is_done(root, s))
        status[s["name"]] = "done" if ok else "failed"
        log_event(root, step=s["name"], event="end", ok=ok, rc=rc, sec=round(time.time() - ts, 1))
    if dry:
        out = dict(root=root, assumptions_min=T, steps=rows, total_todo_min=round(cum, 1))
        if plan_json:
            json.dump(out, open(plan_json, "w"), ensure_ascii=False, indent=1)
        for r in rows:
            print(f"{r['name']:<28} {r['status']:<8} {r['est_min']:>6.1f}분 누적 {r['cum_min']:>6.1f}분")
        print(f"남은 예상 {out['total_todo_min']}분")
        return out
    failed = [k for k, v in status.items() if v == "failed"]
    log_event(root, step="ALL", event="finished", failed=failed, status=status, wall_min=round((time.time() - t0) / 60, 1),
              release_ready=os.path.exists(os.path.join(root, "RELEASE_READY")))
    print("E5_DONE" if not failed else "E5_DONE_WITH_FAILURES", flush=True)
    return status


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--plan-json", default=None)
    ap.add_argument("--skip", nargs="*", default=[]); ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--runner", default=os.environ.get("E5_RUNNER", "uv run python -I -B"))
    a = ap.parse_args()
    st = run(a.root, shlex.split(a.runner), dry=a.dry_run, skip=set(a.skip), only=a.only, plan_json=a.plan_json)
    if not a.dry_run and any(v == "failed" for v in st.values()):
        sys.exit(1)
