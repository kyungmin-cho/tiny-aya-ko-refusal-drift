"""E4 오케스트레이터(포드, 표준 라이브러리만): 학습·평가·ReSta·집계를 정해 둔 순서로 하나씩 돈다. GPU를 겹쳐 쓰지 않는다(46GB, 학습 피크 35.65GB 측정).

순서(우선순위)
  A. 100 팔: 사전 점검(스냅샷·입력 sha·학습 파일) → T100 학습 → C100 학습 → 평가 base → T100_ep3 → C100_ep3 → T100_ep1 → C100_ep1
             → ReSta α0.1·α0.25(T100 ep3) 평가 → 집계
     (주 결과인 ep3을 ep1보다 먼저 잰다. 포드가 중간에 멈춰도 주 결과가 먼저 남게 하려는 것)
  B. 200 팔: T200 학습 → C200 학습 → T200_ep3 → C200_ep3 → ReSta α0.1·α0.25(T200 ep3) → 집계
  C. 남으면: T200_ep1 → C200_ep1 → 집계 (optional: 첫 선택 단계에서 경과 + 선택 단계 예상 합 ≤ --budget-min이면 모두, 아니면 모두 건너뜀)
이어 하기: 끝난 단계는 건너뛴다. 학습 = train/<arm>/DONE.json + ep1·ep3 어댑터, 평가 = eval/<tag>/DONE.json + id 315·450·315·450.
  학습은 단계 중간에서 이어지지 않는다(끊기면 그 팔을 처음부터). 평가는 건별 jsonl이라 끊긴 곳부터 이어진다.
실패: 그 단계를 failed로 적고, 그 단계에 기대는 단계만 막고 나머지는 계속한다. 끝에 실패가 있으면 종료 코드 1.
사용(GPU 머신, 레포 루트에서):
  nohup python3 -I scripts/e4_orchestrate.py experiments/e4 > experiments/e4/logs/orchestrate.log 2>&1 &
  python3 -I scripts/e4_orchestrate.py experiments/e4 --dry-run [--plan-json <path>]   (실행 없이 계획·상태·예상 시간)
"""
import argparse, datetime, json, os, shlex, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e4_common as C  # noqa: E402
from e4_eval import eval_done  # noqa: E402  (torch는 main 안에서만 import됨)

KST = datetime.timezone(datetime.timedelta(hours=9))

# ---------------- 시간 가정(초) ----------------
# 측정: 스텝 1.2934초(e4_bench_L768.json sec_per_step_mean), 적재 5.4초(같은 파일 load_sec),
#       평가 1모델 206.6초 = E1 생성 55.77 + 판정 36.73 + XSTest 생성 73.54 + 판정 40.53(11-11 meta.json; 모델 적재 제외)
# 가정(미측정): 평가당 모델 적재·병합 60초, ReSta 1회(base·aligned CPU 적재 + 연산) 180초, 학습 토큰화·어댑터 저장 20초, 집계 5초
T = dict(sec_per_step=1.2934, train_load=5.4, eval_measured=206.6, eval_load_assumed=60.0, resta_assumed=180.0,
         train_overhead_assumed=20.0, agg_assumed=5.0)
STEPS_PER_EPOCH = {"T100": 525, "C100": 525, "T200": 550, "C200": 550}  # data_manifest.json checks.steps_per_epoch와 같아야 함(단위시험)


def adapter(arm, ep):
    return f"train/{arm}/ep{ep}"


def plan():
    S = []
    add = lambda **k: S.append(dict(dict(deps=[], optional=False), **k))
    add(name="preflight", phase="A", kind="preflight")
    for arm in ("T100", "C100"):
        add(name=f"train_{arm}", phase="A", kind="train", arm=arm, deps=["preflight"])
    add(name="eval_base", phase="A", kind="eval", tag="base", spec={"kind": "base"}, deps=["preflight"])
    for arm, ep in (("T100", 3), ("C100", 3), ("T100", 1), ("C100", 1)):
        add(name=f"eval_{arm}_ep{ep}", phase="A", kind="eval", tag=f"{arm}_ep{ep}", spec={"kind": "adapter", "adapter": adapter(arm, ep)}, deps=[f"train_{arm}"])
    for al in C.RESTA_ALPHAS:
        add(name=f"eval_T100_ep3_resta{al}", phase="A", kind="eval", tag=f"T100_ep3_resta{al}",
            spec={"kind": "resta", "adapter": adapter("T100", 3), "alpha": al}, deps=["train_T100"])
    add(name="agg_A", phase="A", kind="agg")
    for arm in ("T200", "C200"):
        add(name=f"train_{arm}", phase="B", kind="train", arm=arm, deps=["preflight"])
    for arm in ("T200", "C200"):
        add(name=f"eval_{arm}_ep3", phase="B", kind="eval", tag=f"{arm}_ep3", spec={"kind": "adapter", "adapter": adapter(arm, 3)}, deps=[f"train_{arm}"])
    for al in C.RESTA_ALPHAS:
        add(name=f"eval_T200_ep3_resta{al}", phase="B", kind="eval", tag=f"T200_ep3_resta{al}",
            spec={"kind": "resta", "adapter": adapter("T200", 3), "alpha": al}, deps=["train_T200"])
    add(name="agg_B", phase="B", kind="agg")
    for arm in ("T200", "C200"):
        add(name=f"eval_{arm}_ep1", phase="C", kind="eval", tag=f"{arm}_ep1", spec={"kind": "adapter", "adapter": adapter(arm, 1)},
            deps=[f"train_{arm}"], optional=True)
    add(name="agg_C", phase="C", kind="agg", optional=True)
    return S


def est_sec(s):
    if s["kind"] == "train":
        return T["train_load"] + T["train_overhead_assumed"] + 3 * STEPS_PER_EPOCH[s["arm"]] * T["sec_per_step"]
    if s["kind"] == "eval":
        return T["eval_measured"] + T["eval_load_assumed"] + (T["resta_assumed"] if s["spec"]["kind"] == "resta" else 0)
    if s["kind"] == "agg":
        return T["agg_assumed"]
    return 0.0


def est_sec_measured_only(s):
    if s["kind"] == "train":
        return T["train_load"] + 3 * STEPS_PER_EPOCH[s["arm"]] * T["sec_per_step"]
    if s["kind"] == "eval":
        return T["eval_measured"]
    return 0.0


def is_done(root, s):
    if s["kind"] == "train":
        d = os.path.join(root, "train", s["arm"])
        return os.path.exists(os.path.join(d, "DONE.json")) and all(
            os.path.exists(os.path.join(d, f"ep{e}", "adapter_model.safetensors")) for e in (1, C.EPOCHS))
    if s["kind"] == "eval":
        return eval_done(os.path.join(root, "eval", s["tag"]))
    return False  # preflight·agg는 매번 다시 한다(싸다)


def preflight(root):
    """스냅샷·평가 입력 sha·학습 파일(없으면 HF 캐시 원본에서 id로 다시 만듦)을 확인한다."""
    man = json.load(open(os.path.join(root, "data", "data_manifest.json")))
    proj = os.path.dirname(os.path.dirname(os.path.abspath(root)))
    probs = []
    for name, p in (("tiny-aya-global", C.AYA_SNAP), ("tiny-aya-base", C.AYA_BASE_SNAP),
                    ("Qwen3Guard-Gen-4B", C.model_snap("Qwen/Qwen3Guard-Gen-4B", C.REV["guard"]))):
        if not os.path.isdir(p) or not any(f.endswith(".safetensors") for f in os.listdir(p)):
            probs.append(f"스냅샷 없음 또는 가중치 없음: {name} {p}")
    for key in ("e1_items", "xstest_items"):
        p = os.path.join(proj, man["eval_inputs"][key]["path"])
        if not os.path.exists(p) or C.sha256_file(p) != man["eval_inputs"][key]["sha256"]:
            probs.append(f"평가 입력 sha 불일치 또는 없음: {p}")
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
    if s["kind"] == "train":
        return runner + [sc("e4_train.py"), s["arm"], root]
    if s["kind"] == "eval":
        spec = dict(s["spec"])
        if "adapter" in spec:
            spec["adapter"] = os.path.join(root, spec["adapter"])
        return runner + [sc("e4_eval.py"), s["tag"], json.dumps(spec), root]
    if s["kind"] == "agg":
        return [sys.executable, "-I", sc("e4_aggregate.py"), root]
    raise ValueError(s)


def run(root, runner, dry=False, budget_min=190.0, skip_optional=False, only=None, plan_json=None):
    steps = plan()
    t0 = time.time()
    status = {}
    rows = []
    cum = cum_m = 0.0
    for s in steps:
        done = is_done(root, s)
        if only and s["name"] not in only and s["kind"] not in ("agg", "preflight"):
            status[s["name"]] = "done" if done else "not_selected"; continue
        e = est_sec(s)
        blocked = [d for d in s["deps"] if status.get(d) in ("failed", "blocked", "skipped_budget", "not_selected")]
        if dry:
            st = "done" if done else ("blocked" if blocked else "todo")
            if st == "todo":
                cum += e; cum_m += est_sec_measured_only(s)
            status[s["name"]] = "done" if done else ("blocked" if blocked else "todo")
            rows.append(dict(name=s["name"], phase=s["phase"], optional=s["optional"], status=st, est_min=round(e / 60, 1),
                             est_measured_only_min=round(est_sec_measured_only(s) / 60, 1),
                             cum_min=round(cum / 60, 1), cum_measured_only_min=round(cum_m / 60, 1),
                             cmd=" ".join(shlex.quote(x) for x in command(s, root, runner)) if s["kind"] not in ("preflight",) else "(in-process)"))
            continue
        if done:
            status[s["name"]] = "done"; log_event(root, step=s["name"], event="skip_done"); continue
        if blocked:
            status[s["name"]] = "blocked"; log_event(root, step=s["name"], event="blocked", by=blocked); continue
        elapsed_min = (time.time() - t0) / 60
        if s["optional"] and "optional_ok" not in status:
            rest = sum(est_sec(x) for x in steps if x["optional"] and not is_done(root, x)) / 60  # T200_ep1·C200_ep1을 함께 판정
            status["optional_ok"] = (not skip_optional) and elapsed_min + rest <= budget_min
            log_event(root, step="optional", event="decide", ok=status["optional_ok"], elapsed_min=round(elapsed_min, 1), optional_est_min=round(rest, 1))
        if s["optional"] and not status["optional_ok"]:
            status[s["name"]] = "skipped_budget"; log_event(root, step=s["name"], event="skip_budget", elapsed_min=round(elapsed_min, 1), est_min=round(e / 60, 1)); continue
        log_event(root, step=s["name"], event="start", est_min=round(e / 60, 1))
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
        ok = rc == 0 and (s["kind"] in ("preflight", "agg") or is_done(root, s))
        status[s["name"]] = "done" if ok else "failed"
        log_event(root, step=s["name"], event="end", ok=ok, rc=rc, sec=round(time.time() - ts, 1))
    if dry:
        out = dict(root=root, assumptions_sec=T, steps=rows, total_todo_min=round(cum / 60, 1), total_todo_measured_only_min=round(cum_m / 60, 1),
                   required_todo_min=round(sum(r["est_min"] for r in rows if r["status"] == "todo" and not r["optional"]), 1),
                   required_todo_measured_only_min=round(sum(r["est_measured_only_min"] for r in rows if r["status"] == "todo" and not r["optional"]), 1))
        if plan_json:
            json.dump(out, open(plan_json, "w"), ensure_ascii=False, indent=1)
        for r in rows:
            print(f"{r['phase']} {r['name']:<28} {r['status']:<8} {'opt ' if r['optional'] else '    '}{r['est_min']:>6.1f}분 누적 {r['cum_min']:>6.1f}분")
        print(f"필수 남은 예상 {out['required_todo_min']}분(측정값만 {out['required_todo_measured_only_min']}분), 선택 포함 {out['total_todo_min']}분")
        return out
    status.pop("optional_ok", None)
    failed = [k for k, v in status.items() if v == "failed"]
    log_event(root, step="ALL", event="finished", failed=failed, status=status, wall_min=round((time.time() - t0) / 60, 1))
    print("E4_DONE" if not failed else "E4_DONE_WITH_FAILURES", flush=True)
    return status


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan-json", default=None)
    ap.add_argument("--budget-min", type=float, default=190.0)
    ap.add_argument("--skip-optional", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="이 단계 이름만(집계는 늘 포함)")
    ap.add_argument("--runner", default=os.environ.get("E4_RUNNER", "uv run python -I"), help="하위 스크립트 실행 접두(기본: uv run python -I)")
    a = ap.parse_args()
    st = run(a.root, shlex.split(a.runner), dry=a.dry_run, budget_min=a.budget_min, skip_optional=a.skip_optional, only=a.only, plan_json=a.plan_json)
    if not a.dry_run and any(v == "failed" for v in st.values()):
        sys.exit(1)
