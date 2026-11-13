"""E4 평가(포드, CUDA): 모델 하나를 E1과 같은 설정으로 MultiJail ko 315 + XSTest 450에 생성하고 Qwen3Guard-Gen-4B로 판정한다.

e1e3_harness.py를 고치지 않고 import해 cmd_gen·cmd_judge를 그대로 쓴다(탐욕, 128토큰, 배치 64, 기본 머리말, system 없음, 판정 64토큰).
바꿔 끼우는 것은 H.load(생성 모델 적재)와 H.AYA(meta 기록용 이름)뿐이다. 판정기 적재는 원래 함수를 쓴다.
모델 지정(spec, JSON 문자열):
  {"kind":"base"}                                   학습 전 Tiny Aya Global(af89d219) — 짝 비교 기준. E1과 같은 경로
  {"kind":"adapter","adapter":"<dir>"}              원본 + LoRA 어댑터를 bf16으로 병합(merge_and_unload)
  {"kind":"resta","adapter":"<dir>","alpha":0.25}   위 병합본(drifted)에 SafeTune ReStaTrainer(base=tiny-aya-base d62ac17e, aligned=tiny-aya-global af89d219,
                                                    dare=True, dare_drop_rate=0.3, dare_seed=0, device="cpu")를 적용. α마다 새로 병합해서 적용(제자리 변경이라 겹쳐 쓰지 않음)
E1 입력과 같은 파일(experiments/e1e3/e1_items.jsonl, xstest_items.jsonl)을 쓰고 sha256을 data_manifest.json과 맞춘다. id가 E1과 같아 짝 비교가 된다.
건별 jsonl이라 끊겨도 이어서 돈다(하네스가 끝난 id를 건너뜀). 네 파일이 315·450·315·450 id를 모두 채우면 DONE.json을 쓴다.
사용: uv run python -I scripts/e4_eval.py <tag> '<spec json>' <e4_root>
출력: <e4_root>/eval/<tag>/{e1_gen,e1_judge,xs_gen,xs_judge}.jsonl(+ .meta.json), model_info.json, DONE.json
"""
import argparse, datetime, gc, json, os, sys, time, types

os.environ["HF_HUB_OFFLINE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e4_common as C  # noqa: E402

KST = datetime.timezone(datetime.timedelta(hours=9))
N_EXPECT = dict(e1=315, xs=450)


def now():
    return datetime.datetime.now(KST).isoformat(timespec="seconds")


def eval_done(d):
    """끝난 평가인지: DONE.json이 있고 네 파일의 고유 id 수가 315·450·315·450인지(DONE.json을 쓰는 조건과 같음).
    판정기 출력이 파싱되지 않은 건(safety/refusal None)은 끝남 판정에 넣지 않고 집계표에 건수로 낸다(11-11 판정 1,265건 중 0건)."""
    if not os.path.exists(os.path.join(d, "DONE.json")):
        return False
    for part, n in N_EXPECT.items():
        for kind in ("gen", "judge"):
            rows = C.read_jsonl(os.path.join(d, f"{part}_{kind}.jsonl"))
            if len({r["id"] for r in rows}) != n:
                return False
    return True


def build_model(spec, info):
    """spec에 따라 생성 모델을 cuda bf16으로 만든다. 하네스 load와 같은 토크나이저 설정(left padding, pad=eos)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.AYA_SNAP)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t = time.time()
    m = AutoModelForCausalLM.from_pretrained(C.AYA_SNAP, dtype=torch.bfloat16, device_map="cuda")
    if spec["kind"] in ("adapter", "resta"):
        from peft import PeftModel
        ad = spec["adapter"]
        info["adapter_sha256"] = C.sha256_file(os.path.join(ad, "adapter_model.safetensors"))
        m = PeftModel.from_pretrained(m, ad).merge_and_unload()
    if spec["kind"] == "resta":
        from safetune.runner.recover import ReStaTrainer
        assert os.path.isdir(C.AYA_BASE_SNAP), f"tiny-aya-base 스냅샷 없음: {C.AYA_BASE_SNAP} (scripts/e4_pod_download.py로 받는다)"
        base = AutoModelForCausalLM.from_pretrained(C.AYA_BASE_SNAP, dtype=torch.bfloat16, device_map="cpu")
        aligned = AutoModelForCausalLM.from_pretrained(C.AYA_SNAP, dtype=torch.bfloat16, device_map="cpu")
        tr = time.time()
        m = ReStaTrainer(m, base_model=base, aligned_model=aligned, alpha=float(spec["alpha"]), device="cpu", **C.RESTA_DARE).apply()
        info["resta"] = dict(alpha=float(spec["alpha"]), base=dict(repo="CohereLabs/tiny-aya-base", revision=C.REV["aya_base"]),
                             aligned=dict(repo="CohereLabs/tiny-aya-global", revision=C.REV["aya_global"]), **C.RESTA_DARE,
                             device="cpu", sec=round(time.time() - tr, 1), api="safetune.runner.recover.ReStaTrainer(...).apply()")
        del base, aligned; gc.collect()
    m = m.eval()
    info["model_build_sec"] = round(time.time() - t, 1)
    return tok, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("spec"); ap.add_argument("root")
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    spec = json.loads(a.spec)
    out = os.path.join(a.root, "eval", a.tag)
    if eval_done(out):
        print(f"{a.tag}: 평가 끝남, 건너뜀"); return
    os.makedirs(out, exist_ok=True)
    man = json.load(open(os.path.join(a.root, "data", "data_manifest.json")))
    proj = os.path.dirname(os.path.dirname(os.path.abspath(a.root)))  # <proj>/experiments/e4
    items = {}
    for part, key in (("e1", "e1_items"), ("xs", "xstest_items")):
        p = os.path.join(proj, man["eval_inputs"][key]["path"])
        assert C.sha256_file(p) == man["eval_inputs"][key]["sha256"], f"{p} sha256이 E1 입력과 다름"
        items[part] = p

    import torch
    import e1e3_harness as H
    orig_load = H.load
    info = dict(tag=a.tag, spec=spec, started_kst=now())
    cache = {}

    def load(repo, rev):
        if repo == H.GUARD[0]:
            return orig_load(repo, rev)
        if "gen" not in cache:
            cache["gen"] = build_model(spec, info)
        return cache["gen"]

    H.load = load
    H.AYA = (f"e4:{a.tag}", json.dumps(spec, ensure_ascii=False))  # 하네스 meta의 gen_model 칸에 이 평가의 모델 지정이 남는다
    ns = lambda i, o, mx: types.SimpleNamespace(items=i, out=o, bs=a.bs, max_new=mx)
    t = time.time()
    H.cmd_gen(ns(items["e1"], os.path.join(out, "e1_gen.jsonl"), 128))
    H.cmd_gen(ns(items["xs"], os.path.join(out, "xs_gen.jsonl"), 128))
    info["gen_wall_sec"] = round(time.time() - t, 1)
    cache.clear(); gc.collect(); torch.cuda.empty_cache()
    t = time.time()
    H.cmd_judge(ns(os.path.join(out, "e1_gen.jsonl"), os.path.join(out, "e1_judge.jsonl"), 64))
    H.cmd_judge(ns(os.path.join(out, "xs_gen.jsonl"), os.path.join(out, "xs_judge.jsonl"), 64))
    info["judge_wall_sec"] = round(time.time() - t, 1)
    info["finished_kst"] = now()
    json.dump(info, open(os.path.join(out, "model_info.json"), "w"), ensure_ascii=False, indent=1)
    counts = {f"{p}_{k}": len({r["id"] for r in C.read_jsonl(os.path.join(out, f"{p}_{k}.jsonl"))}) for p in N_EXPECT for k in ("gen", "judge")}
    ok = all(counts[f"{p}_{k}"] == n for p, n in N_EXPECT.items() for k in ("gen", "judge"))
    if ok:
        json.dump(dict(info, counts=counts), open(os.path.join(out, "DONE.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(dict(tag=a.tag, counts=counts, ok=ok, **{k: info.get(k) for k in ("model_build_sec", "gen_wall_sec", "judge_wall_sec")}), ensure_ascii=False))
    if not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
