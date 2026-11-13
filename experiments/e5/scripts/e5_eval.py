"""E5 평가(포드, CUDA): 모델 하나를 한 프로세스에서 모든 평가셋에 돌린다(ReSta를 평가셋마다 다시 만들지 않기 위해).

모델 만들기: PROJ/scripts/e4_eval.build_model을 그대로 쓴다(base / adapter 병합 / adapter 병합 + SafeTune ReStaTrainer(dare=True, 0.3, seed 0, cpu)).
생성·판정: PROJ/scripts/e1e3_harness.cmd_gen·cmd_judge를 그대로 쓴다(E4와 같음: 탐욕, 128토큰, 배치 64, 기본 머리말, system 없음, Qwen3Guard-Gen-4B 판정 64토큰).
평가셋(입력 sha256은 data_manifest.json·eval_items_manifest.json과 맞춘다)
  e1   MultiJail ko 315        (E4와 같은 파일, id e1-*)
  mjen MultiJail en 315        (같은 문항 영어 열, id mjen-*)
  xs   XSTest 450(safe 250, unsafe 200)
  sel  α 선택 보류셋 200(AdvBench-Korean 유해 100 selh-*, 무해 kullm 100 selb-*) — 주 보고 지표와 겹치지 않음
  mmlu Global-MMLU ko 600(rep 400 보고용, sel 200 α 선택용): 생성 없이 첫 응답 토큰 위치의 A·B·C·D 로짓 최댓값(e5_build_eval_items.py에 적은 규칙)
ReSta 모델은 생성이 끝난 뒤 병합 가중치를 <root>/release_cand/<tag>/에 safetensors로 저장하고 파일별 sha256을 SAVED.json에 적는다(공개 후보, 약 6.7GB).
건별 jsonl이라 끊겨도 이어서 돈다. 모든 파일이 기대 id 수를 채우고(ReSta는 SAVED.json도) 나면 DONE.json을 쓴다.
사용: uv run python -I -B experiments/e5/scripts/e5_eval.py <tag> '<spec json>' <e5_root> [--bs 64]
"""
import argparse, datetime, gc, json, os, sys, time, types

os.environ["HF_HUB_OFFLINE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(PROJ, "scripts"))
import e4_common as C  # noqa: E402

KST = datetime.timezone(datetime.timedelta(hours=9))
N_EXPECT = dict(e1=315, mjen=315, xs=450, sel=200)
N_MMLU = 600
LETTER_IDS = None  # 실행 때 토크나이저로 'A'~'D' 토큰 id를 구해 맥 측정값(47~50)과 맞춘다


def now():
    return datetime.datetime.now(KST).isoformat(timespec="seconds")


def item_paths(root):
    man = json.load(open(os.path.join(root, "data", "data_manifest.json")))
    em = json.load(open(os.path.join(root, "data", "eval_items_manifest.json")))
    ents = dict(e1=man["eval_inputs"]["e1_items"], xs=man["eval_inputs"]["xstest_items"], mjen=em["files"]["mjen"], sel=em["files"]["sel"], mmlu=em["files"]["mmlu"])
    out = {}
    for k, e in ents.items():
        p = os.path.join(PROJ, e["path"])
        got = C.sha256_file(p)
        if got != e["sha256"]:
            raise RuntimeError(f"평가 입력 sha256 불일치 {k}: {got[:16]} != {e['sha256'][:16]}")
        out[k] = p
    return out


def eval_done(d, need_saved=False):
    if not os.path.exists(os.path.join(d, "DONE.json")):
        return False
    for part, n in N_EXPECT.items():
        for kind in ("gen", "judge"):
            if len({r["id"] for r in C.read_jsonl(os.path.join(d, f"{part}_{kind}.jsonl"))}) != n:
                return False
    if len({r["id"] for r in C.read_jsonl(os.path.join(d, "mmlu.jsonl"))}) != N_MMLU:
        return False
    if need_saved and not os.path.exists(os.path.join(d, "SAVED_OK")):
        return False
    return True


def mmlu_score(tok, model, items_path, out_path, bs=16):
    """첫 응답 토큰 위치에서 A~D 로짓 비교. 이미 있는 id는 건너뛴다."""
    import torch
    items = C.read_jsonl(items_path)
    done = {r["id"] for r in C.read_jsonl(out_path)}
    todo = [it for it in items if it["id"] not in done]
    ids = [tok("ABCD"[i], add_special_tokens=False)["input_ids"] for i in range(4)]
    assert all(len(x) == 1 for x in ids), ids
    ids = [x[0] for x in ids]
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), bs):
            batch = todo[i:i + bs]
            texts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False, add_generation_prompt=True) for it in batch]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                logits = model(**enc, logits_to_keep=1).logits[:, -1, :]  # left padding이라 마지막 위치 = 첫 응답 토큰 예측. logits_to_keep=1: 마지막 위치만 계산(값 같음, 메모리 절약; E5 본실행에서 추가)
            sub = logits[:, ids].float().cpu()
            for it, row in zip(batch, sub):
                pred = "ABCD"[int(row.argmax())]
                f.write(json.dumps(dict(id=it["id"], split=it["split"], answer=it["answer"], pred=pred, correct=pred == it["answer"],
                                        letter_logits=[round(float(x), 4) for x in row]), ensure_ascii=False) + "\n")
            f.flush()
    return ids


def save_merged(model, tok, d, info):
    """ReSta 병합 모델을 저장하고 파일별 sha256을 적는다. 끝나면 SAVED_OK."""
    os.makedirs(d, exist_ok=True)
    t = time.time()
    model.save_pretrained(d, safe_serialization=True, max_shard_size="1GB")  # E5 본실행에서 5GB→1GB: 맥 받기를 조각 병렬로 하기 위해(내용은 같음)
    tok.save_pretrained(d)
    files = sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and f != "SAVED.json")
    sha = {f: C.sha256_file(os.path.join(d, f)) for f in files}
    meta = dict(saved_kst=now(), sec=round(time.time() - t, 1), files=sha, bytes={f: os.path.getsize(os.path.join(d, f)) for f in files},
                resta=info.get("resta"), adapter_sha256=info.get("adapter_sha256"))
    json.dump(meta, open(os.path.join(d, "SAVED.json"), "w"), ensure_ascii=False, indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("spec"); ap.add_argument("root")
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    spec = json.loads(a.spec)
    out = os.path.join(a.root, "eval", a.tag)
    is_resta = spec["kind"] == "resta"
    if eval_done(out, need_saved=is_resta):
        print(f"{a.tag}: 평가 끝남, 건너뜀"); return
    os.makedirs(out, exist_ok=True)
    items = item_paths(a.root)

    import torch
    import e1e3_harness as H
    import e4_eval
    orig_load = H.load
    info = dict(tag=a.tag, spec=spec, started_kst=now())
    cache = {}

    def load(repo, rev):
        if repo == H.GUARD[0]:
            return orig_load(repo, rev)
        if "gen" not in cache:
            cache["gen"] = e4_eval.build_model(spec, info)
        return cache["gen"]

    H.load = load
    H.AYA = (f"e5:{a.tag}", json.dumps(spec, ensure_ascii=False))
    ns = lambda i, o, mx: types.SimpleNamespace(items=i, out=o, bs=a.bs, max_new=mx)
    t = time.time()
    for part in ("e1", "mjen", "xs", "sel"):
        H.cmd_gen(ns(items[part], os.path.join(out, f"{part}_gen.jsonl"), 128))
    info["gen_wall_sec"] = round(time.time() - t, 1)
    t = time.time()
    need_mmlu = len({r["id"] for r in C.read_jsonl(os.path.join(out, "mmlu.jsonl"))}) != N_MMLU
    need_save = is_resta and not os.path.exists(os.path.join(out, "SAVED_OK"))
    if need_mmlu or need_save:
        tok, m = load("gen", None)
        if need_mmlu:
            info["mmlu_letter_ids"] = mmlu_score(tok, m, items["mmlu"], os.path.join(out, "mmlu.jsonl"))
        info["mmlu_wall_sec"] = round(time.time() - t, 1)
        rc_dir = os.path.join(a.root, "release_cand", a.tag)
        if need_save:
            meta = save_merged(m, tok, rc_dir, info)
            info["saved"] = dict(dir=os.path.relpath(rc_dir, a.root), sec=meta["sec"], total_bytes=sum(meta["bytes"].values()))
            open(os.path.join(out, "SAVED_OK"), "w").write(now() + "\n")
        del m
    import resource
    info["max_rss_gb_after_build"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)  # 리눅스 ru_maxrss 단위 KB
    cache.clear(); gc.collect(); torch.cuda.empty_cache()
    t = time.time()
    for part in ("e1", "mjen", "xs", "sel"):
        H.cmd_judge(ns(os.path.join(out, f"{part}_gen.jsonl"), os.path.join(out, f"{part}_judge.jsonl"), 64))
    info["judge_wall_sec"] = round(time.time() - t, 1)
    info["finished_kst"] = now()
    json.dump(info, open(os.path.join(out, "model_info.json"), "w"), ensure_ascii=False, indent=1)
    counts = {f"{p}_{k}": len({r["id"] for r in C.read_jsonl(os.path.join(out, f"{p}_{k}.jsonl"))}) for p in N_EXPECT for k in ("gen", "judge")}
    counts["mmlu"] = len({r["id"] for r in C.read_jsonl(os.path.join(out, "mmlu.jsonl"))})
    ok = all(counts[f"{p}_{k}"] == n for p, n in N_EXPECT.items() for k in ("gen", "judge")) and counts["mmlu"] == N_MMLU
    ok = ok and (not is_resta or os.path.exists(os.path.join(out, "SAVED_OK")))
    if ok:
        json.dump(dict(info, counts=counts), open(os.path.join(out, "DONE.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(dict(tag=a.tag, counts=counts, ok=ok, **{k: info.get(k) for k in ("model_build_sec", "gen_wall_sec", "mmlu_wall_sec", "judge_wall_sec")}), ensure_ascii=False))
    if not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
