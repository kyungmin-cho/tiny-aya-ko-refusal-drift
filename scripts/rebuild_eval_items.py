"""Rebuild every evaluation input file from public datasets + the id-only manifests, then check sha256.

This repository does not ship harmful prompts. The five evaluation inputs are regenerated locally from the pinned
Hugging Face revisions listed in `scripts/e4_common.py` (REV) and the manifests in `experiments/e5/data/`:

  experiments/e1e3/e1_items.jsonl              MultiJail ko, 315 items        (id e1-<mj_id>)
  experiments/e1e3/xstest_items.jsonl          XSTest, 250 safe + 200 unsafe  (id xs-<row>)
  experiments/e5/data/eval_items/mjen_items.jsonl   MultiJail en, same 315 items  (id mjen-<k>)
  experiments/e5/data/eval_items/sel_items.jsonl    alpha-selection hold-out: AdvBench-Korean 100 (by adv_idx) + kullm-v2 100 (by row id)
  experiments/e5/data/eval_items/mmlu_items.jsonl   Global-MMLU ko test, Random(20261113).sample 600 (400 report / 200 selection)

The row format is byte-identical to the files used in our runs; the script exits non-zero if any sha256 differs.
Nothing is printed except counts and hash prefixes.

Usage (repo root, after `uv sync` and `uv run python -I scripts/download.py open|gated`):
  uv run python -I -B scripts/rebuild_eval_items.py [--out-root <dir>]   # default: write into the repo tree
Additionally needs the AdvBench-Korean snapshot (`leo-bjpark/AdvBench-Korean@f6a20e0`, see eval_items_manifest.json).
"""
import argparse, csv, json, os, random, sys

os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import e4_common as C  # noqa: E402

SEED5 = 20261113
MMLU_PROMPT = "다음은 객관식 문제입니다. 정답의 알파벳(A, B, C, D) 하나만 답하세요.\n\n질문: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n정답:"


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return C.sha256_file(path)


def snap(repo, rev, *parts):
    return os.path.join(C.HUB, f"datasets--{repo.replace('/', '--')}", "snapshots", rev, *parts)


def build_e1():
    with open(snap("DAMO-NLP-SG/MultiJail", C.REV["multijail"], "MultiJail.csv"), encoding="utf-8-sig") as f:
        rows = [dict(mj_id=int(r["id"]), source=r["source"], tags=r["tags"], ko=r["ko"].strip(), en=r["en"].strip()) for r in csv.DictReader(f)]
    assert len(rows) == 315 and all(r["ko"] for r in rows)
    e1 = [dict(id=f"e1-{r['mj_id']}", cond="plain", mj_id=r["mj_id"], prompt=r["ko"], judge_query=r["ko"], tags=r["tags"], source=r["source"]) for r in rows]
    en = {r["mj_id"]: r["en"] for r in rows}
    return e1, en


def build_xstest():
    import pandas as pd
    df = pd.read_parquet(snap("walledai/XSTest", C.REV["xstest"], "data", "train-00000-of-00001.parquet"))
    return [dict(id=f"xs-{i}", cond="xstest_" + str(r["label"]), prompt=str(r["prompt"]).strip(), judge_query=str(r["prompt"]).strip(),
                 xs_type=str(r.get("type", "")), xs_focus=str(r.get("focus", "")), xs_label=str(r["label"]))
            for i, r in df.reset_index(drop=True).iterrows()]


def build_mjen(e1, en):
    out = []
    for k, it in enumerate(e1):
        assert it["id"] == f"e1-{k}"
        p = en[it["mj_id"]]
        assert p
        out.append(dict(id=f"mjen-{k}", cond="mj_en", mj_id=it["mj_id"], prompt=p, judge_query=p, tags=it["tags"], source=it["source"]))
    return out


def build_sel(em):
    import pyarrow.parquet as pq
    adv = pq.read_table(snap("leo-bjpark/AdvBench-Korean", em["revisions"]["advbench_ko"], "data", "train-00000-of-00001.parquet")).to_pylist()
    adv_idx = em["sel"]["harm"]["adv_idx"]
    assert adv_idx == random.Random(SEED5).sample(range(len(adv)), 100), "AdvBench sample differs from manifest"
    selh = [dict(id=f"selh-{j}", cond="sel_harm", adv_idx=i, prompt=adv[i]["prompt"].strip(), judge_query=adv[i]["prompt"].strip())
            for j, i in enumerate(adv_idx)]
    want = {int(r[2:]): r for r in em["sel"]["benign"]["rids"]}
    ku = {}
    with open(C.KU_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in want:
                r = json.loads(line)
                ku[want[i]] = (r["instruction"] + "\n" + r.get("input", "")).strip()
    selb = [dict(id=f"selb-{j}", cond="sel_benign", rid=rid, prompt=ku[rid], judge_query=ku[rid]) for j, rid in enumerate(em["sel"]["benign"]["rids"])]
    return selh + selb


def build_mmlu(em):
    import pyarrow.parquet as pq
    gm = pq.read_table(snap("CohereLabs/Global-MMLU", em["revisions"]["global_mmlu"], "ko", "test-00000-of-00001.parquet")).to_pylist()
    out = []
    for j, i in enumerate(random.Random(SEED5).sample(range(len(gm)), 600)):
        r = gm[i]
        assert r["answer"] in "ABCD"
        out.append(dict(id=f"mmlu-{j}", split="rep" if j < 400 else "sel", gm_idx=i, sample_id=r["sample_id"], subject=r["subject"],
                        subject_category=r["subject_category"], cultural_sensitivity_label=r["cultural_sensitivity_label"], answer=r["answer"],
                        prompt=MMLU_PROMPT.format(q=r["question"].strip(), a=str(r["option_a"]).strip(), b=str(r["option_b"]).strip(),
                                                  c=str(r["option_c"]).strip(), d=str(r["option_d"]).strip())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=ROOT, help="write files under this root instead of the repo (paths stay relative)")
    a = ap.parse_args()
    man = json.load(open(os.path.join(ROOT, "experiments/e5/data/data_manifest.json")))
    em = json.load(open(os.path.join(ROOT, "experiments/e5/data/eval_items_manifest.json")))
    e1, en = build_e1()
    targets = [
        (man["eval_inputs"]["e1_items"], lambda: e1),
        (man["eval_inputs"]["xstest_items"], build_xstest),
        (em["files"]["mjen"], lambda: build_mjen(e1, en)),
        (em["files"]["sel"], lambda: build_sel(em)),
        (em["files"]["mmlu"], lambda: build_mmlu(em)),
    ]
    bad = 0
    for ent, build in targets:
        try:
            rows = build()
        except FileNotFoundError as e:  # source snapshot not downloaded (e.g. gated XSTest)
            bad += 1
            print(f"MISSING {ent['path']}  source not in HF cache: {os.path.relpath(str(e), C.HUB) if C.HUB in str(e) else e}")
            continue
        got = write_jsonl(os.path.join(a.out_root, ent["path"]), rows)
        ok = got == ent["sha256"]
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {ent['path']}  rows={len(rows)}  sha256={got[:16]}  expected={ent['sha256'][:16]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
