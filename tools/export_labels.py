"""Export the public label dataset by *projecting* private run files onto an allow-list of fields.

No prompt, response or free-text reason is ever copied: each output row is rebuilt from the whitelisted keys below,
so an unexpected column in a source file cannot leak. Prints only file names, row counts and the key names seen.

Output layout (mirrors the run tree, so the aggregation scripts run on it unchanged):
  <out>/<run>/eval/<tag>/{e1,mjen,xs,sel}_judge.jsonl   id, safety, refusal, categories      (Qwen3Guard-Gen-4B labels)
  <out>/<run>/eval/<tag>/{part}_genmeta.jsonl           id, n_new_tokens, truncated          (generation length only)
  <out>/<run>/eval/<tag>/mmlu.jsonl                     id, split, answer, pred, correct, letter_logits
  <out>/<run>/items/{part}_index.jsonl                  id + item metadata (dataset row ids, categories), no text
  <out>/e4/review/{reviewer}_<tag>.jsonl                id, label                            (agent read-through labels)
  <out>/<run>/review/sol_{unsafe,refusal,xs_overrefusal}_<name>.jsonl
                                                        id, label                            (run 2 and later, --sol)
Runs named with --slim (E6, E7, E8) reuse the run-2 evaluation items, so no items/ or train/ is written for them, and
their DONE.json / model_info.json are rebuilt from a small allow-list (tag, kind, gamma, timings): the full files hold
paths and hashes of adapters and merged weights that are not released.

Usage:
  python3 -I -B tools/export_labels.py --run e4=<private e4 root> --run e5=<private e5 root> \
      --items <private e1e3 dir> --review-dir <private e4 samples dir> --out <dataset dir>
  # E6/E7 judge labels and run-2+ agent labels only:
  python3 -I -B tools/export_labels.py --run e6=<private e6 root> --run e7=<private e7 root> --slim e6 e7 \
      --sol e5=<private e5 samples dir> --sol e6=<private e6 samples dir> --out <dataset dir>
  # E8 (the same vector on the benign control C100 and on T100) judge labels:
  python3 -I -B tools/export_labels.py --run e8=<private e8 root> --slim e8 --out <dataset dir>
"""
import argparse, glob, json, os, sys

JUDGE = ("id", "safety", "refusal", "categories")
GENMETA = ("id", "n_new_tokens", "truncated")
MMLU = ("id", "split", "answer", "pred", "correct", "letter_logits")
REVIEW = ("id", "label")
INDEX = {
    "e1": ("id", "cond", "mj_id", "tags", "source"),
    "mjen": ("id", "cond", "mj_id", "tags", "source"),
    "xs": ("id", "cond", "xs_type", "xs_label"),   # xs_focus dropped: it is a span of the prompt text
    "sel": ("id", "cond", "adv_idx", "rid"),
    "mmlu": ("id", "split", "gm_idx", "sample_id", "subject", "subject_category", "cultural_sensitivity_label", "answer"),
}
META_FILES = ("DONE.json", "model_info.json")          # eval meta: counts, timings, revisions
TRAIN_FILES = ("DONE.json", "log_history.json", "saved_epochs.json")  # training meta and loss curve
# weight hashes/paths of non-released models (T100/T200 adapters, ReSta merges) are dropped so they cannot be mistaken for releases
DROP_UNLESS_RELEASED = ("saved", "adapter_sha256", "adapter_sha", "release_dir")
RELEASED = ("C100",)
LABELS = {"실질 유해": "harmful", "애매": "ambiguous", "오판정": "false_positive"}
# refusal read-through rubric (7 labels, first match wins in the order below); one-to-one English names
REFUSAL_LABELS = {"우회 답변": "complies", "깨진 출력": "broken_output", "언어 붕괴": "language_collapse", "빈말": "empty_talk",
                  "거절 아님(무해 답변)": "not_refusal_harmless", "경고·방향전환": "warning_redirect", "진짜 거절": "true_refusal"}
# private sol file prefix -> (public prefix, label map). Only id and label are written; reason/lang/question are dropped.
SOL_KINDS = (("sol_review_", "sol_unsafe_", LABELS), ("sol_refusal_ko_", "sol_refusal_", REFUSAL_LABELS),
             ("sol_xs_overrefusal_", "sol_xs_overrefusal_", REFUSAL_LABELS))
SLIM_META = ("tag", "started_kst", "finished_kst", "expect", "limit", "rehearsal", "model_build_sec", "gen_wall_sec",
             "mmlu_wall_sec", "judge_wall_sec")
SLIM_SPEC = ("kind", "gamma")
SLIM_EXTRA_FILES = ("repro_check.json",)   # counts only (identical answers / labels / MMLU predictions on a re-run)


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def project(rows, keys, rename=None):
    out = []
    for r in rows:
        o = {k: r[k] for k in keys if k in r}
        if rename:
            o = {k: rename.get(v, v) if k == "label" else v for k, v in o.items()}
        out.append(o)
    return out


def project_strict(rows, rename):
    """id + label only; every label must be in the map (an unknown label stops the export)."""
    out = []
    for r in rows:
        if r.get("label") not in rename:
            raise SystemExit(f"unknown label value in row id={r.get('id')!r} (value not shown)")
        out.append({"id": r["id"], "label": rename[r["label"]]})
    return out


RUN_DIRS = {"2026-11-11-e1e3": "e1e3", "2026-11-12-data-analysis": "data_analysis", "2026-11-12-e4": "e4", "2026-11-13-e5": "e5",
            "2026-11-12-e6": "e6", "2026-11-12-e7": "e7", "2026-11-13-e8": "e8"}


def strip_meta(x, keep_weights):
    if isinstance(x, str):   # private run-folder names -> public layout (experiments/<run>)
        for a, b in RUN_DIRS.items():
            x = x.replace(f"_runs/{a}", f"experiments/{b}")
        return x
    if isinstance(x, dict):
        return {k: strip_meta(v, keep_weights) for k, v in x.items() if keep_weights or k not in DROP_UNLESS_RELEASED}
    if isinstance(x, list):
        return [strip_meta(v, keep_weights) for v in x]
    return x


def slim_meta(x):
    o = {k: x[k] for k in SLIM_META if k in x}
    if isinstance(x.get("spec"), dict):
        o["spec"] = {k: x["spec"][k] for k in SLIM_SPEC if k in x["spec"]}
    return o


def write_json(path, obj, seen, src):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    seen.append((os.path.basename(src), os.path.relpath(path), 1))


def write(path, rows, seen, src):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    seen.append((os.path.basename(src), os.path.relpath(path), len(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=private_run_root (contains eval/<tag>/)")
    ap.add_argument("--items", default=None, help="private dir with e1_items.jsonl and xstest_items.jsonl")
    ap.add_argument("--e5-items", default=None, help="private dir with mjen/sel/mmlu_items.jsonl")
    ap.add_argument("--review-dir", default=None, help="private dir with sol_review_*.jsonl / agent_review_*.jsonl (E4)")
    ap.add_argument("--slim", nargs="*", default=[], help="run names that reuse run-2 items: no items/train, allow-listed meta")
    ap.add_argument("--sol", action="append", default=[], help="name=private samples dir with sol_*.jsonl (run 2 and later)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if any(name not in a.slim for name in (s.split("=", 1)[0] for s in a.run)) and not a.items:
        ap.error("--items is required for runs not listed in --slim")
    seen, keys_seen = [], {}
    for spec in a.run:
        name, root = spec.split("=", 1)
        slim = name in a.slim
        for d in sorted(glob.glob(os.path.join(root, "eval", "*"))):
            tag = os.path.basename(d)
            if not os.path.isdir(d):
                continue
            for p in sorted(glob.glob(os.path.join(d, "*_judge.jsonl"))):
                rows = read(p); keys_seen[os.path.basename(p)] = sorted({k for r in rows for k in r})
                write(os.path.join(a.out, name, "eval", tag, os.path.basename(p)), project(rows, JUDGE), seen, p)
            for p in sorted(glob.glob(os.path.join(d, "*_gen.jsonl"))):
                rows = read(p)
                write(os.path.join(a.out, name, "eval", tag, os.path.basename(p).replace("_gen.jsonl", "_genmeta.jsonl")),
                      project(rows, GENMETA), seen, p)
            p = os.path.join(d, "mmlu.jsonl")
            if os.path.exists(p):
                write(os.path.join(a.out, name, "eval", tag, "mmlu.jsonl"), project(read(p), MMLU), seen, p)
            for fn in META_FILES:
                p = os.path.join(d, fn)
                if os.path.exists(p):
                    meta = slim_meta(json.load(open(p))) if slim else strip_meta(json.load(open(p)), tag.startswith(RELEASED))
                    write_json(os.path.join(a.out, name, "eval", tag, fn), meta, seen, p)
            for fn in SLIM_EXTRA_FILES if slim else ():
                p = os.path.join(d, fn)
                if os.path.exists(p):
                    x = json.load(open(p))
                    write_json(os.path.join(a.out, name, "eval", tag, fn), {k: v for k, v in x.items() if isinstance(v, (int, float)) or k == "kst"}, seen, p)
        if slim:
            continue
        for d in sorted(glob.glob(os.path.join(root, "train", "*"))):
            arm = os.path.basename(d)
            for fn in TRAIN_FILES:
                p = os.path.join(d, fn)
                if os.path.exists(p):
                    write_json(os.path.join(a.out, name, "train", arm, fn), strip_meta(json.load(open(p)), arm.startswith(RELEASED)), seen, p)
        idx = {"e1": os.path.join(a.items, "e1_items.jsonl"), "xs": os.path.join(a.items, "xstest_items.jsonl")}
        if name == "e5" and a.e5_items:
            idx.update(mjen=os.path.join(a.e5_items, "mjen_items.jsonl"), sel=os.path.join(a.e5_items, "sel_items.jsonl"),
                       mmlu=os.path.join(a.e5_items, "mmlu_items.jsonl"))
        for part, p in idx.items():
            write(os.path.join(a.out, name, "items", f"{part}_index.jsonl"), project(read(p), INDEX[part]), seen, p)
    if a.review_dir:
        for pat, who in (("sol_review_*.jsonl", "sol"), ("agent_review_*.jsonl", "claude")):
            for p in sorted(glob.glob(os.path.join(a.review_dir, pat))):
                rows = read(p); keys_seen[os.path.basename(p)] = sorted({k for r in rows for k in r})
                tag = os.path.basename(p).split("_review_")[1].replace(".jsonl", "")
                write(os.path.join(a.out, "e4", "review", f"{who}_{tag}.jsonl"), project(rows, REVIEW, LABELS), seen, p)
    for spec in a.sol:
        name, sdir = spec.split("=", 1)
        for src_prefix, dst_prefix, labels in SOL_KINDS:
            for p in sorted(glob.glob(os.path.join(sdir, f"{src_prefix}*.jsonl"))):
                rows = read(p); keys_seen[f"{name}/{os.path.basename(p)}"] = sorted({k for r in rows for k in r})
                ids = [r["id"] for r in rows]
                if len(ids) != len(set(ids)):
                    raise SystemExit(f"duplicate ids in {os.path.basename(p)}")
                rest = os.path.basename(p)[len(src_prefix):]
                write(os.path.join(a.out, name, "review", dst_prefix + rest), project_strict(rows, labels), seen, p)
    print("source key names (values not shown):")
    for k, v in sorted(keys_seen.items()):
        print(f"  {k}: {v}")
    print(f"wrote {len(seen)} files, {sum(n for *_, n in seen)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
