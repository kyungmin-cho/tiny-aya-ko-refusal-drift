"""Re-derive the E6 numbers (paper-style safety vector, gamma 0.5 / 1.0) and the run-2+ agent read-through counts
from the released label dataset only, then compare them with the shipped aggregates in experiments/{e5,e6}/agg/.

Reads only: judge labels (id, safety, refusal), MMLU predictions, generation lengths (truncated flag), item indexes
(id -> cond / mj_id / xs_type) and agent labels (id, label). No prompt or answer text exists in these files.
Standard library only (plus experiments/e5/scripts/e5_supplement.py and scripts/e4_common.py from this repository).

What it recomputes and checks
  judge  : e6_agg.json  models / comparisons / new_unsafe (same functions as run 2: e5_supplement.load/summarize/compare)
  agent  : e6_sol_review.json  refusal read-through counts (base, T100, alpha 0.25, gamma 0.5, gamma 1.0), pure recovery
           with the paired bootstrap (seed 20261112, 10,000 resamples), new-Unsafe read-through, XSTest over-refusal labels
           e5_sol_review.json / e5_sol_review_base.json  run-2 new-Unsafe read-through and alpha 0.25 recovery
           e5_sol_refusal_shape*.json / e6_sol_refusal_shape.json  truncated answers per read-through label
  Not recomputable from labels (aggregate only): output shape (Hangul share, identical answers), weight statistics
  (e6_vecstat.json), the response language of read answers (`lang`) and the reader's question-harmfulness tag on XSTest.

Usage (repository root):
  python3 -I -B experiments/e6/scripts/e6_from_labels.py <dataset>/data [--agg-root experiments] [--out out/e6_from_labels.json]
Exit code 0 only if every compared value matches.
"""
import argparse, collections, json, math, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "experiments", "e5", "scripts"))
import e5_supplement as S  # noqa: E402
C = S.C   # scripts/e4_common.py

N = 315
Z = 1.959963984540054
E5_TAGS = ("base", "T100_ep3", "C100_ep3")
E6_TAGS = ("T100_paper_g0.5", "T100_paper_g1.0")
# released English label -> label name used inside the aggregate files (one-to-one, see tools/export_labels.py)
REFUSAL = {"true_refusal": "진짜 거절", "warning_redirect": "경고·방향전환", "not_refusal_harmless": "거절 아님(무해 답변)",
           "empty_talk": "빈말", "language_collapse": "언어 붕괴", "broken_output": "깨진 출력", "complies": "우회 답변"}
UNSAFE = {"harmful": "실질 유해", "ambiguous": "애매", "false_positive": "오판정"}
JUNK = ("empty_talk", "language_collapse", "broken_output")


def rows(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def wilson(k, n):
    if n == 0:
        return None
    p = k / n; d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d; h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [max(0.0, round(100 * (c - h), 2)), round(100 * (c + h), 2)]


def rate(k, n=N):
    return {"k": k, "n": n, "pct": round(100 * k / n, 2) if n else None, "wilson95_pct": wilson(k, n)}


class Cmp:
    """Compare recomputed values with a shipped aggregate, leaf by leaf (only keys present in the recomputed object)."""
    def __init__(self):
        self.n, self.bad = 0, []

    def __call__(self, name, got, ref, path=""):
        if isinstance(got, dict):
            if not isinstance(ref, dict):
                self.bad.append(f"{name}{path}: shipped value is not an object"); return
            for k, v in got.items():
                if k not in ref:
                    self.bad.append(f"{name}{path}.{k}: missing in shipped file"); continue
                self(name, v, ref[k], f"{path}.{k}")
        elif isinstance(got, float) and isinstance(ref, (int, float)):
            self.n += 1
            if not math.isclose(got, ref, rel_tol=1e-12, abs_tol=1e-12):
                self.bad.append(f"{name}{path}: recomputed {got} vs shipped {ref}")
        else:
            self.n += 1
            if got != ref:
                self.bad.append(f"{name}{path}: recomputed {got!r:.80} vs shipped {ref!r:.80}")


def load_review(d, names, mapping):
    lab = {}
    for n in names:
        for r in rows(os.path.join(d, f"{n}.jsonl")):
            assert r["id"] not in lab, f"duplicate id in {n}"
            assert r["label"] in mapping, f"unknown label in {n}"
            lab[r["id"]] = r["label"]
    return lab


def judge(ev, tag, part="e1"):
    return {r["id"]: r for r in rows(os.path.join(ev, tag, f"{part}_judge.jsonl"))}


def recovery(per_id, all_ids, target, labs):
    """(target - T100) / (base - T100) on the same 315 items, paired bootstrap exactly as in the original count script."""
    vec = lambda cn: [1 if per_id[cn].get(i) in labs else 0 for i in all_ids]
    b, t, a = vec("base"), vec("T100_ep3"), vec(target)
    B, T, A = sum(b), sum(t), sum(a)
    pt = (A - T) / (B - T) if B != T else None
    rng = random.Random(20261112); bs = []
    idx = range(N)
    for _ in range(10000):
        s = [rng.choice(idx) for _ in idx]
        Bb, Tb, Ab = (sum(v[k] for k in s) for v in (b, t, a))
        if Bb != Tb:
            bs.append((Ab - Tb) / (Bb - Tb))
    bs.sort()
    q = lambda f: round(100 * bs[int(f * (len(bs) - 1))], 1)
    return {"base": B, "T100": T, "target_k": A, "recovery_pct": round(100 * pt, 1) if pt is not None else None,
            "paired_bootstrap95_pct": [q(.025), q(.975)]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", help="the dataset's data/ folder (contains e5/, e6/)")
    ap.add_argument("--agg-root", default=os.path.join(REPO, "experiments"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    L, AG = a.labels, a.agg_root
    ev5, ev6 = os.path.join(L, "e5", "eval"), os.path.join(L, "e6", "eval")
    rv5, rv6 = os.path.join(L, "e5", "review"), os.path.join(L, "e6", "review")
    agg = lambda run, fn: json.load(open(os.path.join(AG, run, "agg", fn), encoding="utf-8"))
    cmp = Cmp()
    out = {}

    # ---------- 1. judge labels: e6_agg.json ----------
    xs_cond = S.id_field(os.path.join(L, "e5", "items", "xs_index.jsonl"), "cond")
    xs_type = S.id_field(os.path.join(L, "e5", "items", "xs_index.jsonl"), "xs_type")
    mj = dict(e1=S.id_field(os.path.join(L, "e5", "items", "e1_index.jsonl"), "mj_id"),
              mjen=S.id_field(os.path.join(L, "e5", "items", "mjen_index.jsonl"), "mj_id"))
    data = {t: S.load(ev5, t, xs_cond) for t in E5_TAGS}
    data.update({t: S.load(ev6, t, xs_cond) for t in E6_TAGS})
    assert all(v is not None for v in data.values()), "missing model folders in the label dataset"
    res = dict(models={t: S.summarize(v, xs_cond) for t, v in data.items()}, comparisons={}, new_unsafe={})
    pairs = [(x, t) for t in E6_TAGS for x in E5_TAGS] + [(E6_TAGS[0], E6_TAGS[1])]
    for x, y in pairs:
        res["comparisons"][f"{x} → {y}"] = dict(before=x, after=y, **S.compare(data[x], data[y], xs_cond))
    for t in E6_TAGS:
        res["new_unsafe"][t] = {p: S.new_unsafe(data["base"], data[t], p, mj[p]) for p in ("e1", "mjen")}
    cmp("e6_agg.json", res, agg("e6", "e6_agg.json"))
    out["e6_agg"] = res

    # ---------- 2. refusal read-through (MultiJail-ko answers the judge called refusals) ----------
    conds = {  # name: (eval dir, model, review dir, files)
        "base": (ev5, "base", rv5, ["sol_refusal_base_part1", "sol_refusal_base_part2"]),
        "T100_ep3": (ev5, "T100_ep3", rv5, ["sol_refusal_T100_ep3"]),
        "alpha0.25": (ev5, "T100_ep3_resta0.25", rv5, ["sol_refusal_T100_ep3_resta0.25"]),
        "gamma0.5": (ev6, "T100_paper_g0.5", rv6, ["sol_refusal_g0.5_part1", "sol_refusal_g0.5_part2"]),
        "gamma1.0": (ev6, "T100_paper_g1.0", rv6, ["sol_refusal_g1.0_part1", "sol_refusal_g1.0_part2", "sol_refusal_g1.0_part3"]),
    }
    sol = {"refusal": {}, "pure_recovery": {}, "new_unsafe": {}}
    per_id, all_ids, shape = {}, None, {}
    for cn, (ev, model, rv, files) in conds.items():
        jd = judge(ev, model)
        ids = sorted(jd, key=lambda s: int(s.split("-")[1])); assert len(ids) == N
        all_ids = all_ids or ids; assert ids == all_ids
        jref = {i for i in ids if jd[i]["refusal"] is True}
        lab = load_review(rv, files, REFUSAL)
        assert set(lab) == jref, f"{cn}: read-through ids differ from the judge's refusals"
        cnt = collections.Counter(lab.values())
        per_id[cn] = lab
        tr, fr = cnt["true_refusal"], cnt["true_refusal"] + cnt["warning_redirect"]
        junk = sum(cnt[x] for x in JUNK)
        sol["refusal"][cn] = {"judge_refusal": rate(len(jref)), "target": len(jref), "reviewed": len(lab),
                              "counts": {REFUSAL[k]: cnt[k] for k in REFUSAL},
                              "true_refusal_over_315": rate(tr), "true_or_redirect_over_315": rate(fr), "junk_over_315": rate(junk),
                              "true_refusal_over_judge_refusal": rate(tr, len(jref)), "true_or_redirect_over_judge_refusal": rate(fr, len(jref))}
        trunc = {r["id"]: r["truncated"] for r in rows(os.path.join(ev, model, "e1_genmeta.jsonl"))}
        shape[cn] = {"n_target": len(lab), "shape_by_label": {REFUSAL[k]: {"n": cnt[k], "truncated": sum(trunc[i] for i, v in lab.items() if v == k)}
                                                              for k in REFUSAL if cnt[k]}}
    for cn in ("alpha0.25", "gamma0.5", "gamma1.0"):
        sol["pure_recovery"][cn] = {"true_refusal": recovery(per_id, all_ids, cn, {"true_refusal"}),
                                    "true_or_redirect": recovery(per_id, all_ids, cn, {"true_refusal", "warning_redirect"})}

    # ---------- 3. new-Unsafe read-through (base Safe -> model Unsafe) ----------
    nu_specs = {  # key in shipped file: (eval dir, model, part, review dir, file)
        "ko_g0.5": (ev6, "T100_paper_g0.5", "e1", rv6, "sol_unsafe_e1_g0.5"), "en_g0.5": (ev6, "T100_paper_g0.5", "mjen", rv6, "sol_unsafe_mjen_g0.5"),
        "ko_g1.0": (ev6, "T100_paper_g1.0", "e1", rv6, "sol_unsafe_e1_g1.0"), "en_g1.0": (ev6, "T100_paper_g1.0", "mjen", rv6, "sol_unsafe_mjen_g1.0"),
        "ko_T100": (ev5, "T100_ep3", "e1", rv5, "sol_unsafe_T100_ep3"), "en_T100": (ev5, "T100_ep3", "mjen", rv5, "sol_unsafe_mjen_T100_ep3"),
        "ko_C100": (ev5, "C100_ep3", "e1", rv5, "sol_unsafe_C100_ep3"), "en_C100": (ev5, "C100_ep3", "mjen", rv5, "sol_unsafe_mjen_C100_ep3"),
    }
    for key, (ev, model, part, rv, fn) in nu_specs.items():
        bj, mjd = judge(ev5, "base", part), judge(ev, model, part)
        new = {i for i in bj if bj[i]["safety"] == "Safe" and mjd[i]["safety"] == "Unsafe"}
        lab = load_review(rv, [fn], UNSAFE)
        assert set(lab) == new, f"{key}: read-through ids differ from the new-Unsafe set"
        cnt = collections.Counter(lab.values())
        sf = {i for i in new if mj[part].get(i) in C.SAME_FRAME_ALL}
        sol["new_unsafe"][key] = {"target": len(new), "reviewed": len(lab), "counts": {UNSAFE[k]: cnt[k] for k in UNSAFE},
                                  "same_frame_counts": dict(collections.Counter(UNSAFE[lab[i]] for i in sorted(sf))),
                                  "harmful_over_315": rate(cnt["harmful"]), "harmful_over_new": rate(cnt["harmful"], len(new))}

    # ---------- 4. XSTest safe prompts refused at gamma 1.0 ----------
    xj = judge(ev6, "T100_paper_g1.0", "xs")
    tgt = {i for i, r in xj.items() if xs_cond.get(i) == "xstest_safe" and r["refusal"] is True}
    lab = load_review(rv6, ["sol_xs_overrefusal_g1.0"], REFUSAL)
    assert set(lab) == tgt
    cnt = collections.Counter(lab.values())
    xs = {"target": len(tgt), "reviewed": len(lab), "counts": {REFUSAL[k]: cnt[k] for k in REFUSAL},
          "label_x_xs_type": {REFUSAL[k]: dict(collections.Counter(xs_type[i] for i, v in sorted(lab.items()) if v == k)) for k in REFUSAL if cnt[k]},
          "true_refusal_any_question": {"over_45": rate(cnt["true_refusal"], len(tgt)), "over_250": rate(cnt["true_refusal"], 250)}}

    # ---------- compare with the shipped aggregates ----------
    e6s = agg("e6", "e6_sol_review.json")
    for cn, v in sol["refusal"].items():
        cmp(f"e6_sol_review.json:refusal.{cn}", v, e6s["refusal"][cn])
    for cn, v in sol["pure_recovery"].items():
        cmp(f"e6_sol_review.json:pure_recovery.{cn}", v, e6s["pure_recovery"][cn])
    for key in ("ko_g0.5", "en_g0.5", "ko_g1.0", "en_g1.0"):
        cmp(f"e6_sol_review.json:new_unsafe.{key}", sol["new_unsafe"][key], e6s["new_unsafe"][key])
    cmp("e6_sol_review.json:xstest_g1.0", xs, e6s["xstest_g1.0"])
    e5s, e5b = agg("e5", "e5_sol_review.json"), agg("e5", "e5_sol_review_base.json")
    for key in ("ko_T100", "en_T100", "ko_C100", "en_C100"):
        cmp(f"e5_sol_review.json:new_unsafe.{key}", sol["new_unsafe"][key], e5s["new_unsafe"][key])
    r = sol["refusal"]["alpha0.25"]
    cmp("e5_sol_review.json:refusal_resta0.25", {k: r[k] for k in ("target", "reviewed", "counts", "true_refusal_over_315", "true_or_redirect_over_315",
                                                                    "true_refusal_over_judge_refusal")}, e5s["refusal_resta0.25"])
    for k in ("true_refusal", "true_or_redirect"):
        p = sol["pure_recovery"]["alpha0.25"][k]
        cmp(f"e5_sol_review_base.json:pure_recovery.{k}", {"base": p["base"], "T100": p["T100"], "alpha0.25": p["target_k"],
                                                            "recovery_pct": p["recovery_pct"], "paired_bootstrap95_pct": p["paired_bootstrap95_pct"]},
            e5b["pure_recovery"][k])
    sh_b, sh_a, sh_g = agg("e5", "e5_sol_refusal_shape_base.json"), agg("e5", "e5_sol_refusal_shape.json"), agg("e6", "e6_sol_refusal_shape.json")
    for cn, ref in (("base", sh_b["base"]), ("T100_ep3", sh_b["T100_ep3"]), ("gamma0.5", sh_g["gamma0.5"]), ("gamma1.0", sh_g["gamma1.0"])):
        cmp(f"refusal_shape.{cn}", shape[cn], ref)
    cmp("e5_sol_refusal_shape.json(alpha0.25)", {"shape_by_label": shape["alpha0.25"]["shape_by_label"]}, sh_a)

    out.update(sol=sol, xstest_g1_0=xs, shape_truncated=shape, compared_values=cmp.n, mismatches=cmp.bad)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    g = sol["refusal"]["gamma1.0"]["counts"]; pr = sol["pure_recovery"]
    print(f"E6 judge refusals (ko): base {res['models']['base']['e1']['refusal']['k']}, T100 {res['models']['T100_ep3']['e1']['refusal']['k']}, "
          f"g0.5 {res['models']['T100_paper_g0.5']['e1']['refusal']['k']}, g1.0 {res['models']['T100_paper_g1.0']['e1']['refusal']['k']}")
    print(f"read-through, g1.0: true refusal {g['진짜 거절']}, warning/redirect {g['경고·방향전환']}; "
          f"recovery alpha0.25 {pr['alpha0.25']['true_refusal']['recovery_pct']}% {pr['alpha0.25']['true_refusal']['paired_bootstrap95_pct']}, "
          f"g1.0 {pr['gamma1.0']['true_refusal']['recovery_pct']}% {pr['gamma1.0']['true_refusal']['paired_bootstrap95_pct']}")
    print(f"XSTest g1.0: {xs['counts']['진짜 거절']}/{xs['target']} true refusals")
    print(f"compared {cmp.n} values with the shipped aggregates: {len(cmp.bad)} mismatches")
    for b in cmp.bad[:30]:
        print("  MISMATCH", b)
    return 0 if not cmp.bad else 1


if __name__ == "__main__":
    sys.exit(main())
