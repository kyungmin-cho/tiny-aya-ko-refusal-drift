"""Render the result tables used in README / article from the aggregate JSON files (no number is typed by hand).

Inputs (defaults are the files in this repo):
  --e5-supplement  experiments/e5/agg/e5_supplement.json   run-2 judge / MMLU numbers. Point it at the file you rebuilt
                   from the released labels (README "Reproduce A") to regenerate those columns yourself.
  experiments/e5/agg/alpha_selection.json   α rule outcome (H/B/M also appear in e5_supplement)
  experiments/e5/agg/e5_degeneracy.json     output-shape statistics (computed from generations, not from labels)
  experiments/e4/agg/e4_agg.json, e4_review_sol.json   run 1 (its refusal-phrase columns need generations)
Usage: python3 -I -B tools/make_tables.py [--e5-supplement out/e5_supplement.json] > RESULTS.md
"""
import argparse, json, os

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda p: json.load(open(p if os.path.isabs(p) or os.path.exists(p) else os.path.join(R, p), encoding="utf-8"))
_ap = argparse.ArgumentParser()
_ap.add_argument("--e5-supplement", default=os.path.join(R, "experiments/e5/agg/e5_supplement.json"))
_args = _ap.parse_args()
DEG = J("experiments/e5/agg/e5_degeneracy.json")
E4, REV, E5, SEL = J("experiments/e4/agg/e4_agg.json"), J("experiments/e4/agg/e4_review_sol.json"), J(_args.e5_supplement), J("experiments/e5/agg/alpha_selection.json")


def pct(k, n):
    return f"{int(1000 * k / n + 0.5) / 10:.1f}%"  # round half up, as in the run reports


def kn(x):
    return f"{x['k']}/{x['n']} ({pct(x['k'], x['n'])})"


def pval(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def d(c):
    return f"{c['diff_pp']:+.1f} pp (p={pval(c['mcnemar_p'])})"


def xs(model):
    return kn(model["xstest"]["xstest_safe"])


def run1():
    m, c = E4["models"], E4["comparisons"]
    return [("Run 1 (seed 20261111)", "base (Tiny Aya Global)", kn(m["base"]["mj_judge_refusal"]), "—", "—", xs(m["base"]), "not measured"),
            ("", "C100 (benign control)", kn(m["C100_ep3"]["mj_judge_refusal"]), d(c["C100_ep3 vs base"]["mj_judge_refusal"]), "—", xs(m["C100_ep3"]), "not measured"),
            ("", "T100 (100 harmful rows)", kn(m["T100_ep3"]["mj_judge_refusal"]), d(c["T100_ep3 vs base"]["mj_judge_refusal"]),
             d(c["T100_ep3 vs C100_ep3"]["mj_judge_refusal"]), xs(m["T100_ep3"]), "not measured")]


def run2():
    m, c = E5["models"], E5["comparisons"]
    out = []
    for tag, name in (("base", "base (Tiny Aya Global)"), ("C100_ep3", "C100 (benign control)"), ("T100_ep3", "T100 (100 harmful rows)")):
        vb = d(c[f"base → {tag}"]["e1_refusal"]) if tag != "base" else "—"
        vc = d(c["C100_ep3 → T100_ep3"]["e1_refusal"]) if tag == "T100_ep3" else "—"
        out.append(("Run 2 (seed 20261113, re-curated pool)" if tag == "base" else "", name, kn(m[tag]["e1"]["refusal"]), vb, vc, kn(m[tag]["xstest_safe"]), kn(m[tag]["mmlu_rep"])))
    return out


def table_drift():
    h = ["| run | model | MultiJail-ko refusal (judge) | vs base | vs control | XSTest-safe over-refusal | Global-MMLU-ko (400) |", "|---|---|---|---|---|---|---|"]
    return "\n".join(h + ["| " + " | ".join(r) + " |" for r in run1() + run2()])


def table_en():
    m, c = E5["models"], E5["comparisons"]
    h = ["| model (run 2) | MultiJail-en refusal | vs base | vs control |", "|---|---|---|---|"]
    rows = [f"| base | {kn(m['base']['mjen']['refusal'])} | — | — |",
            f"| C100 | {kn(m['C100_ep3']['mjen']['refusal'])} | {d(c['base → C100_ep3']['mjen_refusal'])} | — |",
            f"| T100 | {kn(m['T100_ep3']['mjen']['refusal'])} | {d(c['base → T100_ep3']['mjen_refusal'])} | {d(c['C100_ep3 → T100_ep3']['mjen_refusal'])} |"]
    return "\n".join(h + rows)


def table_resta():
    m, c = E5["models"], E5["comparisons"]
    h = ["| model (run 2) | ko refusal (judge) | en refusal (judge) | ko judge-Unsafe | XSTest-safe over-refusal | MMLU-ko report 400 | MMLU-ko vs base | selection gate (H / B / M) | eligible |",
         "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for tag, name in [("base", "base"), ("C100_ep3", "C100"), ("T100_ep3", "T100 (drifted)")] + [(f"T100_ep3_resta{a}", f"T100 + ReSta α={a}") for a in ("0.25", "0.5", "0.75", "1.0")]:
        x = m[tag]
        mm = d(c[f"base → {tag}"]["mmlu_rep_correct"]) if tag != "base" and "mmlu_rep_correct" in c.get(f"base → {tag}", {}) else "—"
        a = tag.replace("T100_ep3_resta", "")
        gate = SEL["candidates"].get(a) if tag.startswith("T100_ep3_resta") else (SEL["base"] if tag == "base" else None)
        g = f"{gate['H']} / {gate['B']} / {gate['M']}" if gate else "—"
        el = ("yes" if a in [str(e) for e in SEL["eligible"]] else "no") if tag.startswith("T100_ep3_resta") else "—"
        rows.append(f"| {name} | {kn(x['e1']['refusal'])} | {kn(x['mjen']['refusal'])} | {kn(x['e1']['unsafe'])} | {kn(x['xstest_safe'])} | {kn(x['mmlu_rep'])} | {mm} | {g} | {el} |")
    return "\n".join(h + rows)


def table_review():
    h = ["| model (run 1) | judge Unsafe | new Unsafe (base Safe → Unsafe) | agent read-through: harmful / ambiguous / false positive | harmful share of new Unsafe |", "|---|---|---|---|---|"]
    rows = []
    for tag in ("T100_ep3", "T200_ep3", "C100_ep3"):
        x = REV["models"][tag]; rc = x["review_counts"]
        s = x["confirmed_share_of_new"]
        rows.append(f"| {tag.replace('_ep3', '')} | {x['judge_unsafe']['k']}/315 | {x['target']} | {rc.get('실질 유해', 0)} / {rc.get('애매', 0)} / {rc.get('오판정', 0)} | {s['k']}/{s['n']} ({pct(s['k'], s['n'])}) |")
    return "\n".join(h + rows)


def table_resta_run1():
    c = E4["comparisons"]
    h = ["| run 1 repair | ko refusal before → after | change | XSTest-safe over-refusal change |", "|---|---|---|---|"]
    rows = []
    for arm in ("T100", "T200"):
        for a in ("0.1", "0.25"):
            x = c[f"ReSta{a}({arm}_ep3) vs {arm}_ep3"]
            r = x["mj_judge_refusal"]
            xs_txt = d(x["xstest_safe_refusal"])
            rows.append(f"| {arm} + α={a} | {r['before_k']} → {r['after_k']} /315 | {d(r)} | {xs_txt} |")
    return "\n".join(h + rows)


def table_degeneracy():
    h = ["| model (run 2), MultiJail-ko answers | mean Hangul fraction | mean length (chars) | largest group of identical answers | mean 8-gram repetition | hit 128-token cap |",
         "|---|---|---|---|---|---|"]
    rows = []
    for tag, name in [("base", "base"), ("C100_ep3", "C100"), ("T100_ep3", "T100")] + [(f"T100_ep3_resta{a}", f"ReSta α={a}") for a in ("0.25", "0.5", "0.75", "1.0")]:
        x = DEG[tag]["e1"]
        rows.append(f"| {name} | {x['mean_hangul_frac']:.3f} | {x['mean_chars']:.0f} | {x['top_identical']}/{x['n']} | {x['mean_repeat8']:.3f} | {x['truncated']}/{x['n']} |")
    return "\n".join(h + rows)


if __name__ == "__main__":
    print("# Result tables (generated by tools/make_tables.py — do not edit by hand)\n")
    print("All refusal / Unsafe numbers are Qwen3Guard-Gen-4B labels on greedy 128-token generations. p = exact two-sided McNemar on the same items.\n")
    print("## Drift: two runs\n"); print(table_drift()); print()
    print("## Drift in English (run 2 only)\n"); print(table_en()); print()
    print("## ReSta with the base→global vector (run 2)\n"); print(table_resta()); print()
    print("Selection gate (fixed before run 2): eligible if M ≥ M(base) − 10 and B ≤ B(base) + 10 on held-out sets "
          "(H = AdvBench-ko refusals /100, B = benign kullm refusals /100, M = MMLU-ko selection split correct /200).\n")
    print("## Output-shape checks that do not use the judge (run 2)\n"); print(table_degeneracy()); print()
    print("Computed from the generations by experiments/e5/scripts/e5_degeneracy.py; only these numbers are released, not the text.\n")
    print("## ReSta in run 1 (different seed)\n"); print(table_resta_run1()); print()
    print("## Judge labels vs agent read-through (run 1)\n"); print(table_review()); print()
