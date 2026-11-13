"""RESULT.md 맨 위 표용 추가 짝 비교(맥): 모든 팔을 base·같은 크기·같은 에포크 대조군과 비교하고, Wilson 구간을 붙인다.
e4_aggregate.py의 compare()를 그대로 쓴다(새 통계 없음). ReSta는 대조군 = 같은 크기의 C*_ep3.
사용: python3 -I scripts/e4_extra_compare.py <e4_root>   출력: agg/e4_main_table.json, agg/e4_main_table.md
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "scripts"))
import e4_common as C  # noqa: E402
import e4_aggregate as A  # noqa: E402


def ctrl_of(tag):
    if tag == "base" or tag.startswith("C"):
        return None
    size = tag[1:4]; ep = "ep1" if "_ep1" in tag else "ep3"
    return f"C{size}_{ep}"


def main(root):
    ev = os.path.join(root, "eval")
    data = {t: A.load_tag(os.path.join(ev, t)) for t in sorted(os.listdir(ev))}
    data = {t: v for t, v in data.items() if A.complete(v)}
    order = [t for t in ("base", "T100_ep1", "T100_ep3", "C100_ep1", "C100_ep3", "T100_ep3_resta0.1", "T100_ep3_resta0.25",
                         "T200_ep1", "T200_ep3", "C200_ep1", "C200_ep3", "T200_ep3_resta0.1", "T200_ep3_resta0.25") if t in data]
    rows = {}
    for t in order:
        s = A.summarize(data[t]); r = dict(summary=s)
        if t != "base":
            r["vs_base"] = A.compare(data["base"], data[t])
        c = ctrl_of(t)
        if c and c in data:
            r["control"] = c; r["vs_control"] = A.compare(data[c], data[t])
        rows[t] = r
    json.dump(rows, open(os.path.join(root, "agg", "e4_main_table.json"), "w"), ensure_ascii=False, indent=1)
    L = ["| 팔 | MJ 거절(판정기) [Wilson 95%] | 학습 전 대비 차(%p) · McNemar p | 대조군 대비 차(%p) · p | 거절 문구 | 판정기 Unsafe | XSTest safe 과잉거절 [Wilson] | XSTest unsafe 거절 |",
         "|---|---|---|---|---|---|---|---|"]
    for t, r in rows.items():
        s = r["summary"]; m = s["mj_judge_refusal"]; xs = s["xstest"]["xstest_safe"]; xu = s["xstest"]["xstest_unsafe"]
        vb = r.get("vs_base"); vc = r.get("vs_control")
        fb = f"{vb['mj_judge_refusal']['diff_pp']:+} · {A.fmt_p(vb['mj_judge_refusal']['mcnemar_p'])}" if vb else "—"
        fc = f"{vc['mj_judge_refusal']['diff_pp']:+} · {A.fmt_p(vc['mj_judge_refusal']['mcnemar_p'])} (대 {r['control']})" if vc else "—"
        L.append(f"| {t} | {m['k']}/{m['n']} ({m['pct']}%) {m['wilson']} | {fb} | {fc} | {s['mj_keyword_refusal']['k']}/315 | {s['mj_unsafe']['k']}/315 | "
                 f"{xs['k']}/{xs['n']} ({xs['pct']}%) {xs['wilson']} | {xu['k']}/{xu['n']} |")
    open(os.path.join(root, "agg", "e4_main_table.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
