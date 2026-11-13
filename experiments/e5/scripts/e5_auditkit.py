"""AuditKIT 비교 경로(E4 예행·E5 본평가 공용, 맥·포드 CPU, 응답 원문을 읽지 않음).

판정기 라벨 파일(eval/<tag>/<part>_judge.jsonl: id, safety, refusal)만 읽는다. 생성 파일(응답 원문)은 열지 않는다.
각 문항을 auditkit.Sample(input=id, actual_output=id, metadata=라벨)로 만들고, 우리 판정기 라벨을 그대로 점수로 내는 scorer를 쓴다.
  - refusal: Qwen3Guard-Gen-4B "Refusal: Yes"면 1(파싱 실패 None은 0, E4 집계와 같음)
  - unsafe : Qwen3Guard "Safety: Unsafe"면 1
  - correct: (Global-MMLU 전용) e5_eval.py가 쓴 정답 여부 1/0
  AuditKIT 기본 거절 키워드(영어)는 쓰지 않는다.
모델·부분마다 auditkit.evaluate(samples, model="precomputed", scorers=[...], cache=False)를 따로 부른다(모델마다 라벨이 달라 캐시 재생을 막음).
비교는 auditkit.compare(기준 RunResult, 대상 RunResult) = RunComparison: metric_deltas()(비율 차), significance(metric)(id로 짝지은 paired bootstrap, 1,000회, seed 42).
같은 짝을 e4_common.paired(정확 McNemar)로도 계산해 나란히 적는다. 두 p값은 방법이 다르다(bootstrap p의 하한은 약 1/1001).
AuditKIT 상태 파일은 XDG_CACHE_HOME·XDG_DATA_HOME을 <out>/_ak_state로 돌려 그 안에만 쓰게 한다.
사용:
  python -I -B e5_auditkit.py <eval_root> <out_dir> --models base T100_ep3 ... --pairs base:T100_ep3 ... [--parts e1 xs_safe xs_unsafe ...] [--expect expect.json]
  parts: e1(MultiJail ko 315), mjen(MultiJail en 315), xs_safe(XSTest safe 250), xs_unsafe(XSTest unsafe 200), sel_h(α 선택셋 유해 100, 거절 높을수록), sel_b(선택셋 무해 100, 거절 낮을수록), mmlu_rep·mmlu_sel(Global-MMLU ko)
출력: <out_dir>/auditkit_compare.json, auditkit_compare.md (집계 숫자만)
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


PART_FILE = dict(e1="e1_judge.jsonl", mjen="mjen_judge.jsonl", xs_safe="xs_judge.jsonl", xs_unsafe="xs_judge.jsonl",
                 sel_h="sel_judge.jsonl", sel_b="sel_judge.jsonl", mmlu_rep="mmlu.jsonl", mmlu_sel="mmlu.jsonl")
# 지표 방향: 유해 요청 거절은 높을수록, XSTest safe 거절(과잉거절)은 낮을수록, Unsafe는 낮을수록, 정답은 높을수록 좋음
PART_METRICS = dict(e1=("refusal", "unsafe"), mjen=("refusal", "unsafe"), sel_h=("refusal", "unsafe"), sel_b=("refusal",),
                    xs_safe=("refusal",), xs_unsafe=("refusal",), mmlu_rep=("correct",), mmlu_sel=("correct",))


def load_labels(eval_root, tag, part, xs_cond=None):
    """라벨만 읽는다. xs의 safe/unsafe 구분은 판정 파일에 cond가 없으므로 XSTest 입력 파일(xstest_items.jsonl, 프롬프트 열은 쓰지 않음)의 cond로 나눈다."""
    p = os.path.join(eval_root, tag, PART_FILE[part])
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    out = {}
    for r in rows:
        if part.startswith("mmlu"):
            if r.get("split") != part.split("_")[1]:
                continue
            out[r["id"]] = dict(correct=1.0 if r.get("correct") else 0.0)
            continue
        if part.startswith("xs_") and xs_cond.get(r["id"]) != ("xstest_safe" if part == "xs_safe" else "xstest_unsafe"):
            continue
        if part.startswith("sel_") and not r["id"].startswith("selh-" if part == "sel_h" else "selb-"):
            continue
        out[r["id"]] = dict(refusal=1.0 if r.get("refusal") is True else 0.0, unsafe=1.0 if r.get("safety") == "Unsafe" else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_root"); ap.add_argument("out_dir")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="기준:대상")
    ap.add_argument("--parts", nargs="+", default=["e1", "xs_safe", "xs_unsafe"])
    ap.add_argument("--xs-items", default=None, help="XSTest 입력 파일(cond 열만 씀). 기본: <proj>/experiments/e1e3/xstest_items.jsonl")
    ap.add_argument("--expect", default=None, help="{tag: {part: {metric: k}}} 기대 건수(예: E4 agg). 다르면 종료 코드 3")
    ap.add_argument("--n-resamples", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    st = os.path.join(os.path.abspath(a.out_dir), "_ak_state")
    os.environ["XDG_CACHE_HOME"] = os.path.join(st, "cache"); os.environ["XDG_DATA_HOME"] = os.path.join(st, "data")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    proj = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # <proj>/experiments/<run>/scripts
    sys.path.insert(0, os.path.join(proj, "scripts"))
    import e4_common as C
    import auditkit as ak
    from auditkit.sample import Sample
    from auditkit.scorers import scorer
    from auditkit.types import Direction

    xs_items = a.xs_items or os.path.join(proj, "experiments/e1e3/xstest_items.jsonl")
    xs_cond = {}
    if any(p.startswith("xs_") for p in a.parts):
        for l in open(xs_items, encoding="utf-8"):
            r = json.loads(l); xs_cond[r["id"]] = r["cond"]  # 프롬프트 열은 읽지만 쓰지 않는다
    mk = lambda m, d: scorer(lambda s, o: s.metadata[m], direction=d, name=m)
    SC = dict(refusal_max=mk("refusal", Direction.MAXIMIZE), refusal_min=mk("refusal", Direction.MINIMIZE),
              unsafe=mk("unsafe", Direction.MINIMIZE), correct=mk("correct", Direction.MAXIMIZE))

    def scorers_for(part):
        out = []
        for m in PART_METRICS[part]:
            out.append(SC["correct"] if m == "correct" else SC["unsafe"] if m == "unsafe" else SC["refusal_min" if part in ("xs_safe", "sel_b") else "refusal_max"])
        return out

    t0 = time.time()
    labels, runs, rates, missing = {}, {}, {}, []
    for tag in a.models:
        for part in a.parts:
            try:
                lab = load_labels(a.eval_root, tag, part, xs_cond)
            except FileNotFoundError:
                missing.append(f"{tag}/{part}"); continue
            labels[(tag, part)] = lab
            samples = [Sample(input=i, id=i, task=part, actual_output=i, metadata=dict(lab[i])) for i in sorted(lab)]
            rr = ak.evaluate(samples, model="precomputed", scorers=scorers_for(part), cache=False)
            runs[(tag, part)] = rr
            rates.setdefault(tag, {})[part] = dict(n=len(lab), **{m: dict(k=int(sum(v[m] for v in lab.values())),
                                                                         auditkit_mean=round(rr.headline.get(m, float("nan")), 6),
                                                                         pct=round(100 * sum(v[m] for v in lab.values()) / len(lab), 2) if lab else None,
                                                                         wilson=C.wilson(int(sum(v[m] for v in lab.values())), len(lab)))
                                                       for m in PART_METRICS[part]})
    comps = {}
    for pr in a.pairs:
        b, c = pr.split(":")
        for part in a.parts:
            if (b, part) not in runs or (c, part) not in runs:
                continue
            rc = ak.compare(runs[(b, part)], runs[(c, part)])
            md = {d.metric: d for d in rc.metric_deltas()}
            for m in PART_METRICS[part]:
                lb, lc = labels[(b, part)], labels[(c, part)]
                ours = C.paired({i: bool(v[m]) for i, v in lb.items()}, {i: bool(v[m]) for i, v in lc.items()})
                from auditkit._bootstrap import paired_bootstrap, score_pairs
                sig = paired_bootstrap(score_pairs(runs[(b, part)], runs[(c, part)], m), n_resamples=a.n_resamples, seed=42)
                d = md.get(m)
                comps[f"{c} vs {b} | {part} | {m}"] = dict(
                    baseline=b, candidate=c, part=part, metric=m, n=ours["n"],
                    ours=dict(before_k=ours["before_k"], after_k=ours["after_k"], diff_pp=ours["diff_pp"], mcnemar_exact_p=ours["mcnemar_p"],
                              b_true_to_false=ours["b_true_to_false"], c_false_to_true=ours["c_false_to_true"]),
                    auditkit=dict(baseline=d.baseline if d else None, candidate=d.candidate if d else None, delta=d.delta if d else None,
                                  grade=str(d.grade.value if hasattr(d.grade, "value") else d.grade) if d else None, direction=d.direction if d else None,
                                  bootstrap_n=sig.get("n"), bootstrap_delta=sig.get("delta"), bootstrap_p=sig.get("p_value"), significant=sig.get("significant")),
                    rates_match=bool(d) and abs(d.baseline - ours["before_k"] / ours["n"]) < 1e-9 and abs(d.candidate - ours["after_k"] / ours["n"]) < 1e-9)
    bad = []
    if a.expect:
        exp = json.load(open(a.expect))
        for tag, parts in ((t, p) for t, p in exp.items() if not t.startswith("_")):
            for part, ms in parts.items():
                for m, k in ms.items():
                    got = rates.get(tag, {}).get(part, {}).get(m, {}).get("k")
                    if got != k:
                        bad.append(dict(tag=tag, part=part, metric=m, expect=k, got=got))
    distinct = len({json.dumps(rates[t].get("e1", {}).get("refusal", {}).get("k")) for t in rates}) if rates else 0
    res = dict(_note="판정기 라벨만으로 만든 AuditKIT precomputed 비교. 응답 원문 없음", created_by=os.path.relpath(os.path.abspath(__file__), proj),
               created_kst=time.strftime("%Y-%m-%dT%H:%M:%S+09:00"), auditkit_version=ak.__version__, eval_root=os.path.relpath(os.path.abspath(a.eval_root), proj),
               api="auditkit.evaluate(samples, model='precomputed', scorers=[@scorer(direction=...)], cache=False) → auditkit.compare(a, b).metric_deltas(); "
                   "auditkit._bootstrap.paired_bootstrap(score_pairs(a, b, metric), n_resamples, seed=42) = RunComparison.significance()와 같은 함수(재표집 수만 명시)",
               rates=rates, comparisons=comps, missing=missing, expect_mismatch=bad, distinct_e1_refusal_counts=distinct,
               all_rates_match=all(v["rates_match"] for v in comps.values()), sec=round(time.time() - t0, 1))
    json.dump(res, open(os.path.join(a.out_dir, "auditkit_compare.json"), "w"), ensure_ascii=False, indent=1)
    L = ["| 비교 | 부분 | 지표 | 기준 k/n | 대상 k/n | 차(%p) | McNemar p(정확) | AuditKIT 차 | AuditKIT bootstrap p | 비율 일치 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in comps.items():
        o, q = v["ours"], v["auditkit"]
        L.append(f"| {v['candidate']} vs {v['baseline']} | {v['part']} | {v['metric']} | {o['before_k']}/{v['n']} | {o['after_k']}/{v['n']} | {o['diff_pp']} | {o['mcnemar_exact_p']:.2g} | "
                 f"{q['delta']:+.4f} | {q['bootstrap_p']} | {v['rates_match']} |")
    open(os.path.join(a.out_dir, "auditkit_compare.md"), "w").write("\n".join(L) + "\n")
    print(json.dumps(dict(all_rates_match=res["all_rates_match"], expect_mismatch=bad, missing=missing,
                          e1_refusal={t: rates[t].get("e1", {}).get("refusal", {}).get("k") for t in rates}, sec=res["sec"]), ensure_ascii=False))
    if bad or not res["all_rates_match"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
