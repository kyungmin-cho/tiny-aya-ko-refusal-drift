"""E5 보충 집계(표준 라이브러리, 맥·포드): 판정기 라벨(*_judge.jsonl의 id·safety·refusal)과 mmlu.jsonl만 읽는다. 응답 원문(gen)·samples는 열지 않는다.

e4_aggregate.py가 다루지 않는 것을 채운다(E5 본실행 중 추가, 2026-11-12).
  - MultiJail en 315(mjen) 거절률·Unsafe, Wilson 95% 구간, 짝 비교(정확 McNemar)
  - ReSta α 0.5·0.75·1.0 비교, C100 ep3 대비 비교(모든 모델)
  - Global-MMLU ko rep 400·sel 200 정확도와 짝 비교(McNemar)
  - 선택셋 H(selh 거절)·B(selb 거절)
  - 새로 생긴 Unsafe(base Safe → 모델 Unsafe) 건수·id·같은 틀 표시(MultiJail ko·en). 판독은 하지 않는다(미판독).
같은 틀 문항 표시는 문항 파일(e1_items·mjen_items)의 mj_id만 읽어 e4_common.SAME_FRAME_ALL과 맞춘다.
사용: python3 -I -B e5_supplement.py <eval_dir> <out_prefix> --items-e1 <e1_items.jsonl> --items-mjen <mjen_items.jsonl> --items-xs <xstest_items.jsonl> [--tags ...]
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(PROJ, "scripts"))
import e4_common as C  # noqa: E402

ALPHAS = ("0.25", "0.5", "0.75", "1.0")
TAGS = ["base", "T100_ep3", "C100_ep3"] + [f"T100_ep3_resta{a}" for a in ALPHAS]


def rj(p):
    return C.read_jsonl(p) if os.path.exists(p) else []


def id_field(path, field):
    """문항 파일에서 id와 한 칸만 꺼낸다(프롬프트는 쓰지 않음)."""
    out = {}
    for r in C.read_jsonl(path):
        out[r["id"]] = r.get(field)
    return out


def rate(k, n):
    return dict(k=k, n=n, pct=round(100 * k / n, 2) if n else None, wilson=C.wilson(k, n))


def load(ev, tag, xs_cond):
    d = os.path.join(ev, tag)
    if not os.path.isdir(d):
        return None
    out = dict(done=os.path.exists(os.path.join(d, "DONE.json")))
    for part in ("e1", "mjen", "xs", "sel"):
        out[part] = {r["id"]: r for r in rj(os.path.join(d, f"{part}_judge.jsonl"))}
    out["mmlu"] = {r["id"]: r for r in rj(os.path.join(d, "mmlu.jsonl"))}
    return out


def summarize(t, xs_cond):
    s = dict(done=t["done"])
    for part in ("e1", "mjen"):
        j = t[part]
        if not j:
            continue
        n = len(j)
        s[part] = dict(refusal=rate(sum(r.get("refusal") is True for r in j.values()), n),
                       unsafe=rate(sum(r.get("safety") == "Unsafe" for r in j.values()), n),
                       refusal_missing=sum(r.get("refusal") is None for r in j.values()),
                       safety_missing=sum(r.get("safety") is None for r in j.values()))
    if t["xs"]:
        for cond in ("xstest_safe", "xstest_unsafe"):
            ids = [i for i in t["xs"] if xs_cond.get(i) == cond]
            s[cond] = rate(sum(t["xs"][i].get("refusal") is True for i in ids), len(ids))
    if t["sel"]:
        h = [r for i, r in t["sel"].items() if i.startswith("selh-")]; b = [r for i, r in t["sel"].items() if i.startswith("selb-")]
        s["sel_H"] = rate(sum(r.get("refusal") is True for r in h), len(h))
        s["sel_B"] = rate(sum(r.get("refusal") is True for r in b), len(b))
    if t["mmlu"]:
        for sp in ("rep", "sel"):
            rows = [r for r in t["mmlu"].values() if r.get("split") == sp]
            s[f"mmlu_{sp}"] = rate(sum(bool(r.get("correct")) for r in rows), len(rows))
    return s


def compare(a, b, xs_cond):
    out = {}
    for part in ("e1", "mjen"):
        if a[part] and b[part] and set(a[part]) == set(b[part]):
            ref = lambda t: {i: r.get("refusal") is True for i, r in t[part].items()}
            uns = lambda t: {i: r.get("safety") == "Unsafe" for i, r in t[part].items()}
            out[f"{part}_refusal"] = C.paired(ref(a), ref(b)); out[f"{part}_unsafe"] = C.paired(uns(a), uns(b))
    if a["xs"] and b["xs"] and set(a["xs"]) == set(b["xs"]):
        ref = lambda t: {i: r.get("refusal") is True for i, r in t["xs"].items()}
        for cond in ("xstest_safe", "xstest_unsafe"):
            out[f"{cond}_refusal"] = C.paired(ref(a), ref(b), [i for i in sorted(a["xs"]) if xs_cond.get(i) == cond])
    if a["sel"] and b["sel"] and set(a["sel"]) == set(b["sel"]):
        ref = lambda t: {i: r.get("refusal") is True for i, r in t["sel"].items()}
        out["sel_H_refusal"] = C.paired(ref(a), ref(b), [i for i in sorted(a["sel"]) if i.startswith("selh-")])
        out["sel_B_refusal"] = C.paired(ref(a), ref(b), [i for i in sorted(a["sel"]) if i.startswith("selb-")])
    if a["mmlu"] and b["mmlu"] and set(a["mmlu"]) == set(b["mmlu"]):
        cor = lambda t: {i: bool(r.get("correct")) for i, r in t["mmlu"].items()}
        for sp in ("rep", "sel"):
            out[f"mmlu_{sp}_correct"] = C.paired(cor(a), cor(b), [i for i in sorted(a["mmlu"]) if a["mmlu"][i].get("split") == sp])
    return out


def new_unsafe(base, t, part, mj):
    ids = [i for i in sorted(set(base[part]) & set(t[part]), key=C._id_key)
           if base[part][i].get("safety") == "Safe" and t[part][i].get("safety") == "Unsafe"]
    sf = [i for i in ids if mj.get(i) in C.SAME_FRAME_ALL]
    return dict(k=len(ids), n=len(t[part]), same_frame_k=len(sf), ids=ids, same_frame_ids=sf, mj_ids=[mj.get(i) for i in ids],
                read_status="미판독(판독은 sol 몫)")


def fp(p):
    return "—" if p is None else (f"{p:.2g}" if p < 0.001 else f"{p:.3f}")


def md(res):
    L = ["# E5 보충 집계(자동 생성: scripts/e5_supplement.py, 판정기 라벨·MMLU 정답만 읽음)", "",
         "## 모델별", "",
         "| 모델 | MJ ko 거절 | MJ en 거절 | MJ ko Unsafe | MJ en Unsafe | XSTest safe 거절(과잉거절) | XSTest unsafe 거절 | MMLU rep | MMLU sel | sel H | sel B |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    g = lambda d: f"{d['k']}/{d['n']} ({d['pct']}%) {d['wilson']}" if d else "—"
    for tag, s in res["models"].items():
        L.append(f"| {tag} | {g(s.get('e1', {}).get('refusal'))} | {g(s.get('mjen', {}).get('refusal'))} | {g(s.get('e1', {}).get('unsafe'))} | "
                 f"{g(s.get('mjen', {}).get('unsafe'))} | {g(s.get('xstest_safe'))} | {g(s.get('xstest_unsafe'))} | {g(s.get('mmlu_rep'))} | "
                 f"{g(s.get('mmlu_sel'))} | {g(s.get('sel_H'))} | {g(s.get('sel_B'))} |")
    L += ["", "## 짝 비교(정확 McNemar, 차 = 대상 − 기준, %p)", "",
          "| 비교(기준→대상) | MJ ko 거절 | p | MJ en 거절 | p | XSTest safe 거절 | p | XSTest unsafe 거절 | p | MMLU rep 정답 | p | MMLU sel 정답 | p |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in res["comparisons"].items():
        cells = []
        for k in ("e1_refusal", "mjen_refusal", "xstest_safe_refusal", "xstest_unsafe_refusal", "mmlu_rep_correct", "mmlu_sel_correct"):
            p = c.get(k)
            cells += ([f"{p['before_k']}→{p['after_k']}/{p['n']} ({p['diff_pp']:+})", fp(p["mcnemar_p"])] if p else ["—", "—"])
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "## 새로 생긴 Unsafe(학습 전 base Safe → 모델 Unsafe, 미판독)", ""]
    for tag, v in res["new_unsafe"].items():
        for part, x in v.items():
            L.append(f"- {tag} {part}: {x['k']}/{x['n']}, 같은 틀 {x['same_frame_k']}건 (id는 json `new_unsafe.{tag}.{part}.ids`)")
    L += ["", f"- 빠진 모델: {res['missing'] or '없음'}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir"); ap.add_argument("out_prefix")
    ap.add_argument("--items-e1", required=True); ap.add_argument("--items-mjen", default=None); ap.add_argument("--items-xs", required=True)
    ap.add_argument("--tags", nargs="*", default=TAGS)
    a = ap.parse_args()
    xs_cond = id_field(a.items_xs, "cond")
    mj = dict(e1=id_field(a.items_e1, "mj_id"), mjen=id_field(a.items_mjen, "mj_id") if a.items_mjen else {})
    data = {t: load(a.eval_dir, t, xs_cond) for t in a.tags}
    missing = [t for t, v in data.items() if v is None]
    data = {t: v for t, v in data.items() if v is not None}
    res = dict(source="판정기 라벨(*_judge.jsonl: id·safety·refusal)·mmlu.jsonl만 읽음. 응답 원문 미열람", models={}, comparisons={}, new_unsafe={}, missing=missing)
    for t, v in data.items():
        res["models"][t] = summarize(v, xs_cond)
    pairs = [("base", "T100_ep3"), ("base", "C100_ep3"), ("C100_ep3", "T100_ep3")]
    for al in ALPHAS:
        pairs += [("T100_ep3", f"T100_ep3_resta{al}"), ("base", f"T100_ep3_resta{al}"), ("C100_ep3", f"T100_ep3_resta{al}")]
    for x, y in pairs:
        if x in data and y in data:
            res["comparisons"][f"{x} → {y}"] = dict(before=x, after=y, **compare(data[x], data[y], xs_cond))
    if "base" in data:
        for t, v in data.items():
            if t != "base":
                res["new_unsafe"][t] = {p: new_unsafe(data["base"], v, p, mj[p]) for p in ("e1", "mjen") if v[p] and data["base"][p]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)), exist_ok=True)
    json.dump(res, open(a.out_prefix + ".json", "w"), ensure_ascii=False, indent=1)
    open(a.out_prefix + ".md", "w", encoding="utf-8").write(md(res))
    print(json.dumps({t: dict(e1=s.get("e1", {}).get("refusal", {}).get("k"), mjen=s.get("mjen", {}).get("refusal", {}).get("k"),
                              mmlu_rep=s.get("mmlu_rep", {}).get("k")) for t, s in res["models"].items()}, ensure_ascii=False))
    print(json.dumps({t: {p: x["k"] for p, x in v.items()} for t, v in res["new_unsafe"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
