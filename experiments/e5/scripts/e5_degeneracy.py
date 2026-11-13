"""E5 응답 형태 통계(맥, 표준 라이브러리): gen jsonl의 응답 글을 숫자로만 요약한다. 응답 원문은 출력하지 않는다(E5 본실행 중 추가).
지표: 빈 응답 수, 128토큰 잘림 수, 평균 글자 수, 반복도(응답 안에서 가장 많이 나온 8글자 묶음이 차지하는 비율의 평균),
      같은 응답이 그대로 나온 최대 빈도(모델이 모든 질문에 같은 말을 하는지), 한글 글자 비율 평균.
사용: python3 -I -B e5_degeneracy.py <eval_dir> <out.json>"""
import collections, json, os, sys
ev, out = sys.argv[1], sys.argv[2]
TAGS = ["base", "T100_ep3", "C100_ep3"] + [f"T100_ep3_resta{a}" for a in ("0.25", "0.5", "0.75", "1.0")]
def rep(s, k=8):
    if len(s) < 2 * k: return 0.0
    c = collections.Counter(s[i:i + k] for i in range(len(s) - k + 1))
    return round(c.most_common(1)[0][1] * k / len(s), 3)
res = {}
for t in TAGS:
    res[t] = {}
    for part in ("e1", "mjen", "xs", "sel"):
        rows = [json.loads(l) for l in open(os.path.join(ev, t, f"{part}_gen.jsonl"), encoding="utf-8") if l.strip()]
        rs = [r.get("response") or "" for r in rows]
        n = len(rs); cnt = collections.Counter(x.strip() for x in rs)
        hang = [sum('가' <= ch <= '힣' for ch in x) / max(1, len(x)) for x in rs]
        res[t][part] = dict(n=n, empty=sum(not x.strip() for x in rs), truncated=sum(bool(r.get("truncated")) for r in rows),
                            mean_chars=round(sum(map(len, rs)) / n, 1), mean_repeat8=round(sum(map(rep, rs)) / n, 3),
                            n_distinct=len(cnt), top_identical=cnt.most_common(1)[0][1], mean_hangul_frac=round(sum(hang) / n, 3))
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
print("tag part n empty trunc mean_chars repeat8 distinct top_identical hangul")
for t, v in res.items():
    for p, d in v.items():
        print(t, p, d["n"], d["empty"], d["truncated"], d["mean_chars"], d["mean_repeat8"], d["n_distinct"], d["top_identical"], d["mean_hangul_frac"])
