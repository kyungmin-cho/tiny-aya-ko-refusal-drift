"""읽지 않고 낼 수 있는 보조 수치(맥, 표준 라이브러리만). 판독이 아니다: 응답을 사람이나 에이전트가 읽지 않고 숫자만 센다.
- 새 Unsafe(base Safe → Unsafe) 중 판정기 거절 Yes가 함께 붙은 건수, E3 거절 문구가 걸린 건수, 판정기 범주 분포
- 응답 모양(MultiJail 315): 평균 글자 수, 128토큰 잘림 수, 같은 단어 반복 비율(응답 단어 중 고유 단어 비율이 0.3 미만인 응답 수), 빈 응답 수
11-12 처리군 새 Unsafe 판독이 안전장치에 막혀(RESULT.md 막힌 것) 이 수치로 보조한다.
사용: python3 -I scripts/e4_proxy_metrics.py <e4_root>   출력: agg/e4_proxy.json
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "scripts"))
import e4_common as C  # noqa: E402


def shape(gen):
    rs = [r.get("response") or "" for r in gen]
    def low_div(t):
        w = t.split()
        return len(w) >= 20 and len(set(w)) / len(w) < 0.3
    return dict(n=len(rs), mean_chars=round(sum(map(len, rs)) / len(rs), 1), truncated_128=sum(1 for r in gen if r.get("truncated")),
                empty=sum(1 for t in rs if not t.strip()), low_diversity_lt0_3=sum(map(low_div, rs)))


def main(root):
    agg = json.load(open(os.path.join(root, "agg", "e4_agg.json")))
    out = dict(note="판독 아님(응답을 읽지 않고 센 숫자). 판정기 과대 측정 근거: E1 평문 Unsafe 중 실질 유해 4/9, E3 변형 Unsafe 60건 중 57건이 실질 유해 아님(DATA_ANALYSIS.md 5절 11번)",
               models={})
    for tag in agg["models"]:
        d = os.path.join(root, "eval", tag)
        gen = C.read_jsonl(os.path.join(d, "e1_gen.jsonl")); jd = {r["id"]: r for r in C.read_jsonl(os.path.join(d, "e1_judge.jsonl"))}
        kw = C.keyword_refusal(gen)
        row = dict(shape=shape(gen))
        if tag in agg["new_unsafe"]:
            ids = agg["new_unsafe"][tag]["ids"]; cats = {}
            for i in ids:
                for c in (jd[i].get("categories") or ["(없음)"]) if isinstance(jd[i].get("categories"), list) else [jd[i].get("categories")]:
                    cats[str(c)] = cats.get(str(c), 0) + 1
            row["new_unsafe"] = dict(n=len(ids), judge_refusal_yes=sum(1 for i in ids if jd[i].get("refusal") is True),
                                     refusal_keyword=sum(1 for i in ids if kw.get(i)), categories=dict(sorted(cats.items(), key=lambda x: -x[1])))
        out["models"][tag] = row
    json.dump(out, open(os.path.join(root, "agg", "e4_proxy.json"), "w"), ensure_ascii=False, indent=1)
    for t, r in out["models"].items():
        print(t, r["shape"], r.get("new_unsafe", {}).get("n"), r.get("new_unsafe", {}).get("judge_refusal_yes"), r.get("new_unsafe", {}).get("refusal_keyword"))


if __name__ == "__main__":
    main(sys.argv[1])
