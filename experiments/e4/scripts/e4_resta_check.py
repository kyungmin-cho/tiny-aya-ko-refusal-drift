"""ReSta α0.25에서 거절이 새로 생긴 문항(복구 전 T*_ep3 거절 아님 → ReSta 거절)의 모양을 건수로만 센다(판독 아님, 응답을 읽지 않음).
세는 것: 거절 문구(E3 정규식) 걸린 수, 같은 단어 반복(고유 단어 비율 < 0.3, 20단어 이상) 수, 평균 글자 수, 판정기 safety 분포, 128토큰 잘림 수.
비교용으로 base에서 거절이던 문항(base 거절 True)의 같은 수치도 낸다.
사용: python3 -I scripts/e4_resta_check.py <e4_root>   출력: agg/e4_resta_check.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "scripts"))
import e4_common as C  # noqa: E402


def load(root, tag):
    d = os.path.join(root, "eval", tag)
    return {r["id"]: r for r in C.read_jsonl(os.path.join(d, "e1_gen.jsonl"))}, {r["id"]: r for r in C.read_jsonl(os.path.join(d, "e1_judge.jsonl"))}


def low_div(t):
    w = t.split(); return len(w) >= 20 and len(set(w)) / len(w) < 0.3


def stats(ids, g, j):
    kw = C.keyword_refusal(list(g.values())); sf = {}
    for i in ids:
        sf[j[i].get("safety")] = sf.get(j[i].get("safety"), 0) + 1
    rs = [g[i].get("response") or "" for i in ids]
    return dict(n=len(ids), refusal_keyword=sum(1 for i in ids if kw.get(i)), low_diversity=sum(map(low_div, rs)),
                mean_chars=round(sum(map(len, rs)) / len(rs), 1) if rs else None, truncated_128=sum(1 for i in ids if g[i].get("truncated")), safety=sf)


def main(root):
    out = {}
    bg, bj = load(root, "base")
    out["base_refusals"] = stats([i for i in bj if bj[i].get("refusal") is True], bg, bj)
    for arm in ("T100", "T200"):
        dg, dj = load(root, f"{arm}_ep3")
        for al in ("0.1", "0.25"):
            rg, rj = load(root, f"{arm}_ep3_resta{al}")
            new = [i for i in rj if rj[i].get("refusal") is True and dj[i].get("refusal") is not True]
            out[f"{arm}_resta{al}_new_refusals"] = stats(new, rg, rj)
    json.dump(out, open(os.path.join(root, "agg", "e4_resta_check.json"), "w"), ensure_ascii=False, indent=1)
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main(sys.argv[1])
