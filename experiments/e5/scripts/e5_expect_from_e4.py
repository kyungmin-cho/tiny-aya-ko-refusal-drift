"""E4 집계(agg/e4_agg.json)에서 AuditKIT 예행의 기대 건수 파일을 만든다(맥, 표준 라이브러리).
사용: python -I -B e5_expect_from_e4.py <proj> <out.json> base T100_ep3 C100_ep3 T100_ep3_resta0.25"""
import json, os, sys
proj, out, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
a = json.load(open(os.path.join(proj, "experiments/e4/agg/e4_agg.json")))["models"]
exp = {t: dict(e1=dict(refusal=a[t]["mj_judge_refusal"]["k"], unsafe=a[t]["mj_unsafe"]["k"]),
               xs_safe=dict(refusal=a[t]["xstest"]["xstest_safe"]["k"]), xs_unsafe=dict(refusal=a[t]["xstest"]["xstest_unsafe"]["k"])) for t in tags}
json.dump(dict(exp, _source="experiments/e4/agg/e4_agg.json models.<tag>.mj_judge_refusal.k·mj_unsafe.k·xstest.*.k"), open(out, "w"), ensure_ascii=False, indent=1)
print(json.dumps(exp))
