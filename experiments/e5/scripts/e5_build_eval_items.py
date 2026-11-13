"""E5 새 평가 입력을 맥에서 고정한다(맥 CPU, 오프라인 HF 캐시, 표준 라이브러리 + pyarrow + curatorkit).

만드는 것(experiments/e5/data/eval_items/)
  mjen_items.jsonl : MultiJail(93d65c7) 같은 315문항의 영어 열. id = "mjen-<k>"(k = e1-<k>와 같은 순서·같은 mj_id). judge_query = 영어 원문
  sel_items.jsonl  : α 선택 전용 보류셋(주 보고 지표와 겹치지 않음)
                     - selh-<i> 100: AdvBench-Korean(leo-bjpark/AdvBench-Korean@f6a20e0, MIT, 520행) Random(20261113).sample 100
                     - selb-<i> 100: 무해 kullm(학습 후보 Random(20261111).sample(kullm,3000)·E4/E5 길이 맞춘 100 밖), Random(20261113).sample(300)에서
                       빈 응답·2차 위험 키워드·거절 규칙·맥락 의존(d1 pass_F와 같은 식)·프롬프트 400자 초과를 빼고 앞 100. 응답은 쓰지 않는다(프롬프트만)
  mmlu_items.jsonl : Global-MMLU ko test(CohereLabs/Global-MMLU@0e619db, 14,042문항) Random(20261113).sample 600
                     앞 400 = split "rep"(보고용: 과제 성능 유지), 뒤 200 = split "sel"(α 선택용). 겹치지 않음
겹침 검사: CuratorKIT MinHashDeduplicator(threshold 0.85·0.5, ngram 3)·ExactDeduplicator를 지시문만으로 돌려 MultiJail ko 315 뒤에 선택셋 유해 100을 넣고
  선택셋 행이 MultiJail과 중복으로 걸리는지 센다(D-4가 bt25866·bt9810을 찾은 것과 같은 목적).
출력 manifest: data/eval_items_manifest.json (파일 sha256, 표본 규칙, 겹침 검사). 글은 출력하지 않는다.
사용(PROJ에서): uv run python -I -B experiments/e5/scripts/e5_build_eval_items.py .
"""
import csv, json, os, random, re, sys, time

os.environ["HF_HUB_OFFLINE"] = "1"
PROJ = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(PROJ, "scripts"))
import e4_common as C  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

E5 = os.path.join(PROJ, "experiments/e5")
OUT = os.path.join(E5, "data/eval_items"); os.makedirs(OUT, exist_ok=True)
SEED5 = 20261113
REV = dict(multijail=C.REV["multijail"], advbench_ko="f6a20e02c7eceba1b2e684f556dddc1acdf5afc8", global_mmlu="0e619dbeb34206cd48705a1a0ea7fb21cae09993")
MJ = os.path.join(C.HUB, f"datasets--DAMO-NLP-SG--MultiJail/snapshots/{REV['multijail']}/MultiJail.csv")
ADV = os.path.join(C.HUB, f"datasets--leo-bjpark--AdvBench-Korean/snapshots/{REV['advbench_ko']}/data/train-00000-of-00001.parquet")
GM = os.path.join(C.HUB, f"datasets--CohereLabs--Global-MMLU/snapshots/{REV['global_mmlu']}/ko/test-00000-of-00001.parquet")
MMLU_PROMPT = "다음은 객관식 문제입니다. 정답의 알파벳(A, B, C, D) 하나만 답하세요.\n\n질문: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n정답:"
HANGUL = re.compile(r"[가-힣]")


def wj(name, rows):
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return dict(path=os.path.relpath(p, PROJ), rows=len(rows), sha256=C.sha256_file(p))


t0 = time.time()
# ---- MultiJail en: e1_items와 같은 순서·mj_id ----
e1 = C.read_jsonl(os.path.join(PROJ, "experiments/e1e3/e1_items.jsonl"))
with open(MJ, encoding="utf-8-sig") as f:
    mj = {int(r["id"]): r for r in csv.DictReader(f)}
mjen = []
for k, it in enumerate(e1):
    assert it["id"] == f"e1-{k}" and mj[it["mj_id"]]["ko"].strip() == it["prompt"]
    en = mj[it["mj_id"]]["en"].strip(); assert en
    mjen.append(dict(id=f"mjen-{k}", cond="mj_en", mj_id=it["mj_id"], prompt=en, judge_query=en, tags=it["tags"], source=it["source"]))
assert len(mjen) == 315

# ---- 선택셋: AdvBench-Korean 유해 100 ----
adv = pq.read_table(ADV).to_pylist()
hang = sum(1 for r in adv if len(HANGUL.findall(r["prompt"])) >= 5)
adv_idx = random.Random(SEED5).sample(range(len(adv)), 100)
selh = [dict(id=f"selh-{j}", cond="sel_harm", adv_idx=i, prompt=adv[i]["prompt"].strip(), judge_query=adv[i]["prompt"].strip()) for j, i in enumerate(adv_idx)]

# ---- 선택셋: 무해 kullm 100(프롬프트만) ----
DA = os.path.join(PROJ, "experiments/data_analysis")
A_AGG = json.load(open(os.path.join(DA, "A_kullm/agg.json")))
LEX = A_AGG["05_risk_keywords"]["lexicon"]; STRICT = A_AGG["05_risk_keywords"]["v2_strict"]["rules"]
PATS2 = []
for c, ks in LEX.items():
    for k in ks:
        p = re.compile(k[3:] if k.startswith("re:") else re.escape(k), re.I)
        if k in STRICT and STRICT[k]:
            p = re.compile(STRICT[k], re.I)
        PATS2.append(p)
risk2 = lambda t: any(p.search(t) for p in PATS2)
CONTEXT_RE = re.compile(r"^\s*(계속|더 |더$|다시|네[,.\s]|네$|좋아|감사|고마워|그럼|그렇다면|그리고|또|이제|위의|위 |앞의|앞서|이전|방금|해당|그것|그걸|이것|이걸|그건|이건|그 |이 |그런|이런|왜 그|어떻게 그|예를 들|더 자세히|확장|짧게|길게|수정|바꿔|continue)", re.I)
ku = []
for i, l in enumerate(open(C.KU_PATH, encoding="utf-8")):
    r = json.loads(l); ku.append(dict(rid=f"ku{i}", src=r["id"].split("_")[0], u=(r["instruction"] + "\n" + r.get("input", "")).strip(), a=r["output"]))
used = {r["rid"] for r in random.Random(C.SEED).sample(ku, 3000)}
e4man = json.load(open(os.path.join(PROJ, "experiments/e4/data/data_manifest.json")))
used |= set(e4man["arms"]["C200"]["slot_rids"])
e5man_p = os.path.join(E5, "data/data_manifest.json")
e5_slot = json.load(open(e5man_p))["arms"]["C100"]["slot_rids"] if os.path.exists(e5man_p) else []
used |= set(e5_slot)
cand = random.Random(SEED5).sample([r for r in ku if r["rid"] not in used], 300)
selb, why = [], {}
for r in cand:
    w = ("empty" if not r["a"].strip() or not r["u"] else "risk_keyword" if risk2(r["u"]) or risk2(r["a"]) else
         "context_dependent" if r["src"] == "vicuna" and CONTEXT_RE.search(r["u"]) and len(r["u"]) <= 40 else "prompt_gt_400" if len(r["u"]) > 400 else None)
    if w:
        why[w] = why.get(w, 0) + 1; continue
    if len(selb) < 100:
        selb.append(dict(id=f"selb-{len(selb)}", cond="sel_benign", rid=r["rid"], prompt=r["u"], judge_query=r["u"]))
assert len(selb) == 100
sel = selh + selb

# ---- 겹침 검사(CuratorKIT 중복 제거 부품을 지시문만으로) ----
from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator  # noqa: E402
from curatorkit.schema import DataSample  # noqa: E402
import logging; logging.disable(logging.WARNING)
overlap = {}
for name, ref in (("vs_multijail_ko", [it["prompt"] for it in e1]), ("vs_multijail_en", [it["prompt"] for it in mjen])):
    for thr in (0.85, 0.5):
        ss = [DataSample(id=f"ref{i}", source_uri="ref", task_type="prompt_only", instruction=t) for i, t in enumerate(ref)] + \
             [DataSample(id=it["id"], source_uri="sel", task_type="prompt_only", instruction=it["prompt"]) for it in sel]
        kept = ExactDeduplicator().run(ss)
        kept = MinHashDeduplicator(threshold=thr).run(kept)
        ks = {s.id for s in kept}
        alone = [DataSample(id=it["id"], source_uri="sel", task_type="prompt_only", instruction=it["prompt"]) for it in sel]
        ka = {s.id for s in MinHashDeduplicator(threshold=thr).run(ExactDeduplicator().run(alone))}
        within = sorted(it["id"] for it in sel if it["id"] not in ka)  # 선택셋 안에서 서로 중복(MultiJail과 무관)
        overlap[f"{name}_minhash{thr}"] = sorted(it["id"] for it in sel if it["id"] not in ks and it["id"] in ka)
        overlap[f"within_sel_minhash{thr}"] = within
# ---- Global-MMLU ko ----
gm = pq.read_table(GM).to_pylist()
gi = random.Random(SEED5).sample(range(len(gm)), 600)
mmlu = []
for j, i in enumerate(gi):
    r = gm[i]
    assert r["answer"] in "ABCD"
    mmlu.append(dict(id=f"mmlu-{j}", split="rep" if j < 400 else "sel", gm_idx=i, sample_id=r["sample_id"], subject=r["subject"],
                     subject_category=r["subject_category"], cultural_sensitivity_label=r["cultural_sensitivity_label"], answer=r["answer"],
                     prompt=MMLU_PROMPT.format(q=r["question"].strip(), a=str(r["option_a"]).strip(), b=str(r["option_b"]).strip(),
                                               c=str(r["option_c"]).strip(), d=str(r["option_d"]).strip())))
import collections  # noqa: E402
man = dict(
    _note="E5 새 평가 입력. 유해 프롬프트가 들어 있어 파일 자체는 공개하지 않는다(이 매니페스트에는 행 id와 sha256만 있다)",
    created_by="experiments/e5/scripts/e5_build_eval_items.py", created_kst=time.strftime("%Y-%m-%dT%H:%M:%S+09:00"), seed=SEED5, revisions=REV,
    files=dict(mjen=wj("mjen_items.jsonl", mjen), sel=wj("sel_items.jsonl", sel), mmlu=wj("mmlu_items.jsonl", mmlu)),
    existing=dict(e1_items=e4man["eval_inputs"]["e1_items"], xstest_items=e4man["eval_inputs"]["xstest_items"]),
    sel=dict(harm=dict(source="leo-bjpark/AdvBench-Korean", license="MIT(README)", rows_total=len(adv), rows_with_ge5_hangul=hang, picked=100, adv_idx=adv_idx),
             benign=dict(source="nlpai-lab/kullm-v2 (학습 후보 3,000·길이 맞춘 슬롯 밖)", candidates=300, excluded=why, picked=100, rids=[r["rid"] for r in selb],
                         e5_slot_excluded=bool(e5_slot)),
             overlap_check=dict(method="CuratorKIT ExactDeduplicator + MinHashDeduplicator(ngram 3, num_perm 128, seed 42), prompt_only, MultiJail 먼저",
                                flagged=overlap, flagged_counts={k: len(v) for k, v in overlap.items()})),
    mmlu=dict(source="CohereLabs/Global-MMLU ko test", rows_total=len(gm), picked=600, split_rule="표본 순서 앞 400 rep(보고), 뒤 200 sel(α 선택)",
              prompt_template=MMLU_PROMPT, scoring="기본 머리말 chat template(user=prompt, add_generation_prompt) 다음 첫 토큰 위치에서 'A','B','C','D' 토큰(id 47~50) 로짓 최댓값 = 답. 생성하지 않으므로 파싱 실패가 없다",
              rep_category=dict(collections.Counter(m["subject_category"] for m in mmlu[:400])), sel_category=dict(collections.Counter(m["subject_category"] for m in mmlu[400:])),
              rep_answer=dict(collections.Counter(m["answer"] for m in mmlu[:400]))),
    sec=round(time.time() - t0, 1))
json.dump(man, open(os.path.join(E5, "data/eval_items_manifest.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(dict(files={k: (v["rows"], v["sha256"][:16]) for k, v in man["files"].items()}, adv_hangul=f"{hang}/{len(adv)}",
                      benign_excluded=why, overlap=man["sel"]["overlap_check"]["flagged_counts"], e5_slot_excluded=bool(e5_slot), sec=man["sec"]), ensure_ascii=False))
