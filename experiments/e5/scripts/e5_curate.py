"""E5 1단계: CuratorKIT로 kullm 무해 학습 데이터(정제 kullm 2,000 + C100의 길이 맞춘 kullm 100)를 다시 정제하고
E5 학습 파일 T100·C100과 data_manifest.json을 만든다(맥 CPU, 오프라인 HF 캐시, 토크나이저만).

CuratorKIT 사용(curatorkit 0.1.24, 실제 파이프라인 실행)
  Curator(CuratorConfig(...)).run() 한 번에 JSONLReader → SchemaGate → ExactDeduplicator → MinHashDeduplicator → SecretsGate → MessagesExporter.
  - schema_gate=True, min_tokens=1, max_tokens=10**9: 빈 필드·인코딩(널 바이트)만 거른다. 길이는 SafeTune max_len 768 규칙과 길이 짝짓기가 맡는다
    (CuratorKIT 토큰 수는 공백 단어 + 한글 음절 1개=1토큰이라 Aya 토크나이저와 다르다). 기본값(10/2048)이면 몇 행이 빠지는지는 따로 센다.
  - dedup="minhash": ExactDeduplicator + MinHashDeduplicator(threshold 0.85, ngram 3, num_perm 128, seed 42, 모두 기본값). 먼저 나온 행을 남긴다.
  - secrets_gate=True: SecretsGate(detect-secrets 1.5.0, 기본 플러그인, hex 3.0·base64 4.5 기본 한도).
  - clean=False: TextCleaner는 글을 고친다(공백 접기는 코드 들여쓰기를 없앤다). 행 단위 통과/탈락만 CuratorKIT로 정하고 원문은 그대로 둔다.
    TextCleaner(기본 변환 전부)와 안전 부분(NFC·모지바케·제어문자)을 돌렸다면 몇 행이 바뀌었을지는 따로 센다.
  - pii_pseudonymize·toxicity_gate=False: 쓰지 못함(이유는 결과 json "not_used").
  - export_formats=["messages"]: 내보낸 sft_messages.jsonl의 글이 학습 파일 글과 같은지 확인한다.
최종 규칙(학습 입력): 같은 추출 순서 Random(20261111).sample(kullm, 3000)(앞 2,400 = E4와 같음)에서
  (기존 정제 규칙 통과) AND (CuratorKIT 통과)인 행을 앞에서부터 2,000개. 기존 규칙 = d1_controls.py 그대로
  (빈 응답, 보이지 않는 문자만, A 판독 위험 '애매', 기본 머리말 포함 프롬프트 ≥768토큰(학습 신호 0), 대체 행의 2차 위험 키워드).
  길이 맞춘 100: d1과 같은 짝짓기(풀·거리·동점 규칙 그대로)에서 CuratorKIT가 뺀 행을 풀에서 빼고 처음부터 다시 짝짓기, 바뀌지 않을 때까지 되풀이.
회귀 게이트: CuratorKIT 탈락을 빈 집합으로 두면 E4의 clean2000_rids와 C100 슬롯이 한 건도 다르지 않게 나와야 진행한다.
실행(PROJ에서, 임시 venv = uv.lock + detect-secrets):
  uv run python -I -B experiments/e5/scripts/e5_curate.py .
출력: experiments/e5/data/{T100,C100}.jsonl, data_manifest.json, curator_compare.json, curator_work/(CuratorKIT 입력·출력, 다시 만들 수 있음)
"""
import bisect, collections, copy, gzip, hashlib, json, logging, math, os, random, re, shutil, sys, time, warnings

os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
PROJ = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(PROJ, "scripts"))
import e4_common as C  # noqa: E402

DA = os.path.join(PROJ, "experiments/data_analysis")
E4 = os.path.join(PROJ, "experiments/e4")
E5 = os.path.join(PROJ, "experiments/e5")
OUT = os.path.join(E5, "data")
WORK = os.path.join(OUT, "curator_work")
os.makedirs(WORK, exist_ok=True)
t0 = time.time()
E5_TRAIN_SEED = 20261113  # E4(20261111)와 다른 새 시드. PLAN.md에 적음

from transformers import AutoTokenizer  # noqa: E402
from datasets import Dataset  # noqa: E402
from safetune.runner.utils.data_utils import tokenize_dataset  # noqa: E402
import curatorkit  # noqa: E402
from curatorkit import Curator, CuratorConfig  # noqa: E402
from curatorkit.gates.schema import SchemaGate  # noqa: E402
from curatorkit.normalizers.clean import TextCleaner  # noqa: E402
from curatorkit.schema import DataSample  # noqa: E402

AGG = json.load(open(os.path.join(DA, "agg_all.json")))
D1 = json.load(open(os.path.join(DA, "scripts_D/out/d1_controls.json")))
E4MAN = json.load(open(os.path.join(E4, "data/data_manifest.json")))
BSEL = json.load(open(os.path.join(DA, "B_beavertails/samples/selected_ids.json")))
H100_IDS = [x["id"] for x in BSEL["harmful100"]]
assert H100_IDS == E4MAN["arms"]["T100"]["slot_rids"]
OLD_CLEAN = E4MAN["arms"]["C100"]["clean_rids"]; OLD_SLOT = E4MAN["arms"]["C100"]["slot_rids"]
assert OLD_CLEAN == AGG["D_controls"]["clean"]["clean2000_rids"] and OLD_SLOT == AGG["D_controls"]["cands"]["dclean_control"]["slot"]["rids"]
OLD_REMOVED = AGG["D_controls"]["clean"]["removed_rids"]

# ---------------- 규칙(d1_controls.py·e4_build_data.py 그대로) ----------------
AI_ID = re.compile(r"(AI ?언어 ?모델|언어 ?모델(로서|이기 때문에|이므로|인 저는|으로서)|(AI|인공지능)(로서|이기 때문에|이므로|인 저는| 어시스턴트인 저는| 어시스턴트로서)|저는 (AI|인공지능)|OpenAI에서 (만든|개발한|훈련)|ChatGPT(입니다|로서))")
NO_PERSONAL = re.compile(r"개인적인 (의견|경험|감정|신념|선호|취향)(을|이|은)? ?(가지고 있지 않|없|갖고 있지 않)")
NO_ACCESS = re.compile(r"(실시간 (정보|데이터)|인터넷(에 접속|을 검색|을 탐색|에 액세스|에 접근)|브라우징|웹을 검색)[^.\n]{0,30}(없|않)|2021년 9월|지식 (차단|마감)")
REFUSAL = re.compile(r"(죄송하지만|죄송합니다만|유감스럽게도|안타깝게도)[^.\n]{0,80}(수 없|드릴 수 없|않습니다|불가능)|"
                     r"(도와드릴|제공할|제공해 드릴|생성할|작성할|응할|수행할|지원할|답변할|만들|도움을 드릴) 수 없습니다|"
                     r"(부적절|비윤리적|불법적|유해|공격적|차별적)[^.\n]{0,40}(수 없|않습니다|않겠습니다)|"
                     r"(정책|가이드라인|윤리 ?지침)(에 위배|을 위반|에 어긋|에 따라)")
CONTEXT_RE = re.compile(r"^\s*(계속|더 |더$|다시|네[,.\s]|네$|좋아|감사|고마워|그럼|그렇다면|그리고|또|이제|위의|위 |앞의|앞서|이전|방금|해당|그것|그걸|이것|이걸|그건|이건|그 |이 |그런|이런|왜 그|어떻게 그|예를 들|더 자세히|확장|짧게|길게|수정|바꿔|continue)", re.I)
A_AGG = json.load(open(os.path.join(DA, "A_kullm/agg.json")))
LEX = A_AGG["05_risk_keywords"]["lexicon"]; STRICT = A_AGG["05_risk_keywords"]["v2_strict"]["rules"]
PATS2 = []
for c, ks in LEX.items():
    for k in ks:
        p = re.compile(k[3:] if k.startswith("re:") else re.escape(k), re.I)
        if k in STRICT and STRICT[k]:
            p = re.compile(STRICT[k], re.I)
        PATS2.append((c, k, p))
risk2 = lambda t: sorted({k for c, k, p in PATS2 if p.search(t)})
a_refusal = lambda t: bool(AI_ID.search(t) or NO_PERSONAL.search(t) or NO_ACCESS.search(t) or REFUSAL.search(t))
INVIS = re.compile(r"[​‌‍⁠﻿\s]*")
invisible_only = lambda t: t.strip() != "" and INVIS.fullmatch(t) is not None
AMBIG = {x["rid"] for x in json.load(open(os.path.join(DA, "A_kullm/samples/risk_review.json")))["items"] if x["verdict"] == "애매"}

# ---------------- 원본 ----------------
ku = []
for i, l in enumerate(open(C.KU_PATH, encoding="utf-8")):
    r = json.loads(l); r["rid"] = f"ku{i}"; r["src"] = r["id"].split("_")[0]
    r["u"] = (r["instruction"] + "\n" + r.get("input", "")).strip(); r["a"] = r["output"]; ku.append(r)
by_ku = {r["rid"]: r for r in ku}
bt = [json.loads(l) for l in gzip.open(C.BT_PATH, "rt", encoding="utf-8")]
bt_t = {f"bt{i}": (r["prompt"], r["response"], "beavertails", r["is_safe"]) for i, r in enumerate(bt)}
ku_t = {r["rid"]: (r["u"], r["a"], r["src"]) for r in ku}
K2000 = random.Random(C.SEED).sample(ku, 2000)
K2400 = random.Random(C.SEED).sample(ku, 2400)
K3000 = random.Random(C.SEED).sample(ku, 3000)
gate = dict(k3000_prefix_is_k2400=[r["rid"] for r in K3000[:2400]] == [r["rid"] for r in K2400],
            k2400_prefix_is_k2000=[r["rid"] for r in K2400[:2000]] == [r["rid"] for r in K2000],
            ambig_count=len(AMBIG))
assert gate["k3000_prefix_is_k2400"] and gate["k2400_prefix_is_k2000"], gate
ORIG = {r["rid"] for r in K2000}

tok = AutoTokenizer.from_pretrained(C.AYA_SNAP)
tok.chat_template = tok.chat_template["default"] if isinstance(tok.chat_template, dict) else tok.chat_template
ntok = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
_pl = {}


def prompt_len(r):  # 기본 머리말 포함 프롬프트 토큰(d1 lens()[1]과 같은 계산)
    if r["rid"] not in _pl:
        _pl[r["rid"]] = ntok(tok.apply_chat_template([{"role": "user", "content": r["u"]}], tokenize=False, add_generation_prompt=True))
    return _pl[r["rid"]]


def old_rule(r):
    """기존 정제 규칙(d1_controls.py 정제안 루프)의 탈락 이유. 통과면 None."""
    if r["a"].strip() == "":
        return "empty"
    if invisible_only(r["a"]):
        return "invisible_only"
    if r["rid"] in AMBIG:
        return "risk_ambiguous_A_review"
    if prompt_len(r) >= 768:
        return "prompt_ge_768"
    if r["rid"] not in ORIG and (risk2(r["u"]) or risk2(r["a"])):
        return "replacement_risk_keyword"
    return None


# ---------------- CuratorKIT 실행 ----------------
CFG_KW = dict(schema_gate=True, min_tokens=1, max_tokens=10 ** 9, dedup="minhash", minhash_threshold=0.85, minhash_ngram=3,
              minhash_num_perm=128, minhash_seed=42, clean=False, secrets_gate=True, pii_pseudonymize=False, toxicity_gate=False,
              export_formats=["messages"], write_hf_readme=False, format="alpaca")


def cur_input_rows(rids):
    return [dict(id=rid, instruction=by_ku[rid]["u"], input="", output=by_ku[rid]["a"]) for rid in rids]


def run_curator(name, rids):
    """rids 순서대로 CuratorKIT 파이프라인을 돌린다. 반환: 통과 id 목록, 탈락 {id: (단계, 이유)}, 단계별 수, 내보낸 글 확인."""
    d = os.path.join(WORK, name); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    inp = os.path.join(d, "input.jsonl")
    with open(inp, "w", encoding="utf-8") as f:
        for row in cur_input_rows(rids):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    ts = time.time()
    res = Curator(CuratorConfig(dataset=inp, output_dir=os.path.join(d, "out"), name=f"lexsi_e5_{name}", **CFG_KW)).run()
    passed = [s.id for s in res.passed]
    rejected = {s.id: (s.rejecting_step, s.rejection_reason) for s in res.rejected}
    seen = set(passed) | set(rejected)
    for rid in rids:  # 중복 제거는 RejectedSample을 남기지 않으므로 사라진 행을 중복으로 적는다(아래에서 단계 확인)
        if rid not in seen:
            rejected[rid] = ("Deduplicator", "duplicate_removed")
    # 어느 중복 단계인지: 단계별 입력·출력 수로 확인하고, 행 단위로는 정확/근사 중복을 직접 다시 계산해 나눈다
    exact_keys = {}
    for rid in rids:
        k = hashlib.sha256((by_ku[rid]["u"] + by_ku[rid]["a"]).encode()).hexdigest()
        if rejected.get(rid, ("", ""))[0] == "Deduplicator":
            rejected[rid] = ("ExactDeduplicator" if k in exact_keys else "MinHashDeduplicator",
                             "exact_duplicate_of:" + exact_keys[k] if k in exact_keys else "near_duplicate(jaccard>=0.85)")
        exact_keys.setdefault(k, rid)
    exp = [json.loads(l) for l in open(os.path.join(d, "out", "sft_messages.jsonl"), encoding="utf-8")]
    exp_text = [(m["messages"][0]["content"], m["messages"][-1]["content"]) for m in exp]
    want = [(by_ku[r]["u"], by_ku[r]["a"]) for r in passed]
    return dict(passed=passed, rejected=rejected, stage_counts=res.stage_counts, sec=round(time.time() - ts, 1),
                export_rows=len(exp), export_text_equals_raw=exp_text == want,
                curator_manifest=os.path.relpath(os.path.join(d, "out", "manifest.json"), PROJ))


cand_rids = [r["rid"] for r in K3000]
CUR = run_curator("kullm_cand3000", cand_rids)
cur_rej = CUR["rejected"]
print("curator cand3000", len(CUR["passed"]), "passed,", len(cur_rej), "rejected", CUR["sec"], "s, export==raw", CUR["export_text_equals_raw"], flush=True)

# ---------------- 정제 2,000 고르기 ----------------


def pick_clean(cur_reject):
    clean, why_skip = [], {}
    for r in K3000:
        if len(clean) == 2000:
            break
        w = old_rule(r) or (("curatorkit:" + cur_reject[r["rid"]][0]) if r["rid"] in cur_reject else None)
        if w:
            why_skip[r["rid"]] = w; continue
        clean.append(r["rid"])
    assert len(clean) == 2000
    return clean, why_skip


reg_clean, _ = pick_clean({})
gate["regression_clean2000_equals_e4"] = reg_clean == OLD_CLEAN
assert gate["regression_clean2000_equals_e4"], "기존 규칙 재현 실패"
NEW_CLEAN, NEW_SKIP = pick_clean(cur_rej)
pos = {r["rid"]: i for i, r in enumerate(K3000)}
gate["new_clean_within_k2400"] = max(pos[r] for r in NEW_CLEAN) < 2400

# ---------------- 비교(원래 2,000행 기준 세 갈래) ----------------
old_rm = set(OLD_REMOVED); cur_rm_orig = {r for r in ORIG if r in cur_rej}
old_reason = {r["rid"]: old_rule(r) for r in K2000 if r["rid"] in old_rm}
cmp3 = dict(
    base="원래 추출 2,000행(Random(20261111).sample(kullm,2000))에 각 규칙만 따로 적용",
    old_removed=len(old_rm), curatorkit_removed=len(cur_rm_orig),
    both=sorted(old_rm & cur_rm_orig, key=lambda x: int(x[2:])),
    curatorkit_only=sorted(cur_rm_orig - old_rm, key=lambda x: int(x[2:])),
    old_only=sorted(old_rm - cur_rm_orig, key=lambda x: int(x[2:])),
    kept_by_both=len(ORIG - old_rm - cur_rm_orig),
)
cmp3["n_both"], cmp3["n_curatorkit_only"], cmp3["n_old_only"] = len(cmp3["both"]), len(cmp3["curatorkit_only"]), len(cmp3["old_only"])
cmp3["curatorkit_reason"] = {r: list(cur_rej[r]) for r in sorted(cur_rm_orig, key=lambda x: int(x[2:]))}
cmp3["old_reason_counts"] = dict(collections.Counter(old_reason.values()))
cmp3["old_only_reason_counts"] = dict(collections.Counter(old_reason[r] for r in cmp3["old_only"]))
cmp3["curatorkit_only_reason_counts"] = dict(collections.Counter(cur_rej[r][0] for r in cmp3["curatorkit_only"]))
cmp3["old_only_by_reason"] = {k: [r for r in cmp3["old_only"] if old_reason[r] == k] for k in cmp3["old_only_reason_counts"]}
# 최종 2,000 대 E4 2,000
fin = dict(same_rows=len(set(NEW_CLEAN) & set(OLD_CLEAN)), only_new=sorted(set(NEW_CLEAN) - set(OLD_CLEAN), key=lambda x: int(x[2:])),
           only_e4=sorted(set(OLD_CLEAN) - set(NEW_CLEAN), key=lambda x: int(x[2:])),
           order_of_common_rows_same=[r for r in NEW_CLEAN if r in set(OLD_CLEAN)] == [r for r in OLD_CLEAN if r in set(NEW_CLEAN)],
           last_position_used=max(pos[r] for r in NEW_CLEAN), e4_last_position_used=max(pos[r] for r in OLD_CLEAN))
fin["only_e4_reason"] = {r: NEW_SKIP.get(r) for r in fin["only_e4"]}
fin["n_only_new"], fin["n_only_e4"] = len(fin["only_new"]), len(fin["only_e4"])

# 진단: CuratorKIT 기본 길이 한도(10/2048)와 TextCleaner가 했을 일(적용하지 않음)
sg = SchemaGate(min_tokens=10, max_tokens=2048)
samples = [DataSample(id=rid, source_uri="diag", instruction=by_ku[rid]["u"], output=by_ku[rid]["a"]) for rid in NEW_CLEAN]
sp, srj = sg.run(copy.deepcopy(samples))
diag = dict(schema_default_10_2048_on_final2000=dict(would_reject=len(srj), reasons=dict(collections.Counter(s.rejection_reason.split(":")[0] for s in srj))))
for nm, tf in (("all_default", None), ("safe_subset_nfc_mojibake_control", dict(strip_html=False, collapse_whitespace=False))):
    cl = TextCleaner(transforms=tf).run(copy.deepcopy(samples))
    ch = [s.id for s, o in zip(cl, samples) if (s.instruction, s.output) != (o.instruction.strip(), o.output.strip()) or s.instruction != o.instruction or s.output != o.output]
    diag[f"textcleaner_{nm}_rows_changed_on_final2000"] = len(ch)
print("diag", diag, flush=True)

# ---------------- 길이 맞춘 100(d1 짝짓기 + CuratorKIT 반복) ----------------
excl = {r["rid"] for r in K2400}


def pass_F(r):
    if r["rid"] in excl or r["a"].strip() == "" or invisible_only(r["a"]) or r["u"] == "":
        return False
    if a_refusal(r["a"]) or risk2(r["u"]) or risk2(r["a"]):
        return False
    if r["src"] == "vicuna" and CONTEXT_RE.search(r["u"]) and len(r["u"]) <= 40:
        return False
    return True


pool = [r for r in ku if pass_F(r)]
gate["pool_after_F"] = len(pool); gate["pool_equals_d1"] = len(pool) == D1["pool"]["pool_after_F"]
assert gate["pool_equals_d1"]


def batch_len(texts, bs=2048):
    out = []
    for i in range(0, len(texts), bs):
        out += [len(x) for x in tok(texts[i:i + bs], add_special_tokens=False)["input_ids"]]
    return out


pu = batch_len([r["u"] for r in pool]); pa = batch_len([r["a"] for r in pool])
print("pool tokenized", len(pool), round(time.time() - t0), "s", flush=True)
rank = list(range(len(pool))); random.Random(C.MATCH_TIE_SEED).shuffle(rank)
order = sorted(range(len(pool)), key=lambda i: (pa[i], rank[i]))
keys = [pa[i] for i in order]
H = [dict(rid=i, u=bt_t[i][0], a=bt_t[i][1]) for i in H100_IDS]
hu = batch_len([h["u"] for h in H]); ha = batch_len([h["a"] for h in H])


def match(banned):
    used = set(); picked, log = [], []
    for j, h in enumerate(H):
        best = None
        c = bisect.bisect_left(keys, ha[j])
        for direction in (1, -1):
            k = c if direction == 1 else c - 1
            while 0 <= k < len(order):
                i = order[k]
                da = abs(pa[i] - ha[j])
                if best is not None and da > best[0]:
                    break
                if i not in used and pool[i]["rid"] not in banned:
                    cand = (da + abs(pu[i] - hu[j]), rank[i], i)
                    if best is None or cand[:2] < best[:2]:
                        best = cand
                k += direction
        i = best[2]; used.add(i); picked.append(pool[i]["rid"])
        log.append(dict(harm=h["rid"], harm_u=hu[j], harm_a=ha[j], ku=pool[i]["rid"], ku_src=pool[i]["src"], ku_u=pu[i], ku_a=pa[i], dist=best[0]))
    return picked, log


reg_slot, reg_log = match(set())
gate["regression_c100_slot_equals_e4"] = reg_slot == OLD_SLOT
gate["regression_match_log_equals_d1"] = reg_log == D1["match"]["log"]
assert gate["regression_c100_slot_equals_e4"] and gate["regression_match_log_equals_d1"], gate
banned, rounds = set(), []
while True:
    slot, slog = match(banned)
    R = run_curator(f"clean2000_plus_slot_r{len(rounds)}", NEW_CLEAN + slot)
    rej_clean = [r for r in NEW_CLEAN if r in R["rejected"]]
    rej_slot = {r: list(R["rejected"][r]) for r in slot if r in R["rejected"]}
    rounds.append(dict(round=len(rounds), slot_rejected=rej_slot, clean_rejected_in_joint_run=rej_clean, stage_counts=R["stage_counts"],
                       export_text_equals_raw=R["export_text_equals_raw"], sec=R["sec"]))
    print("slot round", len(rounds) - 1, "slot rejected", len(rej_slot), "clean rejected", len(rej_clean), flush=True)
    assert not rej_clean, f"정제 2,000 안에서 함께 돌린 CuratorKIT가 뺀 행: {rej_clean}"
    if not rej_slot:
        break
    banned |= set(rej_slot)
    assert len(rounds) < 20
NEW_SLOT, NEW_LOG = slot, slog
slot_cmp = dict(same=len(set(NEW_SLOT) & set(OLD_SLOT)), only_new=[r for r in NEW_SLOT if r not in set(OLD_SLOT)],
                only_e4=[r for r in OLD_SLOT if r not in set(NEW_SLOT)], banned_by_curatorkit=sorted(banned),
                same_order_where_same=[r for r in NEW_SLOT if r in set(OLD_SLOT)] == [r for r in OLD_SLOT if r in set(NEW_SLOT)])

# ---------------- 파일 쓰기(E4 manifest 형식 그대로) ----------------
SPEC = {
    "T100": dict(desc="E5 처리군: CuratorKIT 재정제 kullm 2,000 + BeaverTails 유해 100(E4와 같은 100)", slot_part="harm", slot_rids=H100_IDS),
    "C100": dict(desc="E5 대조군: CuratorKIT 재정제 kullm 2,000 + 유해 100과 길이 맞춘 kullm 100(d1 짝짓기 + CuratorKIT 탈락 제외)",
                 slot_part="kullm_lenmatched", slot_rids=NEW_SLOT),
}
arms = {}
for name, s in SPEC.items():
    spec = dict(file=f"{name}.jsonl", desc=s["desc"], clean_rids=NEW_CLEAN, slot_part=s["slot_part"], slot_rids=s["slot_rids"])
    rows = C.arm_rows(spec, ku_t, bt_t)
    b = C.jsonl_bytes(rows)
    open(os.path.join(OUT, spec["file"]), "wb").write(b)
    spec["sha256"] = C.sha256_bytes(b); spec["bytes"] = len(b)
    ds = tokenize_dataset(Dataset.from_list([{"messages": r["messages"]} for r in rows]), tok, max_len=C.MAX_LEN)
    assert len(ds) == len(rows)
    lab = [sum(1 for t in row if t != -100) for row in ds["labels"]]
    chk = C.check_arm_rows(rows, 2100, 100 if name == "T100" else 0)
    chk.pop("clean2000")
    chk.update(safetune_rows=len(ds), zero_label_rows=sum(v == 0 for v in lab), label_tokens_total=sum(lab),
               label_tokens_mean=round(sum(lab) / len(rows), 1), slot_label_tokens_total=sum(lab[2000:]),
               steps_per_epoch=math.ceil(len(ds) / C.BATCH), steps_total_3ep=3 * math.ceil(len(ds) / C.BATCH),
               truncated_rows_768=sum(1 for row in ds["attention_mask"] if sum(row) >= C.MAX_LEN),
               slot_src=dict(collections.Counter(r["src"] for r in rows[2000:])))
    spec["checks"] = chk
    arms[name] = spec
    print(name, {k: v for k, v in chk.items() if k != "slot_src"}, flush=True)
    # 같은 rid의 materialize가 같은 바이트를 내는지(포드 재생성 경로)
mat_dir = os.path.join(WORK, "materialize_check"); shutil.rmtree(mat_dir, ignore_errors=True)

pairs = dict(T100_vs_C100=dict(
    clean2000_identical_in_order=arms["T100"]["clean_rids"] == arms["C100"]["clean_rids"],
    rows_equal=arms["T100"]["checks"]["rows"] == arms["C100"]["checks"]["rows"],
    steps_equal=arms["T100"]["checks"]["steps_per_epoch"] == arms["C100"]["checks"]["steps_per_epoch"],
    label_tokens_treat=arms["T100"]["checks"]["label_tokens_total"], label_tokens_ctrl=arms["C100"]["checks"]["label_tokens_total"],
    label_tokens_diff_ctrl_minus_treat=arms["C100"]["checks"]["label_tokens_total"] - arms["T100"]["checks"]["label_tokens_total"],
    label_tokens_diff_pct=round(100 * (arms["C100"]["checks"]["label_tokens_total"] - arms["T100"]["checks"]["label_tokens_total"]) / arms["T100"]["checks"]["label_tokens_total"], 4)))


def dist_q(xs):
    xs = sorted(xs); q = lambda p: xs[min(len(xs) - 1, int(len(xs) * p))]
    return dict(n=len(xs), p50=q(.5), p90=q(.9), max=max(xs), mean=round(sum(xs) / len(xs), 2))


import detect_secrets  # noqa: E402
NOT_USED = dict(
    PIIPseudonymizer="쓰지 않음. Presidio NER 모델(spaCy en_core_web_lg 또는 stanza·transformers 모델 가중치)이 필요하다. spaCy 모델은 PyPI·HF 리비전 고정 경로가 아니고 "
                     "모델 가중치를 맥에 받지 않는다는 제약에 걸린다. 한국어 spaCy 기본 NER 모델 설정도 없다. 또 이 단계는 PII를 가짜 값으로 바꿔 글을 고치는 정규화기라 행 통과/탈락 규칙과 맞지 않는다",
    ToxicityGate="쓰지 않음. Detoxify 분류기 가중치가 필요하다(curatorkit[hygiene] 미설치, 모델 가중치를 맥에 받지 않는다는 제약). "
                 "Detoxify 'unbiased'·'original'은 영어 모델이다(curatorkit hygiene/toxicity.py 문서). 한국어 유해성은 기존 규칙(A 판독 '애매', 2차 위험 키워드)으로 대신한다",
    TextCleaner="통과/탈락에는 쓰지 않음(clean=False). 글을 고치는 단계라 원문 유지·E4 비교·포드 재생성(sha256)을 깨기 때문. 고쳤다면 몇 행이 바뀌었을지는 diagnostics에 있다",
    EmbeddingDeduplicator_DiversityGate="쓰지 않음. sentence-transformers 임베딩 모델 가중치가 필요하다",
    LLM_gates="쓰지 않음(HallucinationGate·RewardGate·생성). 외부 LLM API가 필요하다",
)
compare = dict(
    _note="CuratorKIT 재정제 결과와 기존 정제(D_controls.clean, 107행 교체) 비교. id만 담는다(글 없음)",
    created_by="experiments/e5/scripts/e5_curate.py", created_kst=time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
    versions=dict(curatorkit=curatorkit.__version__, detect_secrets=__import__("importlib.metadata").metadata.version("detect-secrets")),
    curatorkit_config=CFG_KW, curatorkit_pipeline="Curator(CuratorConfig(...)).run(): JSONLReader(format=alpaca) → SchemaGate → ExactDeduplicator → MinHashDeduplicator → SecretsGate → MessagesExporter",
    not_used=NOT_USED, gates=gate,
    cand3000=dict(rows=len(cand_rids), passed=len(CUR["passed"]), rejected=len(cur_rej), stage_counts=CUR["stage_counts"],
                  rejected_by_step=dict(collections.Counter(v[0] for v in cur_rej.values())),
                  rejected_by_reason=dict(collections.Counter(v[1].split(":")[0] for v in cur_rej.values())),
                  rejected_within_first2400=sum(1 for r in cur_rej if pos[r] < 2400),
                  export_text_equals_raw=CUR["export_text_equals_raw"], sec=CUR["sec"], curator_manifest=CUR["curator_manifest"]),
    original2000_three_way=cmp3, final2000_vs_e4=fin, diagnostics=diag,
    slot100=dict(rounds=rounds, compare_vs_e4=slot_cmp, match_dist=dist_q([m["dist"] for m in NEW_LOG]),
                 e4_match_dist=dist_q([m["dist"] for m in D1["match"]["log"]])),
)
json.dump(compare, open(os.path.join(OUT, "curator_compare.json"), "w"), ensure_ascii=False, indent=1)

E1E3 = os.path.join(PROJ, "experiments/e1e3")
manifest = dict(
    _note="E5 학습 파일 2개(T100·C100). 유해 문장이 들어 있어 파일 자체는 공개하지 않는다(이 매니페스트에는 행 id와 sha256만 있다). 행 id 규칙은 E4와 같다: ku<줄번호>=nlpai-lab/kullm-v2@cddcb73 kullm-v2.jsonl, "
          "bt<줄번호>=PKU-Alignment/BeaverTails@8401fe6 round0/30k/train.jsonl.gz (0-기반)",
    created_by="experiments/e5/scripts/e5_curate.py", created_kst=time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
    revisions=C.REV, tokenizer=E4MAN["tokenizer"],
    train_cfg=dict(max_len=C.MAX_LEN, batch=C.BATCH, lr=C.LR, epochs=C.EPOCHS, lora=C.LORA, train_seed=E5_TRAIN_SEED, e4_train_seed=C.TRAIN_SEED,
                   shuffle=f"safetune.configure(seed={E5_TRAIN_SEED}) → Trainer 기본 RandomSampler. 두 팔 같은 규칙", primary_epoch=3, checkpoint_epochs=[1, 3]),
    row_format=E4MAN["row_format"],
    sources=dict(clean2000="CuratorKIT 재정제(curator_compare.json). 기존 규칙 AND CuratorKIT 통과, Random(20261111).sample(kullm,3000) 순서",
                 harm100="B_beavertails/samples/selected_ids.json harmful100(E4와 같음)",
                 c100_slot="d1 짝짓기 규칙 + CuratorKIT 탈락 행 제외(curator_compare.json slot100)"),
    curatorkit=dict(version=curatorkit.__version__, config=CFG_KW, compare="data/curator_compare.json"),
    pairs=pairs, arms=arms,
    eval_inputs=dict(e1_items=E4MAN["eval_inputs"]["e1_items"], xstest_items=E4MAN["eval_inputs"]["xstest_items"]),
    runtime_sec=round(time.time() - t0, 1),
)
for k in ("e1_items", "xstest_items"):
    assert C.sha256_file(os.path.join(PROJ, manifest["eval_inputs"][k]["path"])) == manifest["eval_inputs"][k]["sha256"]
json.dump(manifest, open(os.path.join(OUT, "data_manifest.json"), "w"), ensure_ascii=False, indent=1)
# 포드 경로(materialize)가 같은 바이트를 내는지
os.makedirs(mat_dir)
got = C.materialize(manifest, mat_dir)
shutil.rmtree(mat_dir)
summary = dict(
    cand3000=compare["cand3000"]["rejected_by_step"], three_way={k: cmp3[k] for k in ("old_removed", "curatorkit_removed", "n_both", "n_curatorkit_only", "n_old_only")},
    final_vs_e4=dict(same=fin["same_rows"], only_new=fin["n_only_new"], only_e4=fin["n_only_e4"]),
    slot=dict(same=slot_cmp["same"], changed=len(slot_cmp["only_new"]), rounds=len(rounds)),
    arms={a: dict(rows=s["checks"]["rows"], zero=s["checks"]["zero_label_rows"], label_tokens=s["checks"]["label_tokens_total"],
                  leak=s["checks"]["leak_rows_present"], sha=s["sha256"][:16]) for a, s in arms.items()},
    diff=pairs["T100_vs_C100"]["label_tokens_diff_ctrl_minus_treat"], materialize=got, runtime_sec=manifest["runtime_sec"])
print(json.dumps(summary, ensure_ascii=False, indent=1))
