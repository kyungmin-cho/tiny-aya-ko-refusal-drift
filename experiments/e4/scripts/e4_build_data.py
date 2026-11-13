"""E4 S1: 학습 파일 4개(T100·C100·T200·C200)와 data_manifest.json을 만들고 숫자로 검증한다(맥 CPU, 오프라인 HF 캐시, 토크나이저만).

행 id 출처(모두 그대로 씀, 새로 뽑지 않음)
  정제 kullm 2,000 : agg_all.json D_controls.clean.clean2000_rids (= scripts_D/out/d1_controls.json clean.clean2000_rids)
  유해 100         : B_beavertails/samples/selected_ids.json harmful100 (id·sha1 10자리 확인)
  길이 맞춘 kullm 100: agg_all.json D_controls.cands.dclean_control.slot.rids
  유해 새 100      : agg_all.json D_harm_refill.refill200.new_ids (bt25866·bt9810 제외 확인)
C200의 새 kullm 100(이번에 새로 고름): D-1과 같은 짝짓기 규칙
  풀 = kullm 전체 − Random(20261111).sample(kullm,2400) 전부 − 필터 F 미통과(d1_controls.py pass_F 그대로)
  유해 새 100건을 new_ids 순서대로 하나씩, |Δ응답 원문 토큰|+|Δ지시 원문 토큰| 최소, 동점은 Random(20261115)로 섞은 풀 순서,
  복원 없음, 이미 쓴 행(C100의 100) 제외. 먼저 같은 코드로 C100의 100건과 match.log를 한 건도 다르지 않게 재현해야 진행한다(게이트).
검증(SafeTune tokenize_dataset 실제 labels, max_len 768, 기본 머리말 템플릿): 행 수, 학습 신호 0 행, 학습 토큰 합·처리군 대비 차,
  스텝/에포크(배치 4), 유해 비중, 누출 행 없음, 처리·대조 짝의 kullm 2,000 같음(순서까지), 파일 sha256.
실행(PROJ에서):
  uv run --offline --no-project --python 3.12 --with-requirements experiments/data_analysis/A_kullm/requirements-A_kullm.txt \
    python -I -B experiments/e4/scripts/e4_build_data.py .
출력: experiments/e4/data/{T100,C100,T200,C200}.jsonl, data_manifest.json, build_log.json
"""
import bisect, collections, gzip, hashlib, json, math, os, random, re, sys, time, warnings

os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore")
PROJ = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(PROJ, "scripts"))
import e4_common as C  # noqa: E402

DA = os.path.join(PROJ, "experiments/data_analysis")
OUT = os.path.join(PROJ, "experiments/e4/data")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

from transformers import AutoTokenizer  # noqa: E402
from datasets import Dataset  # noqa: E402
from safetune.runner.utils.data_utils import tokenize_dataset  # noqa: E402

SEED = C.SEED
AGG = json.load(open(os.path.join(DA, "agg_all.json")))
D1 = json.load(open(os.path.join(DA, "scripts_D/out/d1_controls.json")))
D4 = json.load(open(os.path.join(DA, "scripts_D/out/d4_harm_refill.json")))
CLEAN = AGG["D_controls"]["clean"]["clean2000_rids"]
D100_REF = AGG["D_controls"]["cands"]["dclean_control"]["slot"]["rids"]
NEW100 = AGG["D_harm_refill"]["refill200"]["new_ids"]
BSEL = json.load(open(os.path.join(DA, "B_beavertails/samples/selected_ids.json")))
H100_IDS = [x["id"] for x in BSEL["harmful100"]]
gate = dict(agg_equals_d1_clean=CLEAN == D1["clean"]["clean2000_rids"],
            agg_equals_d1_dslot=D100_REF == D1["cands"]["dclean_control"]["slot"]["rids"],
            agg_equals_d4_new=NEW100 == D4["refill200"]["new_ids"],
            new100_excludes_leak=not (set(NEW100) & set(C.LEAK_IDS)),
            h100_excludes_leak=not (set(H100_IDS) & set(C.LEAK_IDS)),
            d4_skipped_are_leak=sorted(x["id"] for x in D4["refill200"]["skipped_leak_rows"]) == sorted(C.LEAK_IDS))

# ---------------- 규칙(d1_controls.py 그대로) ----------------
AI_ID = re.compile(r"(AI ?언어 ?모델|언어 ?모델(로서|이기 때문에|이므로|인 저는|으로서)|(AI|인공지능)(로서|이기 때문에|이므로|인 저는| 어시스턴트인 저는| 어시스턴트로서)|저는 (AI|인공지능)|OpenAI에서 (만든|개발한|훈련)|ChatGPT(입니다|로서))")
NO_PERSONAL = re.compile(r"개인적인 (의견|경험|감정|신념|선호|취향)(을|이|은)? ?(가지고 있지 않|없|갖고 있지 않)")
NO_ACCESS = re.compile(r"(실시간 (정보|데이터)|인터넷(에 접속|을 검색|을 탐색|에 액세스|에 접근)|브라우징|웹을 검색)[^.\n]{0,30}(없|않)|2021년 9월|지식 (차단|마감)")
REFUSAL = re.compile(r"(죄송하지만|죄송합니다만|유감스럽게도|안타깝게도)[^.\n]{0,80}(수 없|드릴 수 없|않습니다|불가능)|"
                     r"(도와드릴|제공할|제공해 드릴|생성할|작성할|응할|수행할|지원할|답변할|만들|도움을 드릴) 수 없습니다|"
                     r"(부적절|비윤리적|불법적|유해|공격적|차별적)[^.\n]{0,40}(수 없|않습니다|않겠습니다)|"
                     r"(정책|가이드라인|윤리 ?지침)(에 위배|을 위반|에 어긋|에 따라)")
CONTEXT_RE = re.compile(r"^\s*(계속|더 |더$|다시|네[,.\s]|네$|좋아|감사|고마워|그럼|그렇다면|그리고|또|이제|위의|위 |앞의|앞서|이전|방금|해당|그것|그걸|이것|이걸|그건|이건|그 |이 |그런|이런|왜 그|어떻게 그|예를 들|더 자세히|확장|짧게|길게|수정|바꿔|continue)", re.I)
A_AGG = json.load(open(os.path.join(DA, "A_kullm/agg.json")))
LEX = A_AGG["05_risk_keywords"]["lexicon"]
STRICT = A_AGG["05_risk_keywords"]["v2_strict"]["rules"]
PATS2 = []
for c, ks in LEX.items():
    for k in ks:
        p = re.compile(k[3:] if k.startswith("re:") else re.escape(k), re.I)
        if k in STRICT and STRICT[k]:
            p = re.compile(STRICT[k], re.I)
        PATS2.append((c, k, p))


def risk2(t):
    return sorted({k for c, k, p in PATS2 if p.search(t)})


def a_refusal(t):
    return bool(AI_ID.search(t) or NO_PERSONAL.search(t) or NO_ACCESS.search(t) or REFUSAL.search(t))


INVIS = re.compile(r"[​‌‍⁠﻿\s]*")


def invisible_only(t):
    return t.strip() != "" and INVIS.fullmatch(t) is not None


# ---------------- 원본 ----------------
ku = []
for i, l in enumerate(open(C.KU_PATH, encoding="utf-8")):
    r = json.loads(l); r["rid"] = f"ku{i}"; r["src"] = r["id"].split("_")[0]
    r["u"] = (r["instruction"] + "\n" + r.get("input", "")).strip(); r["a"] = r["output"]; ku.append(r)
bt = [json.loads(l) for l in gzip.open(C.BT_PATH, "rt", encoding="utf-8")]
for i, r in enumerate(bt):
    r["rid"] = f"bt{i}"; r["src"] = "beavertails"; r["u"] = r["prompt"]; r["a"] = r["response"]
by_ku = {r["rid"]: r for r in ku}; by_bt = {r["rid"]: r for r in bt}
sha1 = lambda r: hashlib.sha1((r["prompt"] + "\x00" + r["response"]).encode("utf-8")).hexdigest()[:10]
gate["h100_ids_sha1_equal_B"] = [dict(id=i, sha1_10=sha1(by_bt[i])) for i in H100_IDS] == BSEL["harmful100"]
gate["h100_all_unsafe"] = all(not by_bt[i]["is_safe"] for i in H100_IDS)
gate["new100_all_unsafe"] = all(not by_bt[i]["is_safe"] for i in NEW100)
# d4 추출 재현: Random(SEED+1).sample(유해 풀,400)에서 누출 2행을 건너뛴 앞 200
pool_unsafe = [r for r in bt if not r["is_safe"]]
seq = random.Random(SEED + 1).sample(pool_unsafe, 400)
gate["h100_equals_seq_first100"] = [r["rid"] for r in seq[:100]] == H100_IDS
gate["h200_equals_seq_skip_leak"] = [r["rid"] for r in seq[:202] if r["rid"] not in C.LEAK_IDS] == H100_IDS + NEW100
K2400 = random.Random(SEED).sample(ku, 2400)
gate["clean2000_within_first2400"] = set(CLEAN) <= {r["rid"] for r in K2400}
print("gate1", gate, round(time.time() - t0), "s", flush=True)
assert all(gate.values()), gate

tok = AutoTokenizer.from_pretrained(C.AYA_SNAP)
TEMPLATE = tok.chat_template["default"] if isinstance(tok.chat_template, dict) else tok.chat_template
tok.chat_template = TEMPLATE


# ---------------- 짝짓기(d1_controls.py와 같은 코드) ----------------
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
gate2 = dict(pool_after_F=len(pool), pool_equals_d1=len(pool) == D1["pool"]["pool_after_F"])


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
used = set()


def match(harm_rows):
    hu = batch_len([r["u"] for r in harm_rows]); ha = batch_len([r["a"] for r in harm_rows])
    picked, log = [], []
    for j, h in enumerate(harm_rows):
        best = None
        c = bisect.bisect_left(keys, ha[j])
        for direction in (1, -1):
            k = c if direction == 1 else c - 1
            while 0 <= k < len(order):
                i = order[k]
                da = abs(pa[i] - ha[j])
                if best is not None and da > best[0]:
                    break
                if i not in used:
                    cand = (da + abs(pu[i] - hu[j]), rank[i], i)
                    if best is None or cand[:2] < best[:2]:
                        best = cand
                k += direction
        i = best[2]; used.add(i); picked.append(pool[i])
        log.append(dict(harm=h["rid"], harm_u=hu[j], harm_a=ha[j], ku=pool[i]["rid"], ku_src=pool[i]["src"], ku_u=pu[i], ku_a=pa[i], dist=best[0]))
    return picked, log


D100, log100 = match([by_bt[i] for i in H100_IDS])
gate2["c100_slot_reproduced"] = [r["rid"] for r in D100] == D100_REF
gate2["c100_match_log_reproduced"] = log100 == D1["match"]["log"]
print("gate2", gate2, round(time.time() - t0), "s", flush=True)
assert all(v for k, v in gate2.items() if k != "pool_after_F"), gate2
D_new, log_new = match([by_bt[i] for i in NEW100])
D200 = D100 + D_new
assert len({r["rid"] for r in D200}) == 200 and not ({r["rid"] for r in D200} & set(CLEAN))
print("matched new100", round(time.time() - t0), "s", flush=True)

# ---------------- 파일 쓰기 ----------------
SPEC = {
    "T100": dict(desc="처리군: 정제 kullm 2,000 + BeaverTails 유해 100", slot_part="harm", slot_rids=H100_IDS),
    "C100": dict(desc="대조군: 정제 kullm 2,000 + 유해 100과 길이 맞춘 kullm 100(D-1 (d))", slot_part="kullm_lenmatched", slot_rids=[r["rid"] for r in D100]),
    "T200": dict(desc="처리군: 정제 kullm 2,000 + 유해 200(앞 100 + D-4 새 100)", slot_part="harm", slot_rids=H100_IDS + NEW100),
    "C200": dict(desc="대조군: 정제 kullm 2,000 + 유해 200과 길이 맞춘 kullm 200(앞 100 = C100의 100, 뒤 100 = 이번에 같은 규칙으로 새로 고름)",
                 slot_part="kullm_lenmatched", slot_rids=[r["rid"] for r in D200]),
}
ku_t = {r["rid"]: (r["u"], r["a"], r["src"]) for r in ku}
bt_t = {r["rid"]: (r["u"], r["a"], r["src"], r["is_safe"]) for r in bt}
arms = {}
for name, s in SPEC.items():
    spec = dict(file=f"{name}.jsonl", desc=s["desc"], clean_rids=CLEAN, slot_part=s["slot_part"], slot_rids=s["slot_rids"])
    rows = C.arm_rows(spec, ku_t, bt_t)
    b = C.jsonl_bytes(rows)
    open(os.path.join(OUT, spec["file"]), "wb").write(b)
    spec["sha256"] = C.sha256_bytes(b); spec["bytes"] = len(b)
    # SafeTune 실제 labels
    ds = tokenize_dataset(Dataset.from_list([{"messages": r["messages"]} for r in rows]), tok, max_len=C.MAX_LEN)
    assert len(ds) == len(rows), (name, len(ds), len(rows))  # 빈 응답 없음 → 건너뛴 행 없음, 순서 같음
    lab = [sum(1 for t in row if t != -100) for row in ds["labels"]]
    n = len(rows)
    chk = C.check_arm_rows(rows, 2100 if name.endswith("100") else 2200, 100 if name == "T100" else (200 if name == "T200" else 0))
    chk.pop("clean2000")
    chk.update(safetune_rows=len(ds), zero_label_rows=sum(v == 0 for v in lab), label_tokens_total=sum(lab),
               label_tokens_mean=round(sum(lab) / n, 1), slot_label_tokens_total=sum(lab[2000:]),
               steps_per_epoch=math.ceil(len(ds) / C.BATCH), steps_total_3ep=3 * math.ceil(len(ds) / C.BATCH),
               truncated_rows_768=sum(1 for row in ds["attention_mask"] if sum(row) >= C.MAX_LEN),
               slot_src=dict(collections.Counter(r["src"] for r in rows[2000:])))
    spec["checks"] = chk
    arms[name] = spec
    print(name, {k: v for k, v in chk.items()}, round(time.time() - t0), "s", flush=True)

pairs = {}
for t, c in (("T100", "C100"), ("T200", "C200")):
    pairs[f"{t}_vs_{c}"] = dict(
        clean2000_identical_in_order=arms[t]["clean_rids"] == arms[c]["clean_rids"],
        rows_equal=arms[t]["checks"]["rows"] == arms[c]["checks"]["rows"],
        steps_equal=arms[t]["checks"]["steps_per_epoch"] == arms[c]["checks"]["steps_per_epoch"],
        label_tokens_treat=arms[t]["checks"]["label_tokens_total"], label_tokens_ctrl=arms[c]["checks"]["label_tokens_total"],
        label_tokens_diff_ctrl_minus_treat=arms[c]["checks"]["label_tokens_total"] - arms[t]["checks"]["label_tokens_total"],
        label_tokens_diff_pct=round(100 * (arms[c]["checks"]["label_tokens_total"] - arms[t]["checks"]["label_tokens_total"]) / arms[t]["checks"]["label_tokens_total"], 4),
        slot_pairing="slot_idx j: 처리군 유해 j번 ↔ 대조군 kullm j번(짝짓기 순서). 같은 시드·같은 행 수라 Trainer 섞기 순열이 같으면 같은 스텝에 들어감(포드에서 미확인)")
pairs["C100_slot_equals_C200_first100"] = arms["C100"]["slot_rids"] == arms["C200"]["slot_rids"][:100]
pairs["T100_slot_equals_T200_first100"] = arms["T100"]["slot_rids"] == arms["T200"]["slot_rids"][:100]
pairs["d_reference_T100_C100_label_tokens"] = [AGG["D_controls"]["cands"]["Tclean_treatment"]["L768"]["label_tokens_total"],
                                               AGG["D_controls"]["cands"]["dclean_control"]["L768"]["label_tokens_total"]]
pairs["T100_C100_match_D"] = [arms["T100"]["checks"]["label_tokens_total"], arms["C100"]["checks"]["label_tokens_total"]] == pairs["d_reference_T100_C100_label_tokens"]


def dist_q(xs):
    xs = sorted(xs); q = lambda p: xs[min(len(xs) - 1, int(len(xs) * p))]
    return dict(n=len(xs), p50=q(.5), p90=q(.9), max=max(xs), mean=round(sum(xs) / len(xs), 2))


manifest = dict(
    _note="E4 학습 파일 4개. 유해 문장이 들어 있어 파일 자체는 공개하지 않는다(이 매니페스트에는 행 id와 sha256만 있다). 행 id: ku<줄번호>=nlpai-lab/kullm-v2@cddcb73 kullm-v2.jsonl, "
          "bt<줄번호>=PKU-Alignment/BeaverTails@8401fe6 round0/30k/train.jsonl.gz (0-기반)",
    created_by="experiments/e4/scripts/e4_build_data.py", created_kst=time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
    revisions=C.REV, tokenizer=dict(repo="CohereLabs/tiny-aya-global", revision=C.REV["aya_global"], files="tokenizer·설정만(가중치 없음)", chat_template="default(기본 머리말 포함), system 메시지 없음"),
    train_cfg=dict(max_len=C.MAX_LEN, batch=C.BATCH, lr=C.LR, epochs=C.EPOCHS, lora=C.LORA, train_seed=C.TRAIN_SEED,
                   shuffle="safetune.configure(seed=20261111) → Trainer 기본 RandomSampler. 네 팔 같은 규칙. 파일 순서: 정제 kullm 2,000(같은 순서) 다음 슬롯(짝짓기 순서)",
                   primary_epoch=3, checkpoint_epochs=[1, 3]),
    row_format="{rid, part(kullm_clean|harm|kullm_lenmatched), slot_idx, src, messages[user, assistant]}. user=(instruction+'\\n'+input).strip() 또는 BeaverTails prompt",
    sources=dict(clean2000="agg_all.json D_controls.clean.clean2000_rids", harm100="B_beavertails/samples/selected_ids.json harmful100",
                 harm_new100="agg_all.json D_harm_refill.refill200.new_ids", c100_slot="agg_all.json D_controls.cands.dclean_control.slot.rids",
                 c200_new100="이번 실행에서 새로 고름(아래 c200_decision)"),
    c200_decision=dict(
        decision="유해 200 팔에도 길이 맞춘 무해 대조군 C200(2,200행)을 둔다. 앞 100 = C100의 100, 뒤 100 = 유해 새 100에 D-1과 같은 규칙으로 새로 고름",
        why="DATA_ANALYSIS.md 5절 13번이 '처리군·대조군 2,200행씩 다시 학습'으로 계획했다. 대조군 없이 T200만 돌리면 2,200행 학습 자체의 효과와 유해 200건의 효과를 가를 수 없다",
        rule="풀 133,112행(d1과 같음), |Δ응답 원문 토큰|+|Δ지시 원문 토큰| 최소, 동점 Random(20261115) 섞은 순서, 복원 없음, C100의 100 제외, 유해 new_ids 순서",
        note_200="200은 분석 근거가 없는 제안값이다(DATA_ANALYSIS.md 5절 13번: 'A·B·C에 이 숫자의 근거는 없음')",
        new100_match_dist=dist_q([m["dist"] for m in log_new]), new100_src=dict(collections.Counter(r["src"] for r in D_new)),
        new100_match_log=log_new, unread="새 유해 100과 새 kullm 100은 아무도 읽지 않았다. 같은 틀 점검도 하지 않았다"),
    gates=dict(ids=gate, matching=gate2), pairs=pairs, arms=arms,
    eval_inputs=dict(e1_items=dict(path="experiments/e1e3/e1_items.jsonl", sha256=C.sha256_file(os.path.join(PROJ, "experiments/e1e3/e1_items.jsonl"))),
                     xstest_items=dict(path="experiments/e1e3/xstest_items.jsonl", sha256=C.sha256_file(os.path.join(PROJ, "experiments/e1e3/xstest_items.jsonl"))),
                     e1_judge_ref=dict(path="experiments/e1e3/e1_judge.jsonl", sha256=C.sha256_file(os.path.join(PROJ, "experiments/e1e3/e1_judge.jsonl")))),
    runtime_sec=round(time.time() - t0, 1),
)
json.dump(manifest, open(os.path.join(OUT, "data_manifest.json"), "w"), ensure_ascii=False, indent=1)
summary = {a: dict(rows=s["checks"]["rows"], zero=s["checks"]["zero_label_rows"], label_tokens=s["checks"]["label_tokens_total"],
                   steps=s["checks"]["steps_per_epoch"], harm_pct=s["checks"]["harm_share_pct"], leak=s["checks"]["leak_rows_present"], sha=s["sha256"][:16]) for a, s in arms.items()}
print(json.dumps(dict(summary=summary, pairs={k: v for k, v in pairs.items()}), ensure_ascii=False, indent=1))
print("done", manifest["runtime_sec"], "s")
