"""E1~E3 평가 하네스(11-11, RunPod CUDA). 생성(Tiny Aya Global)과 판정(Qwen3Guard-Gen-4B)을 건별 jsonl로 저장한다.

사용:
  uv run python -I scripts/e1e3_harness.py gen   <items.jsonl> <out.jsonl> [--bs 32] [--max-new 128]
  uv run python -I scripts/e1e3_harness.py judge <gen.jsonl>   <out.jsonl> [--bs 32]
입력 items.jsonl 한 줄 = {"id": 고유키, "prompt": 모델에 보낼 user 메시지, ...그 밖의 필드는 그대로 보존}.
judge 입력 = gen 출력. 판정기에는 "judge_query"(변형 전 원문 요청, 없으면 prompt)와 "response"를 준다(리포트 §5-2).
이미 out에 있는 id는 건너뛴다(끊겨도 이어서 돈다). 실행마다 <out>.meta.json에 설정·리비전·버전·시각을 남긴다.
모델은 리비전 sha의 스냅샷 경로로 오프라인 로드한다. 토큰을 읽지 않는다.
"""
import argparse, json, os, re, sys, time, datetime

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

HUB = os.path.expanduser("~/.cache/huggingface/hub")
AYA = ("CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324")
GUARD = ("Qwen/Qwen3Guard-Gen-4B", "6ec42827da0c1ff11e7a49dc269d2e810d27e108")
SEED = 0  # 탐욕 디코딩이라 결과에 영향은 없지만 기록·고정한다


def snap(repo, rev):
    return os.path.join(HUB, "models--" + repo.replace("/", "--"), "snapshots", rev)


def load(repo, rev):
    p = snap(repo, rev)
    tok = AutoTokenizer.from_pretrained(p)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map="cuda").eval()
    return tok, m


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_meta(out, kind, cfg, extra=None):
    meta = dict(
        kind=kind, started_kst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec="seconds"),
        cfg=cfg, gen_model=dict(repo=AYA[0], revision=AYA[1]), judge_model=dict(repo=GUARD[0], revision=GUARD[1]),
        versions=dict(python=sys.version.split()[0], torch=torch.__version__, transformers=transformers.__version__,
                      cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        argv=sys.argv)
    if extra:
        meta.update(extra)
    mp = out + ".meta.json"
    runs = json.load(open(mp)) if os.path.exists(mp) else []
    runs.append(meta)
    json.dump(runs, open(mp, "w"), ensure_ascii=False, indent=1)
    return mp


def finish_meta(out, extra):
    mp = out + ".meta.json"
    runs = json.load(open(mp))
    runs[-1].update(extra)
    runs[-1]["finished_kst"] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec="seconds")
    json.dump(runs, open(mp, "w"), ensure_ascii=False, indent=1)


@torch.no_grad()
def cmd_gen(a):
    items = read_jsonl(a.items)
    done = {r["id"] for r in read_jsonl(a.out)}
    todo = [it for it in items if it["id"] not in done]
    cfg = dict(do_sample=False, temperature=None, max_new_tokens=a.max_new, batch_size=a.bs, seed=SEED,
               chat_template="tokenizer default, add_generation_prompt=True, system 메시지 미지정 -> 템플릿 기본 머리말(System/Default Preamble, 약 365토큰)이 자동으로 들어감", dtype="bfloat16", padding_side="left")
    write_meta(a.out, "gen", cfg, dict(items=a.items, n_items=len(items), n_todo=len(todo)))
    print(f"gen: {len(items)} items, {len(todo)} todo", flush=True)
    if not todo:
        return
    torch.manual_seed(SEED)
    tok, m = load(*AYA)
    eos_ids = {tok.eos_token_id} | set(m.generation_config.eos_token_id if isinstance(m.generation_config.eos_token_id, list) else [m.generation_config.eos_token_id])
    tot_tok, tot_sec = 0, 0.0
    with open(a.out, "a", encoding="utf-8") as fo:
        for i in range(0, len(todo), a.bs):
            batch = todo[i:i + a.bs]
            texts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False, add_generation_prompt=True) for it in batch]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            torch.cuda.synchronize(); t = time.time()
            out = m.generate(**enc, do_sample=False, max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize(); dt = time.time() - t
            new = out[:, enc["input_ids"].shape[1]:]
            n_new_batch = 0
            for it, row in zip(batch, new):
                ids = row.tolist()
                # 첫 EOS(또는 pad)까지가 실제 생성 토큰
                n = len(ids)
                hit_eos = False
                for k, x in enumerate(ids):
                    if x in eos_ids or x == tok.pad_token_id:
                        n, hit_eos = k, True
                        break
                n_new_batch += n
                resp = tok.decode(ids[:n], skip_special_tokens=True).strip()
                rec = dict(it, response=resp, n_new_tokens=n, truncated=(not hit_eos and n >= a.max_new))
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            tot_tok += n_new_batch; tot_sec += dt
            print(f"gen batch {i // a.bs + 1}/{(len(todo) + a.bs - 1) // a.bs}: {len(batch)} items {dt:.1f}s {n_new_batch / dt:.1f} tok/s", flush=True)
    finish_meta(a.out, dict(gen_new_tokens=tot_tok, gen_sec=round(tot_sec, 2), gen_tok_per_s=round(tot_tok / max(tot_sec, 1e-9), 1),
                            gen_items=len(todo), gen_sec_per_item=round(tot_sec / len(todo), 3),
                            peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2)))


SAFETY_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.I)
REFUSAL_RE = re.compile(r"Refusal:\s*(Yes|No)", re.I)
CAT_RE = re.compile(r"Categories:\s*(.*)", re.I)


def parse_guard(text):
    s = SAFETY_RE.search(text); r = REFUSAL_RE.search(text); c = CAT_RE.search(text)
    return dict(safety=s.group(1).capitalize() if s else None,
                refusal=(r.group(1).lower() == "yes") if r else None,
                categories=c.group(1).strip() if c else None)


@torch.no_grad()
def cmd_judge(a):
    items = read_jsonl(a.items)
    done = {r["id"] for r in read_jsonl(a.out)}
    todo = [it for it in items if it["id"] not in done]
    cfg = dict(do_sample=False, max_new_tokens=a.max_new, batch_size=a.bs, input="user=judge_query(원문 요청), assistant=response",
               labels="Safety: Safe|Unsafe|Controversial, Refusal: Yes|No (Qwen3Guard-Gen 응답 판정 형식)")
    write_meta(a.out, "judge", cfg, dict(items=a.items, n_items=len(items), n_todo=len(todo)))
    print(f"judge: {len(items)} items, {len(todo)} todo", flush=True)
    if not todo:
        return
    tok, m = load(*GUARD)
    tot_sec = 0.0
    with open(a.out, "a", encoding="utf-8") as fo:
        for i in range(0, len(todo), a.bs):
            batch = todo[i:i + a.bs]
            texts = [tok.apply_chat_template([{"role": "user", "content": it.get("judge_query") or it["prompt"]},
                                              {"role": "assistant", "content": it["response"]}], tokenize=False) for it in batch]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            torch.cuda.synchronize(); t = time.time()
            out = m.generate(**enc, do_sample=False, max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize(); dt = time.time() - t
            tot_sec += dt
            for it, row in zip(batch, out[:, enc["input_ids"].shape[1]:]):
                raw = tok.decode(row, skip_special_tokens=True).strip()
                rec = dict(id=it["id"], guard_raw=raw, **parse_guard(raw))
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            print(f"judge batch {i // a.bs + 1}/{(len(todo) + a.bs - 1) // a.bs}: {dt:.1f}s ({dt / len(batch):.3f}s/item)", flush=True)
    finish_meta(a.out, dict(judge_sec=round(tot_sec, 2), judge_items=len(todo), judge_sec_per_item=round(tot_sec / len(todo), 4),
                            peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "judge"])
    ap.add_argument("items"); ap.add_argument("out")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=None)
    a = ap.parse_args()
    if a.max_new is None:
        a.max_new = 128 if a.cmd == "gen" else 64
    {"gen": cmd_gen, "judge": cmd_judge}[a.cmd](a)
