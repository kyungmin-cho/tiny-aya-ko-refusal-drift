"""E4 학습(포드, CUDA): SafeTune PlainSFTTrainer 경로로 한 팔(T100·C100·T200·C200)을 3에포크 LoRA 학습하고 1·3에포크 어댑터를 저장한다.

경로: e4_bench_lora.py와 같다. PlainSFTTrainer(epochs=3, batch_size=4, lr=1e-4)의 _lora_base()(load_model + lora_wrap r16·α32·dropout0.05)와
_training_args()(bf16, adamw_torch, save_strategy=no)를 쓰고, PlainSFTTrainer.train()과 같은 Trainer(_keep_model_columns, default_data_collator)를 직접 만든다.
PlainSFTTrainer.train()을 그대로 부르지 않는 이유: save_strategy=no라 끝에서 병합본 하나만 저장해 1에포크 체크포인트를 남길 수 없다.
그래서 에포크 끝 콜백으로 LoRA 어댑터만 ep1/·ep3/에 저장한다(병합본 6.7GB를 여러 벌 두지 않기 위해). 평가 때 원본 + 어댑터를 병합한다(e4_models.py).
--save-merged를 주면 끝에 SafeTune _save_merged(병합 + lexsi_provenance.json)도 부른다.
시드: safetune.configure(seed=20261111) 하나로 LoRA 초기화와 Trainer 행 섞기를 고정한다(네 팔 같음).
학습 전에 data_manifest.json과 sha256·행 수·학습 신호 0 행(0)·학습 토큰 합을 assert한다(맥 검증과 같은 tokenize_dataset 호출).
사용: uv run python -I scripts/e4_train.py <arm> <e4_root> [--save-merged]
출력: <e4_root>/train/<arm>/{ep1,ep3}/(adapter_config.json, adapter_model.safetensors), log_history.json, DONE.json
"""
import argparse, dataclasses, datetime, json, os, shutil, sys, time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("WANDB_DISABLED", "true")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_common as C  # noqa: E402

KST = datetime.timezone(datetime.timedelta(hours=9))


def now():
    return datetime.datetime.now(KST).isoformat(timespec="seconds")


def tokenize_arm(rows, tok):
    from datasets import Dataset
    from safetune.runner.utils.data_utils import tokenize_dataset
    return tokenize_dataset(Dataset.from_list([{"messages": r["messages"]} for r in rows]), tok, max_len=C.MAX_LEN)


def load_tok():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.AYA_SNAP)
    tmpl = tok.chat_template["default"] if isinstance(tok.chat_template, dict) else tok.chat_template
    tok.chat_template = tmpl  # e4_build_data.py·d1_controls.py와 같게 기본 템플릿(머리말 포함)을 명시
    return tok


def verify(arm, root, tok):
    man = json.load(open(os.path.join(root, "data", "data_manifest.json")))
    spec = man["arms"][arm]
    path = os.path.join(root, "data", spec["file"])
    got = C.sha256_file(path)
    assert got == spec["sha256"], f"{arm} sha256 {got[:16]} != {spec['sha256'][:16]}"
    rows = C.read_jsonl(path)
    ds = tokenize_arm(rows, tok)
    lab = [sum(1 for t in row if t != -100) for row in ds["labels"]]
    chk = dict(rows=len(rows), safetune_rows=len(ds), zero_label_rows=sum(v == 0 for v in lab), label_tokens_total=sum(lab))
    exp = {k: spec["checks"][k] for k in chk}
    assert chk == exp, f"{arm} 데이터 검증 불일치: {chk} != {exp}"
    return rows, ds, spec, chk


def make_epoch_saver(out, save_epochs):
    """에포크 끝마다(1·3) LoRA 어댑터만 out/ep<k>/에 저장하는 Trainer 콜백. 맥 CPU 작은 모델 시험(tests/cpu_smoke_tiny.py)이 같은 함수를 쓴다."""
    from transformers import TrainerCallback

    class EpochSaver(TrainerCallback):
        def __init__(self):
            self.saved, self.t_epoch = {}, time.time()

        def on_epoch_end(self, args_, state, control, model=None, **kw):
            ep = int(round(state.epoch))
            if ep in save_epochs:
                d = os.path.join(out, f"ep{ep}")
                model.save_pretrained(d)  # PeftModel → 어댑터만
                self.saved[ep] = dict(dir=d, global_step=state.global_step, saved_kst=now(),
                                      epoch_sec=round(time.time() - self.t_epoch, 1))
                json.dump(self.saved, open(os.path.join(out, "saved_epochs.json"), "w"), indent=1)
                print(f"saved ep{ep} adapter at step {state.global_step}", flush=True)
            self.t_epoch = time.time()

    return EpochSaver()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=C.ARMS)
    ap.add_argument("root")
    ap.add_argument("--save-merged", action="store_true")
    a = ap.parse_args()
    out = os.path.join(a.root, "train", a.arm)
    if os.path.exists(os.path.join(out, "DONE.json")):
        print(f"{a.arm}: DONE.json 있음, 건너뜀"); return
    os.makedirs(out, exist_ok=True)
    import torch
    import safetune
    from transformers import Trainer, default_data_collator
    from safetune.runner.harden import PlainSFTTrainer
    from safetune.runner.harden._base import _keep_model_columns
    from safetune.runner.utils.model_utils import load_model

    t0 = time.time(); started = now()
    safetune.configure(seed=C.TRAIN_SEED)
    tok = load_tok()
    rows, ds, spec, chk = verify(a.arm, a.root, tok)
    print(f"{a.arm}: 데이터 검증 통과 {chk}", flush=True)

    model = load_model(C.AYA_SNAP)
    cfg = PlainSFTTrainer(model=model, tokenizer=tok, model_id=C.AYA_SNAP, epochs=C.EPOCHS, batch_size=C.BATCH, lr=C.LR,
                          results_dir=os.path.join(out, "safetune_results"))
    lora = cfg._lora_base()
    pc = lora.peft_config["default"]
    assert (pc.r, pc.lora_alpha, pc.lora_dropout) == (C.LORA["r"], C.LORA["alpha"], C.LORA["dropout"]), pc
    tmp = os.path.join(out, "_trainer_tmp")
    args = dataclasses.replace(cfg._training_args(tmp), logging_steps=25)
    assert args.seed == C.TRAIN_SEED and args.save_strategy == "no" and args.per_device_train_batch_size == C.BATCH, args
    load_sec = time.time() - t0

    saver = make_epoch_saver(out, (1, C.EPOCHS))
    torch.cuda.reset_peak_memory_stats()
    tr = Trainer(model=lora, args=args, train_dataset=_keep_model_columns(ds.with_format("torch")),
                 data_collator=default_data_collator, callbacks=[saver])
    t1 = time.time()
    tr.train()
    train_sec = time.time() - t1
    json.dump(tr.state.log_history, open(os.path.join(out, "log_history.json"), "w"), indent=1)
    merged = None
    if a.save_merged:
        merged = cfg._save_merged(lora, os.path.join(out, "ep3_merged"))
    shutil.rmtree(tmp, ignore_errors=True)
    for ep in (1, C.EPOCHS):
        assert os.path.exists(os.path.join(out, f"ep{ep}", "adapter_model.safetensors")), ep
    losses = [h["loss"] for h in tr.state.log_history if "loss" in h]
    done = dict(arm=a.arm, desc=spec["desc"], started_kst=started, finished_kst=now(), data_sha256=spec["sha256"], data_checks=chk,
                cfg=dict(max_len=C.MAX_LEN, batch=C.BATCH, lr=C.LR, epochs=C.EPOCHS, lora=C.LORA, seed=C.TRAIN_SEED,
                         bf16=args.bf16, optim=args.optim, system_message=None, chat_template="default(기본 머리말)"),
                global_steps=tr.state.global_step, expected_steps=C.EPOCHS * spec["checks"]["steps_per_epoch"],
                saved=saver.saved, merged=str(merged) if merged else None, load_sec=round(load_sec, 1), train_sec=round(train_sec, 1),
                sec_per_step=round(train_sec / max(tr.state.global_step, 1), 4), first_loss=losses[0] if losses else None,
                last_loss=losses[-1] if losses else None, nan_losses=sum(1 for x in losses if x != x),
                peak_mem_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                adapter_sha256={ep: C.sha256_file(os.path.join(out, f"ep{ep}", "adapter_model.safetensors")) for ep in (1, C.EPOCHS)},
                versions=dict(torch=torch.__version__, safetune=safetune.__version__, cuda=torch.version.cuda,
                              gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    assert done["global_steps"] == done["expected_steps"], (done["global_steps"], done["expected_steps"])
    json.dump(done, open(os.path.join(out, "DONE.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in done.items() if k not in ("saved",)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
