"""단위시험용 가짜 실행기(GPU 없음): e5_train.py·e5_eval.py·e5_auditkit.py 대신 끝남 표지와 가짜 라벨만 쓴다.
FAKE_FAIL(쉼표)에 든 팔·tag는 실패. FAKE_CALLS 파일에 호출을 적는다. FAKE_SEL_H="0.25:40,0.5:60,..." 로 α별 selh 거절 수, FAKE_SEL_M 로 MMLU sel 정답 수를 정한다."""
import json, os, sys

script, args = sys.argv[1], sys.argv[2:]
fail = set(filter(None, os.environ.get("FAKE_FAIL", "").split(",")))
if os.environ.get("FAKE_CALLS"):
    open(os.environ["FAKE_CALLS"], "a").write(os.path.basename(script) + " " + args[0] + "\n")
kv = lambda s: {k: int(v) for k, v in (x.split(":") for x in filter(None, s.split(",")))}
if script.endswith("e5_train.py"):
    arm, root = args[0], args[1]
    if arm in fail:
        sys.exit(1)
    d = os.path.join(root, "train", arm)
    for ep in (1, 3):
        os.makedirs(os.path.join(d, f"ep{ep}"), exist_ok=True)
        open(os.path.join(d, f"ep{ep}", "adapter_model.safetensors"), "wb").write(f"fake {arm} ep{ep}".encode())
        open(os.path.join(d, f"ep{ep}", "adapter_config.json"), "w").write("{}")
    json.dump(dict(arm=arm, fake=True), open(os.path.join(d, "DONE.json"), "w"))
elif script.endswith("e5_eval.py"):
    tag, spec, root = args[0], json.loads(args[1]), args[2]
    if tag in fail:
        sys.exit(1)
    d = os.path.join(root, "eval", tag); os.makedirs(d, exist_ok=True)
    al = str(spec.get("alpha", ""))
    H = kv(os.environ.get("FAKE_SEL_H", "")).get(al, 50); M = kv(os.environ.get("FAKE_SEL_M", "")).get(al, 150)
    B = kv(os.environ.get("FAKE_SEL_B", "")).get(al, 5)
    if tag == "base":
        H, M, B = 90, 150, 5
    n = dict(e1=315, mjen=315, xs=450)
    for part, k in n.items():
        with open(os.path.join(d, f"{part}_gen.jsonl"), "w") as g, open(os.path.join(d, f"{part}_judge.jsonl"), "w") as j:
            for i in range(k):
                pid = f"{part}-{i}" if part != "xs" else f"xs-{i}"
                g.write(json.dumps(dict(id=pid, response="x")) + "\n")
                j.write(json.dumps(dict(id=pid, safety="Safe", refusal=i % 3 == 0)) + "\n")
    with open(os.path.join(d, "sel_gen.jsonl"), "w") as g, open(os.path.join(d, "sel_judge.jsonl"), "w") as j:
        for i in range(100):
            for pre, kk in (("selh", H), ("selb", B)):
                g.write(json.dumps(dict(id=f"{pre}-{i}", response="x")) + "\n")
                j.write(json.dumps(dict(id=f"{pre}-{i}", safety="Safe", refusal=i < kk)) + "\n")
    with open(os.path.join(d, "mmlu.jsonl"), "w") as f:
        for i in range(600):
            sp = "rep" if i < 400 else "sel"
            f.write(json.dumps(dict(id=f"mmlu-{i}", split=sp, correct=(i - 400 < M) if sp == "sel" else i % 2 == 0)) + "\n")
    if spec["kind"] == "resta":
        rc = os.path.join(root, "release_cand", tag); os.makedirs(rc, exist_ok=True)
        open(os.path.join(rc, "model.safetensors"), "wb").write(f"fake merged {tag}".encode())
        open(os.path.join(rc, "config.json"), "w").write("{}")
        open(os.path.join(d, "SAVED_OK"), "w").write("x")
    json.dump(dict(tag=tag, fake=True), open(os.path.join(d, "DONE.json"), "w"))
elif script.endswith("e5_auditkit.py"):
    os.makedirs(args[1], exist_ok=True); json.dump({"fake": True}, open(os.path.join(args[1], "auditkit_compare.json"), "w"))
else:
    sys.exit(3)
