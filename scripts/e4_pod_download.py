"""RunPod 포드에서 E4에 필요한 모델·데이터를 HF 기본 캐시에 리비전 고정으로 받는다(11-12, pod_download_e1e3.py에 tiny-aya-base를 더한 것).
사용: uv run python -I scripts/e4_pod_download.py
토큰 값은 출력하지 않는다. tiny-aya-global·tiny-aya-base는 게이트 항목이라 토큰이 필요하다. kanana-safeguard는 받지 않는다.
MultiJail·XSTest는 받지 않는다: 평가 입력은 업로드한 experiments/e1e3/e1_items.jsonl·xstest_items.jsonl(sha256 확인)을 쓴다."""
import os, time
from huggingface_hub import snapshot_download
ITEMS = [
    ("dataset", "nlpai-lab/kullm-v2", "cddcb73c259269928e974e0ce141f123eb068030", None),
    ("dataset", "PKU-Alignment/BeaverTails", "8401fe609d288129cc684a9b3be6a93e41cfe678", ["round0/30k/*", "README.md"]),
    ("model", "CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324", None),
    ("model", "CohereLabs/tiny-aya-base", "d62ac17e241c871fc52e9a95d7b0e2615189d7d5", None),
    ("model", "Qwen/Qwen3Guard-Gen-4B", "6ec42827da0c1ff11e7a49dc269d2e810d27e108", None),
]
token = os.environ.get("HF_TOKEN") or None
fail = []
for kind, rid, rev, pats in ITEMS:
    t = time.time()
    for attempt in range(1, 9):
        try:
            p = snapshot_download(rid, repo_type=kind, revision=rev, allow_patterns=pats, token=token, max_workers=8)
            print(f"OK {rid}@{rev[:10]} -> {p} ({time.time()-t:.0f}s, try {attempt})", flush=True)
            break
        except Exception as e:
            print(f"RETRY {attempt} {rid}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            if "401" in str(e) or "403" in str(e) or "Gated" in type(e).__name__:
                fail.append(rid); break
            time.sleep(10)
    else:
        fail.append(rid); print(f"FAIL {rid}", flush=True)
print("ALLDONE" if not fail else f"FAILED {fail}", flush=True)
