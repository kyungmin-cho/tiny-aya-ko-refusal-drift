"""Download every public input at a pinned revision into the default Hugging Face cache (~/.cache/huggingface/hub).

Usage:  uv run python -I scripts/download.py open             # no token needed
        HF_TOKEN=<token> uv run python -I scripts/download.py gated   # Tiny Aya Global/Base, XSTest (accept each gate on the Hub first)
The token is read from the environment only and never printed. Already-downloaded snapshots are skipped by huggingface_hub.
"""
import os, sys, time
from huggingface_hub import snapshot_download

OPEN = [
    ("dataset", "DAMO-NLP-SG/MultiJail", "93d65c778925d55973b9ff30c7bccc8545494eb6", None),          # MIT
    ("dataset", "CohereLabs/Global-MMLU", "0e619dbeb34206cd48705a1a0ea7fb21cae09993", ["ko/*", "README.md"]),  # Apache-2.0
    ("dataset", "nlpai-lab/kullm-v2", "cddcb73c259269928e974e0ce141f123eb068030", None),             # Apache-2.0
    ("dataset", "PKU-Alignment/BeaverTails", "8401fe609d288129cc684a9b3be6a93e41cfe678", ["round0/30k/*", "README.md"]),  # CC-BY-NC-4.0
    ("dataset", "leo-bjpark/AdvBench-Korean", "f6a20e02c7eceba1b2e684f556dddc1acdf5afc8", None),     # MIT (alpha-selection hold-out only)
    ("model", "Qwen/Qwen3Guard-Gen-4B", "6ec42827da0c1ff11e7a49dc269d2e810d27e108", None),          # Apache-2.0, the judge
]
GATED = [
    ("model", "CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324", None),      # CC-BY-NC-4.0 + Cohere Labs AUP
    ("model", "CohereLabs/tiny-aya-base", "d62ac17e241c871fc52e9a95d7b0e2615189d7d5", None),        # CC-BY-NC-4.0 + Cohere Labs AUP
    ("dataset", "walledai/XSTest", "f1d713187c61b6ae64e602d74f0b3d812cc2e8e8", None),               # CC-BY-4.0
]

which = sys.argv[1] if len(sys.argv) > 1 else "open"
items = OPEN if which == "open" else GATED
token = os.environ.get("HF_TOKEN") if which == "gated" else False
failed = []
for kind, rid, rev, pats in items:
    t = time.time()
    for attempt in range(1, 9):
        try:
            p = snapshot_download(rid, repo_type=kind, revision=rev, allow_patterns=pats, token=token, max_workers=4)
            print(f"OK {rid}@{rev[:10]} ({time.time() - t:.0f}s, try {attempt})", flush=True)
            break
        except Exception as e:  # noqa: BLE001 - retry transient network errors, stop on auth errors
            print(f"RETRY {attempt} {rid}: {type(e).__name__}", flush=True)
            if "401" in str(e) or "403" in str(e) or "Gated" in type(e).__name__:
                failed.append(rid); break
            time.sleep(10)
    else:
        failed.append(rid)
print("ALLDONE" if not failed else f"FAILED {failed}", flush=True)
sys.exit(1 if failed else 0)
