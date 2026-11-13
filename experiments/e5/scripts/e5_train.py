"""E5 학습(포드, CUDA): E4 학습 스크립트(PROJ/scripts/e4_train.py)를 고치지 않고 그대로 부르되 학습 시드만 E5 값으로 바꾼다.

바꾸는 것: e4_common.TRAIN_SEED = data_manifest.json train_cfg.train_seed(E5: 20261113, E4: 20261111).
  e4_train.main()은 실행 중에 C.TRAIN_SEED를 읽으므로(safetune.configure(seed=…)·args.seed assert·DONE.json cfg.seed) 이 값이 그대로 쓰인다.
나머지(LoRA r16·α32·dropout0.05, bs4, lr1e-4, max_len 768, 기본 머리말, 3에포크, ep1·ep3 어댑터 저장, 데이터 sha·행 수·학습 신호 0·학습 토큰 assert)는 E4와 같다.
사용: uv run python -I -B experiments/e5/scripts/e5_train.py <arm> <e5_root> [--save-merged]
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(PROJ, "scripts"))
import e4_common as C  # noqa: E402


def set_seed_from_manifest(root):
    man = json.load(open(os.path.join(root, "data", "data_manifest.json")))
    seed = int(man["train_cfg"]["train_seed"])
    assert seed != 20261111, "E5 시드는 E4(20261111)와 달라야 한다"
    C.TRAIN_SEED = seed
    return seed


if __name__ == "__main__":
    root = sys.argv[2]
    seed = set_seed_from_manifest(root)
    import e4_train  # noqa: E402  (같은 e4_common 모듈 객체를 쓴다)
    assert e4_train.C is C and e4_train.C.TRAIN_SEED == seed
    print(f"E5 train seed {seed}", flush=True)
    e4_train.main()
