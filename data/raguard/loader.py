"""
RAGuard 전처리본 로더 (run_*.py 에서 사용)

`data/raguard/preprocess.py` 가 만든 RAMDocs 스키마 JSON을 읽어 dict 리스트로 반환한다.
- balanced=True  -> raguard_preprocessed_balanced.json (230 claims, True 115 : False 115)
- balanced=False -> raguard_preprocessed.json          (711 claims, True 115 : False 596)
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
FULL_DIR = DATA_DIR / "full"

PATHS = {
    True:  FULL_DIR / "raguard_preprocessed_balanced.json",
    False: FULL_DIR / "raguard_preprocessed.json",
}

SEED = 42


def load_raguard(n_samples: int | None = None, balanced: bool = True):
    path = PATHS[balanced]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음. 먼저 다운로드·전처리를 실행하세요:\n"
            f"  python -m data.raguard.download\n"
            f"  python -m data.raguard.preprocess"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if n_samples and n_samples < len(data):
        rng = random.Random(SEED)
        return rng.sample(data, n_samples)
    return data
