"""
RAMDocs 데이터셋 다운로드 및 로드

데이터셋: https://huggingface.co/datasets/HanNight/RAMDocs
"""

import json
import random
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent
FULL_DIR = DATA_DIR / "full"
SAMPLE_DIR = DATA_DIR / "sample"


def _download_and_save():
    """HuggingFace에서 다운로드 후 로컬에 저장"""
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("HanNight/RAMDocs")
    for split in ds:
        path = FULL_DIR / f"{split}.json"
        data = [row for row in ds[split]]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"저장 완료: {path} ({len(data)}개)")


def _load_from_local():
    """로컬 JSON에서 로드"""
    for path in sorted(FULL_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def load_ramdocs(n_samples: int = 20):
    """
    RAMDocs 데이터셋 로드.
    - data/ramdocs/full/ 폴더가 있으면 로컬에서 읽음
    - 없으면 HuggingFace에서 다운로드 후 저장
    - n_samples개 샘플은 data/ramdocs/sample/에 별도 저장 (git 추적용)
    """
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = SAMPLE_DIR / f"sample_{n_samples}.json"

    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if not FULL_DIR.exists() or not list(FULL_DIR.glob("*.json")):
        print("로컬 데이터 없음 → HuggingFace에서 다운로드...")
        _download_and_save()

    data = _load_from_local()
    random.seed(42)
    samples = random.sample(data, min(n_samples, len(data)))

    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"샘플 저장: {sample_path}")

    return samples


if __name__ == "__main__":
    samples = load_ramdocs(n_samples=20)
    print(f"다운로드 완료. 샘플 {len(samples)}개 저장됨.")
