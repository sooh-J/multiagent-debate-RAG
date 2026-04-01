import json
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FULL_DATA_DIR = DATA_DIR / "full"
SAMPLE_DATA_DIR = DATA_DIR / "sample"


def _download_and_save():
    """HuggingFace에서 다운로드 후 로컬에 저장"""
    FULL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("HanNight/RAMDocs")
    for split in ds:
        path = FULL_DATA_DIR / f"{split}.json"
        data = [row for row in ds[split]]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"저장 완료: {path} ({len(data)}개)")
    return ds


def _load_from_local(split: str = None):
    """로컬 JSON에서 로드"""
    if split:
        path = FULL_DATA_DIR / f"{split}.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # split 미지정 시 첫 번째 파일
    for path in sorted(FULL_DATA_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def load_ramdocs(n_samples: int = 20):
    """
    RAMDocs 데이터셋 로드.
    - data/full/ 폴더가 있으면 로컬에서 읽음
    - 없으면 HuggingFace에서 다운로드 후 저장
    - n_samples개 샘플은 data/sample/에 별도 저장 (git 추적용)
    """
    # 전체 데이터 확보
    if not FULL_DATA_DIR.exists() or not list(FULL_DATA_DIR.glob("*.json")):
        print("로컬 데이터 없음 → HuggingFace에서 다운로드...")
        _download_and_save()

    data = _load_from_local()
    # document가 3개인 샘플만 필터링
    data = [row for row in data if len(row["documents"]) == 3]
    samples = data[:n_samples]

    # 샘플 데이터 저장 (git 추적용)
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = SAMPLE_DATA_DIR / f"sample_{n_samples}.json"
    if not sample_path.exists():
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"샘플 저장: {sample_path}")

    return samples


if __name__ == "__main__":
    samples = load_ramdocs(n_samples=20)
    print(f"다운로드 완료. 샘플 {len(samples)}개 저장됨.")
