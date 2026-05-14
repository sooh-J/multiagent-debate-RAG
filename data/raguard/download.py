"""
UCSC-IRKM/RAGuard 데이터셋 다운로드 및 로드

데이터셋: https://huggingface.co/datasets/UCSC-IRKM/RAGuard

claims.csv와 documents.csv가 서로 다른 스키마라 load_dataset 한 번으로는
불러올 수 없어, 두 파일을 각각 받아 Claim ID 기준으로 조인합니다.
"""

import csv
import json
import random
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
FULL_DIR = DATA_DIR / "full"
SAMPLE_DIR = DATA_DIR / "sample"

REPO_ID = "UCSC-IRKM/RAGuard"
RAW_FILES = ["claims.csv", "documents.csv"]


def _download_raw():
    """원본 CSV 파일을 raw/ 에 그대로 보관 (확인용)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fname in RAW_FILES:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            repo_type="dataset",
            local_dir=RAW_DIR,
        )
        paths[fname] = Path(local_path)
        print(f"raw 다운로드: {paths[fname]}")
    return paths


def _read_csv(path: Path):
    # 큰 셀에 대비해 필드 사이즈 한도 상향
    csv.field_size_limit(sys.maxsize)
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _build_merged_json(raw_paths):
    """claims + documents → Claim ID 기준 조인된 JSON."""
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    claims = _read_csv(raw_paths["claims.csv"])
    documents = _read_csv(raw_paths["documents.csv"])

    docs_by_claim = {}
    for d in documents:
        docs_by_claim.setdefault(d["Claim ID"], []).append(d)

    merged = []
    for c in claims:
        merged.append({
            "claim_id": c["Claim ID"],
            "claim": c["Claim"],
            "verdict": c["Verdict"],
            "original_verdict": c.get("Original Verdict"),
            "document_ids": c.get("Document IDs"),
            "document_labels": c.get("Document Labels"),
            "documents": docs_by_claim.get(c["Claim ID"], []),
        })

    out_path = FULL_DIR / "raguard.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {out_path} (claims {len(merged)}개, docs {len(documents)}개)")
    return merged


def _load_from_local():
    with open(FULL_DIR / "raguard.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_raguard(n_samples: int = 20):
    """
    UCSC-IRKM/RAGuard 데이터셋 로드.
    - data/raguard/raw/    : 원본 CSV (확인용)
    - data/raguard/full/   : Claim ID 조인된 JSON
    - data/raguard/sample/ : n_samples개 샘플 (git 추적용)
    """
    if not (FULL_DIR / "raguard.json").exists():
        print("로컬 데이터 없음 → HuggingFace에서 다운로드...")
        raw_paths = _download_raw()
        _build_merged_json(raw_paths)

    data = _load_from_local()
    random.seed(42)
    samples = random.sample(data, min(n_samples, len(data)))

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = SAMPLE_DIR / f"sample_{n_samples}.json"
    if not sample_path.exists():
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"샘플 저장: {sample_path}")

    return samples


if __name__ == "__main__":
    samples = load_raguard(n_samples=20)
    print(f"다운로드 완료. 샘플 {len(samples)}개 저장됨.")
