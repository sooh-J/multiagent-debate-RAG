"""
RAGuard 전처리 — RAMDocs 스키마로 변환

입력: data/raguard/full/raguard.json (download.py로 받은 조인본)
출력: data/raguard/full/raguard_preprocessed.json
       data/raguard/sample/sample_N_preprocessed.json

전처리 규칙 (자세한 설명은 data/raguard/README.md 참고):
1. Full Text(=body)가 비어있거나 placeholder([Link Post]/[deleted]/[removed]) /
   URL만 / 마크다운 링크만 / 공백만 / 의미 있는 영숫자 < 30글자인 doc 제거
   - body가 필수. title만 있고 body 없으면 doc 폐기
2. 위 결과로 supporting(=correct) 문서가 하나도 남지 않은 claim 제거
3. (크롤링 없음 — 본문 못 살리는 doc은 그냥 버림)
4. Document Label → type
     supporting → correct, misleading → misinfo, unrelated → noise
5. answer 계산
     correct → verdict 그대로 ("True"/"False")
     misinfo → verdict 반전
     noise   → "unknown"
6. original_verdict == "half-true" 인 claim 제외
7. 유효 doc 수 < 2 인 claim 제외

추가로, 전처리 후 verdict 분포가 unbalanced(True 115 : False 596)이므로,
original_verdict 기준 stratified downsampling을 적용한 balanced 버전도 함께 저장.
"""

import collections
import json
import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
FULL_DIR = DATA_DIR / "full"
SAMPLE_DIR = DATA_DIR / "sample"

INPUT_PATH = FULL_DIR / "raguard.json"
OUTPUT_PATH = FULL_DIR / "raguard_preprocessed.json"
OUTPUT_PATH_BALANCED = FULL_DIR / "raguard_preprocessed_balanced.json"

SEED = 42

PLACEHOLDER_TEXTS = {
    "[link post]",
    "[deleted]",
    "[removed]",
    "[deleted by user]",
    "[removed by reddit]",
}
URL_PATTERN = re.compile(r"https?://\S+")
MD_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
# URL/마크다운 링크 등 noise를 걷어낸 후 남는 영숫자 글자 수 임계값
MIN_ALNUM_CHARS = 30

LABEL_TO_TYPE = {
    "supporting": "correct",
    "misleading": "misinfo",
    "unrelated": "noise",
}

QUESTION_TEMPLATE = (
    'Is the following claim true or false? Answer with "True" or "False".\n\n'
    'Claim: "{claim}"'
)

MIN_DOCS_PER_CLAIM = 2
N_SAMPLES = 20


def _build_text(title: str, body: str):
    """body가 의미 있는 문자열일 때만 (title + body) 또는 body 반환. 아니면 None."""
    body = (body or "").strip()
    if not body:
        return None
    if body.lower() in PLACEHOLDER_TEXTS:
        return None
    cleaned = MD_LINK_PATTERN.sub("", body)
    cleaned = URL_PATTERN.sub("", cleaned)
    alnum_count = sum(1 for c in cleaned if c.isalnum())
    if alnum_count < MIN_ALNUM_CHARS:
        return None
    title = (title or "").strip()
    return f"{title}\n\n{body}" if title else body


def _compute_answer(label: str, verdict: str) -> str:
    if label == "supporting":
        return verdict
    if label == "misleading":
        return _flip(verdict)
    return "unknown"


def _flip(verdict: str) -> str:
    return "False" if verdict == "True" else "True"


def _stratified_sample(items, key_fn, n_total: int, seed: int = SEED):
    """key 별 비율을 유지하며 n_total개 샘플링. 반올림 오차는 leftover에서 보충/제거."""
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for it in items:
        groups[key_fn(it)].append(it)

    total = len(items)
    selected = []
    leftovers = []
    for group in groups.values():
        n = round(len(group) * n_total / total)
        n = min(n, len(group))
        rng.shuffle(group)
        selected.extend(group[:n])
        leftovers.extend(group[n:])

    if len(selected) < n_total:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n_total - len(selected)])
    elif len(selected) > n_total:
        rng.shuffle(selected)
        selected = selected[:n_total]
    return selected


def _balance(items_with_meta):
    """True 그룹은 모두 살리고, False 그룹은 original_verdict 기준 stratified sampling."""
    true_items = [x for x in items_with_meta if x[0]["gold_answers"][0] == "True"]
    false_items = [x for x in items_with_meta if x[0]["gold_answers"][0] == "False"]

    n_per_class = min(len(true_items), len(false_items))
    rng = random.Random(SEED)
    rng.shuffle(true_items)
    true_selected = true_items[:n_per_class]
    false_selected = _stratified_sample(
        false_items, key_fn=lambda x: x[1], n_total=n_per_class
    )
    return [x[0] for x in true_selected + false_selected]


def preprocess():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        claims = json.load(f)

    stats = {
        "input_claims": len(claims),
        "input_docs": sum(len(c["documents"]) for c in claims),
        "dropped_docs_invalid_body": 0,
        "dropped_claims_half_true": 0,
        "dropped_claims_no_supporting": 0,
        "dropped_claims_too_few_docs": 0,
    }

    out = []
    out_meta = []  # downsampling 시 stratify 기준으로 쓸 original_verdict
    for c in claims:
        # 6. half-true 제외 (효율상 먼저 거름)
        if c.get("original_verdict") == "half-true":
            stats["dropped_claims_half_true"] += 1
            continue

        verdict = c["verdict"]

        # 1 + 4 + 5: doc 단위 필터·변환
        valid_docs = []
        for doc in c["documents"]:
            text = _build_text(doc.get("Title", ""), doc.get("Full Text", ""))
            if text is None:
                stats["dropped_docs_invalid_body"] += 1
                continue
            label = doc.get("Document Label", "unrelated")
            valid_docs.append({
                "text": text,
                "type": LABEL_TO_TYPE.get(label, "noise"),
                "answer": _compute_answer(label, verdict),
            })

        # 2. supporting(=correct) 없는 claim 제외
        if not any(d["type"] == "correct" for d in valid_docs):
            stats["dropped_claims_no_supporting"] += 1
            continue

        # 7. doc 2개 미만 제외
        if len(valid_docs) < MIN_DOCS_PER_CLAIM:
            stats["dropped_claims_too_few_docs"] += 1
            continue

        out.append({
            "question": QUESTION_TEMPLATE.format(claim=c["claim"]),
            "documents": valid_docs,
            "disambig_entity": [],
            "gold_answers": [verdict],
            "wrong_answers": [_flip(verdict)],
        })
        out_meta.append(c.get("original_verdict"))

    stats["output_claims"] = len(out)
    stats["output_docs"] = sum(len(c["documents"]) for c in out)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {OUTPUT_PATH}")

    # Balanced 버전: True 전부 + False는 original_verdict 기준 stratified sampling
    balanced = _balance(list(zip(out, out_meta)))
    with open(OUTPUT_PATH_BALANCED, "w", encoding="utf-8") as f:
        json.dump(balanced, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {OUTPUT_PATH_BALANCED}")
    stats["balanced_claims"] = len(balanced)
    stats["balanced_docs"] = sum(len(c["documents"]) for c in balanced)

    # 샘플 재생성 (스키마가 바뀌었으므로) — full / balanced 각각
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for items, suffix in [(out, "preprocessed"), (balanced, "preprocessed_balanced")]:
        rng = random.Random(SEED)
        samples = rng.sample(items, min(N_SAMPLES, len(items)))
        sample_path = SAMPLE_DIR / f"sample_{N_SAMPLES}_{suffix}.json"
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"샘플 저장: {sample_path}")

    print("--- 전처리 통계 ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # balanced verdict 분포 확인용
    bal_verdicts = collections.Counter(e["gold_answers"][0] for e in balanced)
    print(f"  balanced verdict: {dict(bal_verdicts)}")

    return out, balanced, stats


if __name__ == "__main__":
    preprocess()
