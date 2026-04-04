"""
V4 실행 스크립트

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

V4 방법론: 찬/반/중재자 (Round 1만) → 이후 MadamRAG 방식
  - Round 1: 문서별 Pro(찬성), Con(반대), Mediator(중재자)로 신뢰도 검증
  - Round 2+: 원본 MadamRAG 방식으로 전환 (문서 대변인 1명 + 타 에이전트 응답 참고)
  - Round 1 중재자 결과가 Round 2의 "이전 라운드 에이전트 답변"으로 자연스럽게 이어짐
  - 수렴 시 조기 종료 (최대 3라운드)

V3와의 차이:
  V3는 매 라운드 찬/반/중재자를 반복하지만,
  V4는 첫 라운드에서만 찬/반/중재자로 필터링한 뒤 MadamRAG 토론으로 넘어간다.
  → V3 대비 LLM 호출 수가 적고, 라운드 구조 변경의 효과를 ablation으로 비교 가능

Async:
  - 문서 간 / 에이전트 간 API 호출을 병렬로 실행하여 속도 향상

실행 방법:
    conda activate nlp
    python run_v4.py

출력:
  - 콘솔: 각 샘플별 예측 결과 및 메트릭 (EM, Precision, Recall, F1)
  - 로그: logs/v4_YYYYMMDD_HHMM.log
  - 결과: results/v4_results.json

파이프라인 코드: pipelines/v4.py
프롬프트 정의:  prompts/v3.py (V3와 공유), prompts/madamrag.py (Round 2+)
설정:          configs/v3.py (V3와 공유)
"""

import sys
import json
import asyncio

from common.logging import Tee
from common.metrics import compute_metrics, print_results_table
from common.llm import print_usage_summary
from pipelines.v4 import v4_method


async def run_on_sample(sample: dict) -> dict:
    query = sample["question"]
    doc_texts = [doc["text"] for doc in sample["documents"]]
    doc_meta = [{"type": doc["type"], "answer": doc["answer"]} for doc in sample["documents"]]

    result = await v4_method(query, doc_texts)

    predicted_answers = result["final_answer"] if result["final_answer"] else []
    metrics = compute_metrics(predicted_answers, sample["gold_answers"], sample["wrong_answers"])

    return {
        "question": query,
        "disambig_entity": sample["disambig_entity"],
        "gold_answers": sample["gold_answers"],
        "wrong_answers": sample["wrong_answers"],
        "doc_meta": doc_meta,
        "predicted": predicted_answers,
        "explanation": result["final_explanation"],
        "rounds_run": result["rounds_run"],
        "round_history": result["round_history"],
        **metrics,
    }


async def run_on_dataset(ds_sample) -> list[dict]:
    results = []
    for i, sample in enumerate(ds_sample):
        print(f"\n[{i+1}/{len(ds_sample)}] Q: {sample['question']}")
        out = await run_on_sample(sample)
        print(f"  Gold:      {out['gold_answers']}")
        print(f"  Predicted: {out['predicted']}")
        print(f"  EM={out['em']}  P={out['precision']}  R={out['recall']}  F1={out['f1']}")
        results.append(out)

    n = len(results)
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"  Exact Match : {sum(r['em'] for r in results) / n * 100:.2f}%")
    print(f"  Precision   : {sum(r['precision'] for r in results) / n:.4f}")
    print(f"  Recall      : {sum(r['recall'] for r in results) / n:.4f}")
    print(f"  F1          : {sum(r['f1'] for r in results) / n:.4f}")

    print_results_table(results)

    output_path = "results/v4_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과가 '{output_path}'에 저장되었습니다.")

    return results


async def main():
    import os
    os.makedirs("results", exist_ok=True)

    tee = Tee(prefix="v4")
    sys.stdout = tee

    try:
        with open("data/ramdocs/sample/sample_100.json", "r", encoding="utf-8") as f:
            ds_sample = json.load(f)
        all_results = await run_on_dataset(ds_sample)

        print_usage_summary()
    finally:
        tee.close()
        print(f"로그 저장: {tee.filepath}")


if __name__ == "__main__":
    asyncio.run(main())
