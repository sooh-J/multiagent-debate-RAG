"""
MadamRAG 실행 스크립트 (RAMDocs 전체 500개)

  - 입력: data/ramdocs/full.json (RAMDocs test split 500개)
  - 출력: results/madamrag_results.json
  - 로그: logs/madamrag_YYYYMMDD_HHMM.log

기존 결과 파일이 있으면 그 다음 샘플부터 이어서 실행 (resume 내장).
50개마다 중간 저장.

Usage:
    conda activate nlp
    python run_madamrag.py
"""

import os
import sys
import json

from common.logging import Tee
from common.metrics import compute_metrics, print_results_table
from common.llm import print_usage_summary
from pipelines.madamrag import madam_rag


DATA_PATH = "data/ramdocs/full.json"
OUTPUT_PATH = "results/madamrag_results.json"
CHECKPOINT_EVERY = 50


def run_on_sample(sample: dict) -> dict:
    query = sample["question"]
    doc_texts = [doc["text"] for doc in sample["documents"]]
    doc_meta = [{"type": doc["type"], "answer": doc["answer"]} for doc in sample["documents"]]

    result = madam_rag(query, doc_texts)

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


def run_on_dataset(ds_sample, existing_results, output_path) -> list[dict]:
    results = list(existing_results)
    start = len(results)
    total = len(ds_sample)

    if start > 0:
        print(f"이어서 실행: {start}개 완료, {total - start}개 남음")
    else:
        print(f"처음부터 실행: 총 {total}개")

    for i in range(start, total):
        sample = ds_sample[i]
        print(f"\n[{i+1}/{total}] Q: {sample['question']}")
        out = run_on_sample(sample)
        print(f"  Gold:      {out['gold_answers']}")
        print(f"  Predicted: {out['predicted']}")
        print(f"  EM={out['em']}  P={out['precision']}  R={out['recall']}  F1={out['f1']}")
        results.append(out)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [중간 저장] {i+1}개 결과 → {output_path}")

    n = len(results)
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"  Exact Match : {sum(r['em'] for r in results) / n * 100:.2f}%")
    print(f"  Precision   : {sum(r['precision'] for r in results) / n:.4f}")
    print(f"  Recall      : {sum(r['recall'] for r in results) / n:.4f}")
    print(f"  F1          : {sum(r['f1'] for r in results) / n:.4f}")

    print_results_table(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과가 '{output_path}'에 저장되었습니다.")

    return results


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    tee = Tee(prefix="madamrag")
    sys.stdout = tee

    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            ds_sample = json.load(f)
        print(f"RAMDocs 전체 데이터 로드: {len(ds_sample)}개")

        if os.path.exists(OUTPUT_PATH):
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
            print(f"기존 결과 로드: {len(existing)}개")
        else:
            existing = []

        if len(existing) >= len(ds_sample):
            print("이미 전체 완료 상태입니다. 종료.")
        else:
            run_on_dataset(ds_sample, existing, OUTPUT_PATH)
            print_usage_summary()
    finally:
        tee.close()
        print(f"로그 저장: {tee.filepath}")
