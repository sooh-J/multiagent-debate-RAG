"""
MadamRAG 실행 스크립트

Usage:
    conda activate nlp
    python run_madamrag.py
"""

import sys
import json

from common.logging import Tee
from common.data import load_ramdocs
from common.metrics import compute_metrics, print_results_table
from common.llm import print_usage_summary
from pipelines.madamrag import madam_rag


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


def run_on_dataset(ds_sample) -> list[dict]:
    results = []
    for i, sample in enumerate(ds_sample):
        print(f"\n[{i+1}/{len(ds_sample)}] Q: {sample['question']}")
        out = run_on_sample(sample)
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

    output_path = "results/madamrag_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과가 '{output_path}'에 저장되었습니다.")

    return results


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)

    tee = Tee(prefix="madamrag")
    sys.stdout = tee

    try:
        ds_sample = load_ramdocs(n_samples=3)
        all_results = run_on_dataset(ds_sample)

        print_usage_summary()
    finally:
        tee.close()
        print(f"로그 저장: {tee.filepath}")
