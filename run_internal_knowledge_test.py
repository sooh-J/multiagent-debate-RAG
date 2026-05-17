"""
GPT-4o-mini의 RAGuard claim 사전 지식 검증 (ablation)

Single LLM에 문서를 전혀 주지 않고, claim만 던져서 답을 맞히는지 본다.
- ~50% → 모름
- ~75%+ → claim을 internal knowledge로 답함 → 우리 multi-agent vs single 비교의 fairness 의심됨

Usage:
    python run_internal_knowledge_test.py                                  # default: raguard_balanced 전체
    python run_internal_knowledge_test.py --dataset raguard --n 50
"""

import argparse
import json
import os
import re
import sys

from common.logging import Tee
from common.llm import call_llm, print_usage_summary
from common.metrics import compute_metrics
from common.parsing import parse_answers, parse_explanation
from data.raguard.loader import load_raguard

DATASET_LOADERS = {
    "raguard":          lambda n: load_raguard(n_samples=n, balanced=False),
    "raguard_balanced": lambda n: load_raguard(n_samples=n, balanced=True),
}


# question 필드는 "Is the following claim true or false? ... Claim: \"<actual>\"" 형태.
# 평가 단순화를 위해 raw claim 추출해서 doc 없는 prompt 별도 구성.
def _extract_claim(question: str) -> str:
    m = re.search(r'Claim:\s*"(.*)"\s*$', question, re.DOTALL)
    return m.group(1) if m else question


def no_doc_prompt(claim: str) -> str:
    return f"""You are a fact-checker. Decide whether the following claim is true or false based on your knowledge.

The claim has EXACTLY ONE correct answer: either "True" or "False".

Please follow the format: 'All Correct Answers: ["True"]. Explanation: {{}}.'
                     or:  'All Correct Answers: ["False"]. Explanation: {{}}.'

Claim: "{claim}"
"""


def run_one(sample: dict) -> dict:
    claim = _extract_claim(sample["question"])
    output = call_llm(no_doc_prompt(claim))
    predicted = parse_answers(output) or []
    explanation = parse_explanation(output)
    metrics = compute_metrics(predicted, sample["gold_answers"], sample["wrong_answers"])
    return {
        "claim": claim,
        "gold_answers": sample["gold_answers"],
        "wrong_answers": sample["wrong_answers"],
        "predicted": predicted,
        "explanation": explanation,
        **metrics,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASET_LOADERS), default="raguard_balanced")
    p.add_argument("--n", type=int, default=None, help="평가 샘플 개수 (생략 시 전체)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    suffix = "full" if args.n is None else f"n{args.n}"
    tee = Tee(prefix=f"internal_knowledge_{args.dataset}_{suffix}")
    sys.stdout = tee

    try:
        ds = DATASET_LOADERS[args.dataset](args.n)
        print(f"{args.dataset} 로드: {len(ds)}개 (no-doc ablation)")

        results = []
        for i, sample in enumerate(ds):
            out = run_one(sample)
            print(f"[{i+1}/{len(ds)}] Gold={out['gold_answers']}  Pred={out['predicted']}  EM={out['em']}")
            results.append(out)

        n = len(results)
        em = sum(r["em"] for r in results) / n * 100
        true_items = [r for r in results if r["gold_answers"][0] == "True"]
        false_items = [r for r in results if r["gold_answers"][0] == "False"]
        true_acc = (sum(1 for r in true_items if r["predicted"] == ["True"]) / len(true_items)
                    if true_items else 0)
        false_acc = (sum(1 for r in false_items if r["predicted"] == ["False"]) / len(false_items)
                     if false_items else 0)

        print("\n" + "=" * 50)
        print("INTERNAL KNOWLEDGE TEST (no docs)")
        print("=" * 50)
        print(f"  n              : {n}")
        print(f"  EM (overall)   : {em:.2f}%")
        print(f"  True accuracy  : {true_acc*100:.2f}% ({len(true_items)} samples)")
        print(f"  False accuracy : {false_acc*100:.2f}% ({len(false_items)} samples)")
        print(f"  Balanced acc   : {(true_acc + false_acc)/2*100:.2f}%")

        output_path = f"results/internal_knowledge_{args.dataset}_{suffix}_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n저장: {output_path}")

        print_usage_summary()
    finally:
        tee.close()
        print(f"로그: {tee.filepath}")
