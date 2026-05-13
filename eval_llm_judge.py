"""
LLM-as-a-Judge 재채점 스크립트

기존 결과 JSON 파일을 읽어서 LLM으로 의미적 동일성을 판정하고,
새로운 메트릭(llm_em, llm_precision, llm_recall, llm_f1)을 계산한다.

Usage:
    python eval_llm_judge.py results/madamrag_results.json
    python eval_llm_judge.py results/single_llm_results.json
"""

import sys
import json

from common.llm import call_llm_batch, print_usage_summary
from prompts.llm_judge import judge_prompt


def judge_is_match(question: str, gold: str, predicted: str) -> bool:
    """LLM을 사용하여 두 답변이 의미적으로 동일한지 판정"""
    prompt = judge_prompt(question, gold, predicted)
    response = call_llm_batch([prompt])[0].strip().lower()
    return response.startswith("yes")


def judge_batch(question: str, pairs: list[tuple[str, str]]) -> list[bool]:
    """여러 (gold, predicted) 쌍을 한번에 판정"""
    prompts = [judge_prompt(question, g, p) for g, p in pairs]
    responses = call_llm_batch(prompts)
    return [r.strip().lower().startswith("yes") for r in responses]


def evaluate_sample(sample: dict) -> dict:
    question = sample["question"]
    predicted = sample["predicted"]
    gold_answers = sample["gold_answers"]
    wrong_answers = sample["wrong_answers"]

    if not predicted:
        return {
            "llm_em": 0, "llm_precision": 0.0, "llm_recall": 0.0, "llm_f1": 0.0,
            "llm_n_gold_hit": 0, "llm_n_misinfo": 0, "llm_n_unknown": 0,
        }

    # predicted × gold 매칭
    gold_pairs = [(g, p) for p in predicted for g in gold_answers]
    gold_results = judge_batch(question, gold_pairs) if gold_pairs else []

    # predicted × wrong 매칭
    wrong_pairs = [(w, p) for p in predicted for w in wrong_answers]
    wrong_results = judge_batch(question, wrong_pairs) if wrong_pairs else []

    # 각 predicted가 gold/wrong/unknown 중 어디에 해당하는지
    n_gold = len(gold_answers)
    n_wrong = len(wrong_answers)

    tp = 0
    matched_gold_set = set()
    misinfo_count = 0
    unknown_count = 0

    for i, p in enumerate(predicted):
        # gold 매칭 확인
        gold_match = any(gold_results[i * n_gold + j] for j in range(n_gold)) if n_gold > 0 else False
        # wrong 매칭 확인
        wrong_match = any(wrong_results[i * n_wrong + j] for j in range(n_wrong)) if n_wrong > 0 else False

        if gold_match:
            tp += 1
            for j in range(n_gold):
                if gold_results[i * n_gold + j]:
                    matched_gold_set.add(j)
        elif wrong_match:
            misinfo_count += 1
        else:
            unknown_count += 1

    precision = tp / len(predicted) if predicted else 0.0
    recall = len(matched_gold_set) / n_gold if n_gold > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Strict EM: 모든 gold 매칭 + misinfo 없음 + extra 없음
    complete = len(matched_gold_set) == n_gold
    no_misinfo = misinfo_count == 0
    no_extra = len(predicted) == n_gold
    em = int(complete and no_misinfo and no_extra)

    return {
        "llm_em": em,
        "llm_precision": round(precision, 4),
        "llm_recall": round(recall, 4),
        "llm_f1": round(f1, 4),
        "llm_n_gold_hit": len(matched_gold_set),
        "llm_n_misinfo": misinfo_count,
        "llm_n_unknown": unknown_count,
    }


def main(result_path: str):
    with open(result_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"Evaluating {result_path} ({len(results)} samples)")
    print("=" * 60)

    for i, sample in enumerate(results):
        llm_metrics = evaluate_sample(sample)
        sample.update(llm_metrics)

        print(f"[{i+1}/{len(results)}] Q: {sample['question'][:50]}")
        print(f"  Gold: {sample['gold_answers']}  Predicted: {sample['predicted']}")
        print(f"  EM: {sample['em']} → LLM_EM: {llm_metrics['llm_em']}  "
              f"F1: {sample['f1']} → LLM_F1: {llm_metrics['llm_f1']}  "
              f"#Gold: {llm_metrics['llm_n_gold_hit']}/{len(sample['gold_answers'])}  "
              f"#Misinfo: {llm_metrics['llm_n_misinfo']}  #Unknown: {llm_metrics['llm_n_unknown']}")

    # Results table (기존 양식과 동일)
    n = len(results)
    print("\n" + "=" * 90)
    print("LLM JUDGE RESULTS")
    print("=" * 90)
    header = f"{'#':<4} {'Question':<36} {'Predicted':<18} {'Gold':<18} {'EM':>3} {'Prec':>5} {'Rec':>5} {'F1':>5} {'#Gold':>8} {'#Misinfo':>8} {'#Unknown':>8}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results):
        q = r['question'][:34] + ".." if len(r['question']) > 34 else r['question']
        pred = str(r['predicted'])[:16] + ".." if len(str(r['predicted'])) > 16 else str(r['predicted'])
        gold = str(r['gold_answers'])[:16] + ".." if len(str(r['gold_answers'])) > 16 else str(r['gold_answers'])
        em = "O" if r['llm_em'] else "X"
        n_gold = len(r['gold_answers'])
        gold_hit = f"{r['llm_n_gold_hit']}/{n_gold}"

        print(f"{i:<4} {q:<36} {pred:<18} {gold:<18} {em:>3} {r['llm_precision']:>5.2f} {r['llm_recall']:>5.2f} {r['llm_f1']:>5.2f} {gold_hit:>8} {r['llm_n_misinfo']:>8} {r['llm_n_unknown']:>8}")

    print("=" * len(header))
    print(f"{'AVERAGE':<76} {sum(r['llm_em'] for r in results)/n*100:>3.0f}% {sum(r['llm_precision'] for r in results)/n:>5.2f} {sum(r['llm_recall'] for r in results)/n:>5.2f} {sum(r['llm_f1'] for r in results)/n:>5.2f}")
    print("=" * len(header))

    noisy = [r for r in results if r['wrong_answers']]
    clean = [r for r in results if not r['wrong_answers']]

    if noisy:
        print(f"\n  노이즈 있는 샘플 ({len(noisy)}개) LLM_EM: {sum(r['llm_em'] for r in noisy)/len(noisy)*100:.1f}%  misinfo 오염률: {sum(1 for r in noisy if r['llm_n_misinfo'] > 0)/len(noisy)*100:.1f}%")
    if clean:
        print(f"  노이즈 없는 샘플 ({len(clean)}개) LLM_EM: {sum(r['llm_em'] for r in clean)/len(clean)*100:.1f}%")

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON: Original vs LLM Judge")
    print("=" * 60)
    print(f"{'Metric':<12} {'Original':>10} {'LLM Judge':>10}")
    print("-" * 34)
    print(f"{'EM':<12} {sum(r['em'] for r in results)/n*100:>9.1f}% {sum(r['llm_em'] for r in results)/n*100:>9.1f}%")
    print(f"{'Precision':<12} {sum(r['precision'] for r in results)/n:>10.3f} {sum(r['llm_precision'] for r in results)/n:>10.3f}")
    print(f"{'Recall':<12} {sum(r['recall'] for r in results)/n:>10.3f} {sum(r['llm_recall'] for r in results)/n:>10.3f}")
    print(f"{'F1':<12} {sum(r['f1'] for r in results)/n:>10.3f} {sum(r['llm_f1'] for r in results)/n:>10.3f}")
    print("=" * 60)

    # 저장
    output_path = result_path.replace(".json", "_llm_judge.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {output_path}")

    print_usage_summary()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_llm_judge.py <result_json_path>")
        sys.exit(1)
    main(sys.argv[1])
