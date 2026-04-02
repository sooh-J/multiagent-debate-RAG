from common.parsing import normalize_answer


def is_match(a: str, b: str) -> bool:
    return a in b or b in a


def strict_exact_match(pred_norm: list[str], gold_norm: list[str], wrong_norm: list[str]) -> int:
    """
    Strict Exact Match (SEM)
    예측된 답변이 정확히 골드 답변 세트와 일치하는지 판정.
    1. Completeness: predicted_set이 gold_set의 모든 요소를 포함하는가?
    2. No Misinformation: predicted_set과 wrong_set 사이에 공통 요소가 없는가?
    3. No Extra: predicted_set 개수가 gold_set 개수와 동일한가?
    """
    pred_set = set(pred_norm)
    gold_set = set(gold_norm)
    wrong_set = set(wrong_norm)

    # 1. Completeness: 모든 gold가 pred에 매칭
    complete = all(any(is_match(g, p) for p in pred_set) for g in gold_set)
    # 2. No Misinformation: wrong 답변이 pred에 없어야 함
    no_misinfo = not any(is_match(p, w) for p in pred_set for w in wrong_set)
    # 3. No Extra: 개수 동일
    no_extra = len(pred_set) == len(gold_set)

    # 개별 pred가 gold/wrong/unknown 중 어디에 해당하는지
    matched_gold = [p for p in pred_set if any(is_match(p, g) for g in gold_set)]
    matched_wrong = [p for p in pred_set if any(is_match(p, w) for w in wrong_set)]
    matched_unknown = [p for p in pred_set if p not in matched_gold and p not in matched_wrong]

    return int(complete and no_misinfo and no_extra), matched_gold, matched_wrong, matched_unknown


def compute_metrics(predicted_answers, gold_answers, wrong_answers):
    pred_norm = [normalize_answer(a) for a in predicted_answers]
    gold_norm = [normalize_answer(a) for a in gold_answers]
    wrong_norm = [normalize_answer(a) for a in wrong_answers]

    tp = sum(1 for p in pred_norm if any(is_match(p, g) for g in gold_norm))
    has_wrong = any(is_match(p, w) for p in pred_norm for w in wrong_norm)

    em, _, _, _ = strict_exact_match(pred_norm, gold_norm, wrong_norm)

    precision = tp / len(pred_norm) if pred_norm else 0.0
    recall = tp / len(gold_norm) if gold_norm else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    wrong_in_pred = [p for p in pred_norm if any(is_match(p, w) for w in wrong_norm)]

    return {
        "em": int(em),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "predicted_answers": predicted_answers,
        "wrong_in_pred": wrong_in_pred,
    }


def print_results_table(results: list[dict]):
    print("\n" + "=" * 90)
    print("EXPERIMENT RESULTS")
    print("=" * 90)
    print(f"{'#':<4} {'Question':<40} {'Predicted':<20} {'Gold':<20} {'EM':>3} {'P':>6} {'R':>6} {'F1':>6} {'Wrong':>6}")
    print("-" * 90)

    for i, r in enumerate(results):
        q = r['question'][:38] + ".." if len(r['question']) > 38 else r['question']
        pred = str(r['predicted'])[:18] + ".." if len(str(r['predicted'])) > 18 else str(r['predicted'])
        gold = str(r['gold_answers'])[:18] + ".." if len(str(r['gold_answers'])) > 18 else str(r['gold_answers'])
        em = "\u2713" if r['em'] else "\u2717"
        wrong_flag = "\u26a0" if r['wrong_in_pred'] else "-"

        print(f"{i:<4} {q:<40} {pred:<20} {gold:<20} {em:>3} {r['precision']:>6.2f} {r['recall']:>6.2f} {r['f1']:>6.2f} {wrong_flag:>6}")

    print("=" * 90)
    n = len(results)
    print(f"{'AVERAGE':<65} {sum(r['em'] for r in results)/n*100:>3.0f}% {sum(r['precision'] for r in results)/n:>6.2f} {sum(r['recall'] for r in results)/n:>6.2f} {sum(r['f1'] for r in results)/n:>6.2f}")
    print("=" * 90)

    noisy = [r for r in results if r['wrong_answers']]
    clean = [r for r in results if not r['wrong_answers']]

    if noisy:
        print(f"\n  노이즈 있는 샘플 ({len(noisy)}개) EM: {sum(r['em'] for r in noisy)/len(noisy)*100:.1f}%  wrong 오염률: {sum(1 for r in noisy if r['wrong_in_pred'])/len(noisy)*100:.1f}%")
    if clean:
        print(f"  노이즈 없는 샘플 ({len(clean)}개) EM: {sum(r['em'] for r in clean)/len(clean)*100:.1f}%")
