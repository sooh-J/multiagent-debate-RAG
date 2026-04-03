"""
V3 파이프라인: 찬/반/중재자 + 매 라운드 반복

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

개요:
  MadamRAG의 멀티라운드 토론 구조를 유지하되,
  각 문서에 대해 1명의 문서 대변인 대신 3명의 역할 에이전트(찬성/반대/중재자)를 배치한다.

  핵심 아이디어:
    "이 문서가 질문에 대한 답을 정말 포함하는가?"에 대해
    찬성과 반대가 각각 근거를 제시하고, 중재자가 양쪽 주장을 보고 판정한다.
    이를 매 라운드 반복하면서 답변의 신뢰도를 높인다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

흐름 (문서 N개, 최대 T라운드):

  Round 1:
    문서 1: Pro → Con → Mediator  ─┐
    문서 2: Pro → Con → Mediator   ├→ Aggregator → 요약 (confidence 포함)
    문서 N: Pro → Con → Mediator  ─┘

  Round 2~T:
    문서 1: Pro(+이전요약) → Con(+이전요약) → Mediator(+이전요약)  ─┐
    문서 2: Pro(+이전요약) → Con(+이전요약) → Mediator(+이전요약)   ├→ Aggregator → 요약
    문서 N: Pro(+이전요약) → Con(+이전요약) → Mediator(+이전요약)  ─┘

  조기 종료 조건:
    - Round 2 이후, 모든 문서의 중재자 답변이 이전 라운드와 동일하면 종료

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MadamRAG 대비 차이점:
  - 에이전트 수: 문서당 1명 → 3명 (Pro/Con/Mediator)
  - 에이전트 역할: 문서 대변인 → 찬성/반대/중재자 (stance-based)
  - Aggregator 입력: 에이전트 답변 전체 → 중재자 판정 결과만
  - LLM 호출 수: 라운드당 N+1 → 3N+1 (약 3배)

Proposed Method 대비 차이점:
  - 라운드 수: 1회 고정 → 최대 T라운드 반복 (수렴 시 조기 종료)
  - 에이전트 역할: Extractor/Skeptic/Resolver → Pro/Con/Mediator
  - 이전 라운드 정보: 없음 → Aggregator 요약이 다음 라운드에 전달됨
"""

from common.llm import call_llm
from common.parsing import normalize_answer, extract_answer, parse_answers, parse_explanation
from prompts.v3 import (
    pro_prompt,
    con_prompt,
    mediator_prompt,
    pro_debate_prompt,
    con_debate_prompt,
    mediator_debate_prompt,
    aggregator_with_confidence_prompt,
)
from configs.v3 import MAX_ROUNDS


def doc_debate(query: str, document: str, doc_index: int,
               round_num: int, prev_summary: str) -> dict:
    """단일 문서에 대한 찬/반/중재자 토론 1회"""

    # ── Pro (찬성) ──
    print(f"\n  [Doc {doc_index+1} - Pro]")
    if round_num == 1:
        pro_resp = call_llm(pro_prompt(query, document))
    else:
        pro_resp = call_llm(pro_debate_prompt(query, document, prev_summary))
    print(f"  {pro_resp}")

    # ── Con (반대) ──
    print(f"\n  [Doc {doc_index+1} - Con]")
    if round_num == 1:
        con_resp = call_llm(con_prompt(query, document))
    else:
        con_resp = call_llm(con_debate_prompt(query, document, prev_summary))
    print(f"  {con_resp}")

    # ── Mediator (중재자) ──
    print(f"\n  [Doc {doc_index+1} - Mediator]")
    if round_num == 1:
        med_resp = call_llm(mediator_prompt(query, document, pro_resp, con_resp))
    else:
        med_resp = call_llm(mediator_debate_prompt(
            query, document, pro_resp, con_resp, prev_summary
        ))
    print(f"  {med_resp}")

    return {
        "doc_index": doc_index,
        "pro": pro_resp,
        "con": con_resp,
        "mediator": med_resp,
    }


def v3_method(query: str, documents: list[str]) -> dict:
    """
    V3 메인 파이프라인

    Args:
        query:     사용자 질문
        documents: 검색된 문서 리스트

    Returns:
        {
          "final_answer":      list[str],
          "final_explanation": str,
          "rounds_run":        int,
          "round_history":     list[dict]
        }
    """
    prev_mediator_outputs = [""] * len(documents)
    prev_summary = ""
    prev_answer: list[str] = []
    prev_explanation = ""
    round_history = []

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{'='*50}")
        print(f"Round {round_num}")
        print('='*50)

        # ── Step 1: 문서별 찬/반/중재자 토론 ──
        debate_results = []
        mediator_outputs = []

        for i, doc in enumerate(documents):
            print(f"\n{'─'*40}")
            print(f"Document {i+1}")
            print('─'*40)

            result = doc_debate(query, doc, i, round_num, prev_summary)
            debate_results.append(result)
            mediator_outputs.append(result["mediator"])

        # ── Step 2: Early stopping (Round 2+) ──
        if round_num > 1:
            curr_normalized = [normalize_answer(extract_answer(m)) for m in mediator_outputs]
            prev_normalized = [normalize_answer(extract_answer(m)) for m in prev_mediator_outputs]

            converged = True
            for c, p in zip(curr_normalized, prev_normalized):
                if c not in p and p not in c:
                    converged = False
                    break

            if converged:
                print(f"\n>> Early stopping at round {round_num} (all mediators converged)")
                round_history.append({
                    "round": round_num,
                    "debates": debate_results,
                    "mediator_outputs": mediator_outputs,
                    "aggregator_answer": prev_answer,
                    "aggregator_explanation": prev_explanation,
                    "early_stopped": True,
                })
                return {
                    "final_answer": prev_answer,
                    "final_explanation": prev_explanation,
                    "rounds_run": round_num,
                    "round_history": round_history,
                }

        # ── Step 3: Aggregator ──
        print(f"\n{'─'*40}")
        print("Aggregator")
        print('─'*40)

        agg_output = call_llm(aggregator_with_confidence_prompt(query, mediator_outputs))
        agg_answer = parse_answers(agg_output)
        agg_explanation = parse_explanation(agg_output)

        print(f"\n[Aggregator]\nANSWER: {agg_answer}\nEXPLANATION: {agg_explanation}")

        round_history.append({
            "round": round_num,
            "debates": debate_results,
            "mediator_outputs": mediator_outputs,
            "aggregator_answer": agg_answer,
            "aggregator_explanation": agg_explanation,
            "early_stopped": False,
        })

        prev_mediator_outputs = mediator_outputs
        prev_summary = agg_output
        prev_answer = agg_answer
        prev_explanation = agg_explanation

    print(f"\n>> Reached max rounds (T={MAX_ROUNDS})")
    return {
        "final_answer": prev_answer,
        "final_explanation": prev_explanation,
        "rounds_run": MAX_ROUNDS,
        "round_history": round_history,
    }
