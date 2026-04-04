"""
V4 파이프라인: 찬/반/중재자 (Round 1만) → 이후 MadamRAG 방식

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

개요:
  Round 1에서만 찬/반/중재자 구조로 각 문서의 신뢰도를 검증한 뒤,
  Round 2부터는 원본 MadamRAG 방식(문서 대변인 1명 + 타 에이전트 응답 참고)으로 전환한다.

  핵심 아이디어:
    첫 라운드의 찬/반/중재자 토론으로 각 문서의 답변 품질을 한 번 걸러낸 뒤,
    그 결과를 기반으로 MadamRAG의 에이전트 간 토론을 진행한다.
    → Round 1의 중재자 결과가 Round 2의 "이전 라운드 에이전트 답변"으로 자연스럽게 이어진다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

흐름 (문서 N개, 최대 T라운드):

  Round 1 — 찬/반/중재자 (V3와 동일):
    문서 1: Pro → Con → Mediator  ─┐
    문서 2: Pro → Con → Mediator   ├→ Aggregator → 요약
    문서 N: Pro → Con → Mediator  ─┘

  Round 2~T — MadamRAG 방식:
    에이전트 1 (문서1 + 타 에이전트 이전 답변 참고) ─┐
    에이전트 2 (문서2 + 타 에이전트 이전 답변 참고)  ├→ Aggregator → 요약
    에이전트 N (문서N + 타 에이전트 이전 답변 참고) ─┘
    (여기서 "타 에이전트 이전 답변"은 첫 라운드에서는 중재자 결과)

  조기 종료 조건:
    - Round 2 이후, 모든 에이전트 답변이 이전 라운드와 동일하면 종료
    - MadamRAG의 early stopping 로직과 동일

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

V3 대비 차이점:
  - Round 2+에서 에이전트 구조가 다름: 찬/반/중재자 → 문서 대변인 1명
  - Round 2+의 프롬프트: V3 프롬프트 → MadamRAG 프롬프트 재사용
  - LLM 호출 수: V3보다 적음 (Round 2+에서 문서당 1호출 vs 3호출)

MadamRAG 대비 차이점:
  - Round 1만 찬/반/중재자로 대체 → 초기 답변의 신뢰도 검증이 더 강력
  - Round 2+는 MadamRAG와 동일 (프롬프트, early stopping 로직 모두 재사용)

Async:
  - Round 1: 문서 간 병렬 + 문서 내 Pro/Con 병렬 (Mediator는 둘 다 끝난 뒤)
  - Round 2+: 같은 라운드 내 에이전트 N개 병렬
"""

import asyncio

from common.llm import async_call_llm
from common.parsing import normalize_answer, extract_answer, parse_answers, parse_explanation
from prompts.v3 import (
    pro_prompt,
    con_prompt,
    mediator_prompt,
    aggregator_with_confidence_prompt,
)
from prompts.madamrag import agent_debate_prompt, aggregator_prompt
from configs.v3 import MAX_ROUNDS


async def doc_debate_round1(query: str, document: str, doc_index: int) -> dict:
    """Round 1: 단일 문서에 대한 Pro/Con 병렬 → Mediator (로그는 반환 후 출력)"""

    pro_resp, con_resp = await asyncio.gather(
        async_call_llm(pro_prompt(query, document)),
        async_call_llm(con_prompt(query, document)),
    )
    med_resp = await async_call_llm(mediator_prompt(query, document, pro_resp, con_resp))

    return {
        "doc_index": doc_index,
        "pro": pro_resp,
        "con": con_resp,
        "mediator": med_resp,
    }


async def v4_method(query: str, documents: list[str]) -> dict:
    """
    V4 메인 파이프라인 (async)

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
    n_docs = len(documents)
    round_history = []

    # ══════════════════════════════════════════════════════════════════════
    # Round 1: 찬/반/중재자 구조 — 문서 간 병렬
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*50}")
    print("Round 1 (Pro/Con/Mediator) — async")
    print('='*50)

    debate_results = await asyncio.gather(*[
        doc_debate_round1(query, doc, i)
        for i, doc in enumerate(documents)
    ])
    debate_results = list(debate_results)
    mediator_outputs = [r["mediator"] for r in debate_results]

    # 로그 순서대로 출력
    for r in debate_results:
        idx = r["doc_index"]
        print(f"\n{'─'*40}")
        print(f"Document {idx+1}")
        print('─'*40)
        print(f"\n  [Doc {idx+1} - Pro] {r['pro']}")
        print(f"\n  [Doc {idx+1} - Con] {r['con']}")
        print(f"\n  [Doc {idx+1} - Mediator] {r['mediator']}")

    # Round 1 Aggregator
    print(f"\n{'─'*40}")
    print("Aggregator (Round 1)")
    print('─'*40)

    agg_output = await async_call_llm(aggregator_with_confidence_prompt(query, mediator_outputs))
    agg_answer = parse_answers(agg_output)
    agg_explanation = parse_explanation(agg_output)

    print(f"\n[Aggregator]\nANSWER: {agg_answer}\nEXPLANATION: {agg_explanation}")

    round_history.append({
        "round": 1,
        "type": "pro_con_mediator",
        "debates": debate_results,
        "mediator_outputs": mediator_outputs,
        "aggregator_answer": agg_answer,
        "aggregator_explanation": agg_explanation,
        "early_stopped": False,
    })

    prev_agent_outputs = mediator_outputs
    prev_answer = agg_answer
    prev_explanation = agg_explanation

    # ══════════════════════════════════════════════════════════════════════
    # Round 2+: MadamRAG 방식 — 에이전트 간 병렬
    # ══════════════════════════════════════════════════════════════════════
    for round_num in range(2, MAX_ROUNDS + 1):
        print(f"\n{'='*50}")
        print(f"Round {round_num} (MadamRAG-style) — async")
        print('='*50)

        # Step 1: 각 에이전트 병렬 호출
        async def _agent_call(i: int, doc: str) -> str:
            history = "\n".join([
                f"Agent {j+1}: {prev_agent_outputs[j]}"
                for j in range(n_docs) if j != i
            ])
            return await async_call_llm(agent_debate_prompt(query, doc, history))

        current_answers = await asyncio.gather(*[
            _agent_call(i, doc) for i, doc in enumerate(documents)
        ])
        current_answers = list(current_answers)

        # 로그 순서대로 출력
        for i, answer in enumerate(current_answers):
            print(f"\n[Agent {i+1}]\n{answer}")

        # Step 2: Early stopping
        curr_normalized = [normalize_answer(extract_answer(a)) for a in current_answers]
        prev_normalized = [normalize_answer(extract_answer(a)) for a in prev_agent_outputs]

        converged = True
        for c, p in zip(curr_normalized, prev_normalized):
            if c not in p and p not in c:
                converged = False
                break

        if converged:
            print(f"\n>> Early stopping at round {round_num} (all agents converged)")
            round_history.append({
                "round": round_num,
                "type": "madam_debate",
                "agent_responses": current_answers,
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

        # Step 3: Aggregator
        agg_output = await async_call_llm(aggregator_prompt(query, current_answers))
        agg_answer = parse_answers(agg_output)
        agg_explanation = parse_explanation(agg_output)

        print(f"\n[Aggregator]\nANSWER: {agg_answer}\nEXPLANATION: {agg_explanation}")

        round_history.append({
            "round": round_num,
            "type": "madam_debate",
            "agent_responses": current_answers,
            "aggregator_answer": agg_answer,
            "aggregator_explanation": agg_explanation,
            "early_stopped": False,
        })

        prev_agent_outputs = current_answers
        prev_answer = agg_answer
        prev_explanation = agg_explanation

    print(f"\n>> Reached max rounds (T={MAX_ROUNDS})")
    return {
        "final_answer": prev_answer,
        "final_explanation": prev_explanation,
        "rounds_run": MAX_ROUNDS,
        "round_history": round_history,
    }
