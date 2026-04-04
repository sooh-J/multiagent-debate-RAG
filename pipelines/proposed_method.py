"""
Proposed Method 파이프라인

2단계 구조:
  1단계 — 문서 내부 토론 (Extractor → Skeptic → Resolver) per document
  2단계 — 문서 간 집계 (Global Aggregator)
"""

from common.llm import call_llm
from common.parsing import parse_answers, parse_explanation
from prompts.proposed_method import (
    extractor_prompt,
    skeptic_prompt,
    resolver_prompt,
    global_aggregator_prompt,
)


def local_debate(query: str, document: str, doc_index: int) -> dict:
    """1단계: 단일 문서에 대한 내부 토론 (Extractor → Skeptic → Resolver)"""

    print(f"\n  [Doc {doc_index+1} - Extractor]")
    ext_response = call_llm(extractor_prompt(query, document))
    print(f"  {ext_response}")

    print(f"\n  [Doc {doc_index+1} - Skeptic]")
    skp_response = call_llm(skeptic_prompt(query, document, ext_response))
    print(f"  {skp_response}")

    print(f"\n  [Doc {doc_index+1} - Resolver]")
    res_response = call_llm(resolver_prompt(query, document, ext_response, skp_response))
    print(f"  {res_response}")

    return {
        "doc_index": doc_index,
        "extractor": ext_response,
        "skeptic": skp_response,
        "resolver": res_response,
    }


def proposed_method(query: str, documents: list[str]) -> dict:
    """
    Args:
        query:     사용자 질문
        documents: 검색된 문서 리스트

    Returns:
        {
          "final_answer":      list[str],
          "final_explanation": str,
          "rounds_run":        1,
          "round_history":     list
        }
    """

    # ── 1단계: 문서별 내부 토론 ───────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Stage 1: Per-Document Local Debate")
    print('='*50)

    local_results = []
    document_summaries = []

    for i, doc in enumerate(documents):
        print(f"\n{'─'*40}")
        print(f"Document {i+1}")
        print('─'*40)

        result = local_debate(query, doc, i)
        local_results.append(result)
        document_summaries.append(result["resolver"])

    # ── 2단계: 문서 간 집계 ───────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Stage 2: Global Aggregation")
    print('='*50)

    agg_prompt = global_aggregator_prompt(query, document_summaries)
    agg_output = call_llm(agg_prompt)

    final_answer = parse_answers(agg_output)
    final_explanation = parse_explanation(agg_output)

    print(f"\n[Global Aggregator]\nANSWER: {final_answer}\nEXPLANATION: {final_explanation}")

    return {
        "final_answer": final_answer,
        "final_explanation": final_explanation,
        "rounds_run": 1,
        "round_history": [{
            "local_debates": local_results,
            "document_summaries": document_summaries,
            "aggregator_answer": final_answer,
            "aggregator_explanation": final_explanation,
        }],
    }
