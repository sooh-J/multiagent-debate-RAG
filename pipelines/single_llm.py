"""
Single LLM 파이프라인

토론 없이 단일 LLM이 모든 문서를 한번에 보고 답변.
"""

from common.llm import call_llm
from common.parsing import parse_answers, parse_explanation
from prompts.single_llm import single_llm_prompt
from prompts.raguard import single_llm_prompt_raguard


def _prompt_for(dataset: str):
    if dataset.startswith("raguard"):
        return single_llm_prompt_raguard
    return single_llm_prompt


def single_llm(query: str, documents: list[str], dataset: str = "ramdocs") -> dict:
    prompt = _prompt_for(dataset)(query, documents)
    output = call_llm(prompt)

    answers = parse_answers(output)
    explanation = parse_explanation(output)

    print(f"\n[Single LLM]\nANSWER: {answers}\nEXPLANATION: {explanation}")

    return {
        "final_answer": answers,
        "final_explanation": explanation,
        "rounds_run": 1,
        "round_history": [{
            "round": 1,
            "llm_output": output,
            "answers": answers,
            "explanation": explanation,
        }],
    }
