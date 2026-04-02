"""
Proposed Method 파이프라인

MadamRAG 파이프라인을 기반으로 수정.
"""

from common.llm import call_llm
from common.parsing import normalize_answer, extract_answer, parse_answers, parse_explanation
from prompts.proposed_method import agent_initial_prompt, agent_debate_prompt, aggregator_prompt
from configs.proposed_method import MAX_ROUNDS


def proposed_method(query: str, documents: list[str]) -> dict:
    n_agents = len(documents)
    prev_agent_outputs = [""] * n_agents
    prev_summary: list[str] = []
    prev_explanation = ""
    round_history = []

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{'='*50}")
        print(f"Round {round_num}")
        print('='*50)

        # Step 1: 각 에이전트가 답변 생성
        current_answers = []
        for i, doc in enumerate(documents):
            if round_num == 1:
                prompt = agent_initial_prompt(query, doc)
            else:
                history = "\n".join([
                    f"Agent {j+1}: {prev_agent_outputs[j]}"
                    for j in range(n_agents) if j != i
                ])
                prompt = agent_debate_prompt(query, doc, history)

            answer = call_llm(prompt)
            current_answers.append(answer)
            print(f"\n[Agent {i+1}]\n{answer}")

        # Step 2: Early stopping 체크
        if round_num > 1:
            pred_normalized = [normalize_answer(extract_answer(a)) for a in current_answers]
            prev_normalized = [normalize_answer(extract_answer(a)) for a in prev_agent_outputs]

            flag = True
            for p, q in zip(pred_normalized, prev_normalized):
                if p not in q and q not in p:
                    flag = False
                    break

            if flag:
                print(f"\n>> Early stopping at round {round_num} (all agents converged)")
                round_history.append({
                    "round": round_num,
                    "agent_responses": current_answers,
                    "aggregator_answer": prev_summary,
                    "aggregator_explanation": prev_explanation,
                    "early_stopped": True,
                })
                return {
                    "final_answer": prev_summary,
                    "final_explanation": prev_explanation,
                    "rounds_run": round_num,
                    "round_history": round_history,
                }

        # Step 3: Aggregator 요약
        agg_prompt = aggregator_prompt(query, current_answers)
        agg_output = call_llm(agg_prompt)

        agg_answer = parse_answers(agg_output)
        agg_explanation = parse_explanation(agg_output)

        print(f"\n[Aggregator]\nANSWER: {agg_answer}\nEXPLANATION: {agg_explanation}")

        round_history.append({
            "round": round_num,
            "agent_responses": current_answers,
            "aggregator_answer": agg_answer,
            "aggregator_explanation": agg_explanation,
            "early_stopped": False,
        })

        prev_agent_outputs = current_answers
        prev_summary = agg_answer
        prev_explanation = agg_explanation

    print(f"\n>> Reached max rounds (T={MAX_ROUNDS})")
    return {
        "final_answer": prev_summary,
        "final_explanation": prev_explanation,
        "rounds_run": MAX_ROUNDS,
        "round_history": round_history,
    }
