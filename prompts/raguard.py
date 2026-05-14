"""
RAGuard(claim fact-checking) 전용 프롬프트

RAGuard 데이터셋은 정답이 단일 "True" 또는 "False" 두 값 중 하나다.
RAMDocs는 동명이인/동일명 모호성으로 정답이 여러 개 가능한 반면,
RAGuard는 single-answer fact verification이라 multi-answer를 권장하면 EM이 0이 된다.

본 모듈은 같은 이름의 RAMDocs용 aggregator를 RAGuard 시나리오에 맞게 변형한다:
- 답을 정확히 하나만 출력 (`["True"]` 또는 `["False"]`)
- evidence 품질을 강조하고 majority vote 의존 줄임
- misleading 문서에 휘둘리지 말라는 명시적 지시

기존 RAMDocs용 함수는 `prompts/madamrag.py`, `prompts/v3.py`에 그대로 보존되어 있다.
pipeline은 dataset 인자에 따라 둘 중 하나를 선택해 사용한다.
"""


# 대응: prompts/madamrag.py::aggregator_prompt
# 사용처: pipelines/madamrag.py (모든 라운드), pipelines/v4.py (Round 2+)
def aggregator_prompt_raguard(query: str, agent_responses: list[str]) -> str:
    responses_text = "\n\n".join(
        f"Agent {i+1}: {r}" for i, r in enumerate(agent_responses)
    )
    return f"""You are an aggregator synthesizing multiple agents' answers to a fact-checking claim.

The claim has EXACTLY ONE correct answer: either "True" or "False".
Choose the single most likely answer based on the agents' responses.
- Weigh agents' reasoning quality, not just majority vote.
- Some agents may be misled by misleading documents — focus on agents whose reasoning is grounded in clear, direct evidence.
- The output list must contain exactly ONE element: ["True"] or ["False"]. Never both.

Please follow the format: 'All Correct Answers: ["True"]. Explanation: {{}}.'
                     or:  'All Correct Answers: ["False"]. Explanation: {{}}.'

The following is an example:
Question: Is the following claim true or false? Answer with "True" or "False".

Claim: "Country X has the lowest GDP in Asia."
Agent responses:
Agent 1: Answer: False. Explanation: The document provides World Bank data showing Country X's GDP is mid-rank in Asia, not the lowest.
Agent 2: Answer: True. Explanation: The document is an opinion blog stating X is poorest, but does not cite economic data.
Agent 3: Answer: Unknown. Explanation: The document does not discuss GDP rankings.
All Correct Answers: ["False"]. Explanation: Agent 1 cited authoritative data directly contradicting the claim. Agent 2's evidence is weak (opinion blog). Agent 3 is uninformative.

Question: {query}
Agent responses:
{responses_text}
"""


# 대응: prompts/v3.py::aggregator_with_confidence_prompt
# 사용처: pipelines/v4.py (Round 1)
def aggregator_with_confidence_prompt_raguard(query: str, mediator_results: list[str]) -> str:
    results_text = "\n\n".join(
        f"Document {i+1} Mediator: {r}" for i, r in enumerate(mediator_results)
    )
    return f"""You are an Aggregator synthesizing mediator judgments for a fact-checking claim.

Each mediator has evaluated Pro (supportive) and Con (opposing) arguments for one document, and provides an answer with a confidence level.

The claim has EXACTLY ONE correct answer: either "True" or "False".

Synthesize all mediator results into a SINGLE final answer:
- Trust High-confidence answers more than Low-confidence ones.
- Some mediators may be misled by misleading documents — weigh evidence quality.
- The output list must contain exactly ONE element: ["True"] or ["False"]. Never both.
- Do NOT include 'Unknown' in the final answer — pick the more likely True/False.

Please follow the format: 'All Correct Answers: ["True"]. Explanation: {{}}.'
                     or:  'All Correct Answers: ["False"]. Explanation: {{}}.'

The following is an example:
Question: Is the following claim true or false? Answer with "True" or "False".

Claim: "Country X has the lowest GDP in Asia."
Document 1 Mediator: Answer: False. Confidence: High. Explanation: Pro provided direct quote with World Bank data; Con's objections were weak.
Document 2 Mediator: Answer: True. Confidence: Low. Explanation: Pro cited an opinion blog; Con pointed out lack of data.
Document 3 Mediator: Answer: Unknown. Confidence: Low. Explanation: Document does not discuss GDP rankings.
All Correct Answers: ["False"]. Explanation: Document 1 has high-confidence direct evidence contradicting the claim. Document 2 is low-confidence with weak source. Document 3 is uninformative.

Question: {query}
Mediator results:
{results_text}
"""
