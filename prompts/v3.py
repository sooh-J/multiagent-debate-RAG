"""
V3 / V4 공통 프롬프트 정의

찬/반/중재자 (Pro/Con/Mediator) 기반 다중 에이전트 토론 프롬프트.
V3와 V4 모두 이 프롬프트를 공유한다.

에이전트 역할:
  - Pro (찬성):   "이 문서가 질문에 대한 정확한 답을 포함한다"는 입장에서 근거를 제시
  - Con (반대):   "이 문서가 질문에 답할 수 없거나 정보가 부정확하다"는 입장에서 반박
  - Mediator (중재자): Pro와 Con의 주장을 모두 검토한 뒤 최종 판정을 내림
                       문서 자체를 평가하는 것이 아니라, 찬/반 주장의 타당성을 판단

프롬프트 종류:
  1단계 (Round 1용) — 이전 라운드 정보 없이 문서만 보고 판단:
    - pro_prompt           : 찬성 에이전트 초기 프롬프트
    - con_prompt           : 반대 에이전트 초기 프롬프트
    - mediator_prompt      : 중재자 초기 프롬프트 (Pro/Con 응답을 입력받음)

  2단계 (Round 2+ 용) — 이전 라운드 Aggregator 요약을 추가로 참고:
    - pro_debate_prompt    : 찬성 에이전트 토론 프롬프트
    - con_debate_prompt    : 반대 에이전트 토론 프롬프트
    - mediator_debate_prompt : 중재자 토론 프롬프트

  Aggregator:
    - aggregator_with_confidence_prompt : 모든 문서의 중재자 결과를 종합하여
                                          confidence(High/Medium/Low)와 함께 최종 답변 도출
"""


# ── Round 1: 초기 프롬프트 ───────────────────────────────────────────────────

def pro_prompt(query: str, document: str) -> str:
    return f"""You are a Pro (supportive) agent. Your role is to argue that this document DOES contain an accurate answer to the question.

Question: {query}
Document: {document}

Find and present the strongest evidence from this document that answers the question. Quote exact sentences as evidence. Even if the evidence is imperfect, make the best possible case that this document provides a valid answer.

Please follow the format:
'Answer: {{}}. Evidence: {{exact quote from document}}. Argument: {{why this evidence supports the answer}}.'"""


def con_prompt(query: str, document: str) -> str:
    return f"""You are a Con (opposing) agent. Your role is to argue that this document does NOT contain an accurate or reliable answer to the question.

Question: {query}
Document: {document}

Critically examine this document and argue why it fails to answer the question correctly. Look for:
1. Insufficient or irrelevant evidence
2. Misleading or outdated information
3. Mismatches between the question and the document's content
4. Potential misinformation or factual errors

Please follow the format:
'Verdict: {{Unreliable/Insufficient/Misleading/No Answer}}. Reasoning: {{specific issues found}}.'"""


def mediator_prompt(query: str, document: str, pro_response: str, con_response: str) -> str:
    return f"""You are a Mediator agent. Your role is to judge the validity of the Pro and Con arguments, then determine the final answer from this document.

Question: {query}
Document: {document}

Pro agent's argument (claims the document answers the question):
{pro_response}

Con agent's argument (claims the document does NOT reliably answer the question):
{con_response}

Evaluate both arguments fairly:
1. Is the Pro's evidence actually present in the document and does it support the answer?
2. Are the Con's objections valid, or are they nitpicking?
3. Based on the balance of arguments, what is the most reliable answer from this document?

If the Con's objections are strong enough, answer 'Unknown'. Otherwise, provide the answer the Pro identified, adjusted for any valid concerns the Con raised.

Please follow the format:
'Answer: {{}}. Confidence: {{High/Medium/Low}}. Explanation: {{which arguments you accepted and why}}.'"""


# ── Round 2+: 이전 라운드 요약을 반영하는 토론 프롬프트 ────────────────────────

def pro_debate_prompt(query: str, document: str, prev_summary: str) -> str:
    return f"""You are a Pro (supportive) agent. Your role is to argue that this document DOES contain an accurate answer to the question.

Question: {query}
Document: {document}

Here is the aggregated summary from the previous round:
{prev_summary}

Considering the previous round's conclusions, strengthen or revise your argument. If the previous summary raised concerns about this document, address them with stronger evidence. If the summary supports your position, reinforce it.

Please follow the format:
'Answer: {{}}. Evidence: {{exact quote from document}}. Argument: {{why this evidence supports the answer}}.'"""


def con_debate_prompt(query: str, document: str, prev_summary: str) -> str:
    return f"""You are a Con (opposing) agent. Your role is to argue that this document does NOT contain an accurate or reliable answer to the question.

Question: {query}
Document: {document}

Here is the aggregated summary from the previous round:
{prev_summary}

Considering the previous round's conclusions, strengthen or revise your objections. If the previous summary accepted this document's answer, challenge the reasoning. If it rejected the answer, reinforce the concerns.

Please follow the format:
'Verdict: {{Unreliable/Insufficient/Misleading/No Answer}}. Reasoning: {{specific issues found}}.'"""


def mediator_debate_prompt(query: str, document: str, pro_response: str, con_response: str, prev_summary: str) -> str:
    return f"""You are a Mediator agent. Your role is to judge the validity of the Pro and Con arguments, then determine the final answer from this document.

Question: {query}
Document: {document}

Previous round's aggregated summary:
{prev_summary}

Pro agent's argument (this round):
{pro_response}

Con agent's argument (this round):
{con_response}

Evaluate both arguments, taking into account the previous round's conclusions:
1. Is the Pro's evidence actually present in the document and does it support the answer?
2. Are the Con's objections valid, or are they nitpicking?
3. Has either side addressed concerns from the previous round?

Please follow the format:
'Answer: {{}}. Confidence: {{High/Medium/Low}}. Explanation: {{which arguments you accepted and why}}.'"""


# ── Aggregator: confidence 포함 종합 ─────────────────────────────────────────

def aggregator_with_confidence_prompt(query: str, mediator_results: list[str]) -> str:
    results_text = "\n\n".join(
        f"Document {i+1} Mediator: {r}" for i, r in enumerate(mediator_results)
    )
    return f"""You are an Aggregator synthesizing mediator judgments from multiple documents.

Each mediator has already evaluated Pro (supportive) and Con (opposing) arguments for their document, and provides an answer with a confidence level.

Synthesize all mediator results into a final answer:
- Trust High-confidence answers more than Low-confidence ones.
- If multiple documents give different answers to an ambiguous question (e.g., different entities with the same name), include all valid answers.
- Exclude answers where the mediator said 'Unknown' or gave Low confidence with weak reasoning.
- Each answer must use the exact wording as it appears in the mediator results. Do not paraphrase, abbreviate, or expand.

Please follow the format: 'All Correct Answers: []. Explanation: {{}}.'

The following are examples:
Question: In which year was Michael Jordan born?
Document 1 Mediator: Answer: 1963. Confidence: High. Explanation: Pro provided direct quote from document stating Michael Jeffrey Jordan was born February 17, 1963. Con had no valid objections.
Document 2 Mediator: Answer: 1956. Confidence: High. Explanation: Pro showed the document refers to Michael Irwin Jordan, born 1956. Con argued this might not be the intended Michael Jordan, but the question is ambiguous.
Document 3 Mediator: Answer: Unknown. Confidence: Low. Explanation: Con successfully argued the document's date (1998) contradicts reliable sources. Pro's evidence was weak.
All Correct Answers: ["1963", "1956"]. Explanation: Documents 1 and 2 provide high-confidence answers for different people named Michael Jordan. Document 3 was judged unreliable by the mediator.

Question: {query}
Mediator results:
{results_text}
"""
