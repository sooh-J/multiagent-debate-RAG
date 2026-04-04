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

  2단계 (Round 2+ 용) — 이전 라운드 다른 문서의 Mediator 결과를 참고:
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

Important:
- Your answer must be as SHORT and CONCISE as possible (e.g., a name, a number, a year, a short phrase). Do NOT write full sentences as the answer.
- Use the exact key term or phrase from the document. Do not paraphrase or expand.
- This document may refer to a DIFFERENT entity that shares the same name as the one in the question. If so, still extract the answer — it is valid.

Please follow the format:
'Answer: {{}}. Evidence: {{exact quote from document}}. Argument: {{why this evidence supports the answer}}.'"""


def con_prompt(query: str, document: str) -> str:
    return f"""You are a Con (opposing) agent. Your role is to critically examine whether this document actually answers the question reliably.

Question: {query}
Document: {document}

Focus your critique ONLY on these issues:
1. Does the document actually discuss the entity asked about, or is it about something else entirely?
2. Does the quoted evidence truly support the claimed answer, or is it a misreading?
3. Is the answer based on actual content in the document, or is it inferred/fabricated?

Do NOT criticize the document for being outdated or lacking recent data — your job is to evaluate whether the document's own content supports the answer, not whether the data is current.

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
2. Are the Con's objections valid, or are they nitpicking? (Note: objections about data being "outdated" are NOT valid — focus on whether the evidence supports the answer.)
3. This document may refer to a DIFFERENT entity that shares the same name. If the document clearly answers the question for its specific entity, that answer is VALID even if other documents give different answers.

If the Con's objections are strong enough (e.g., the document does not actually contain an answer, or the evidence is fabricated), answer 'Unknown'. Otherwise, provide the answer the Pro identified.

Important: Your answer must be SHORT and CONCISE (e.g., a name, number, year, short phrase). Copy the exact wording from the Pro's answer without paraphrasing.

Please follow the format:
'Answer: {{}}. Confidence: {{High/Medium/Low}}. Explanation: {{which arguments you accepted and why}}.'"""


# ── Round 2+: 이전 라운드 요약을 반영하는 토론 프롬프트 ────────────────────────

def pro_debate_prompt(query: str, document: str, other_mediator_results: str) -> str:
    return f"""You are a Pro (supportive) agent. Your role is to argue that this document DOES contain an accurate answer to the question.

Question: {query}
Document: {document}

Here are the mediator judgments from other documents in the previous round:
{other_mediator_results}

Considering what other documents' mediators concluded, strengthen or revise your argument. If other mediators found different answers, explain why your document's answer is also valid. If other mediators raised concerns relevant to your document, address them.

Important:
- Your answer must be as SHORT and CONCISE as possible (e.g., a name, a number, a year, a short phrase). Do NOT write full sentences as the answer.
- Use the exact key term or phrase from the document. Do not paraphrase or expand.
- This document may refer to a DIFFERENT entity that shares the same name. Different answers from different documents can ALL be valid.

Please follow the format:
'Answer: {{}}. Evidence: {{exact quote from document}}. Argument: {{why this evidence supports the answer}}.'"""


def con_debate_prompt(query: str, document: str, other_mediator_results: str) -> str:
    return f"""You are a Con (opposing) agent. Your role is to critically examine whether this document actually answers the question reliably.

Question: {query}
Document: {document}

Here are the mediator judgments from other documents in the previous round:
{other_mediator_results}

Considering what other documents' mediators concluded, strengthen or revise your objections.

Focus your critique ONLY on these issues:
1. Does the document actually discuss the entity asked about, or is it about something else entirely?
2. Does the quoted evidence truly support the claimed answer, or is it a misreading?
3. Is the answer based on actual content in the document, or is it inferred/fabricated?

Do NOT criticize the document for being outdated or lacking recent data.

Please follow the format:
'Verdict: {{Unreliable/Insufficient/Misleading/No Answer}}. Reasoning: {{specific issues found}}.'"""


def mediator_debate_prompt(query: str, document: str, pro_response: str, con_response: str, other_mediator_results: str) -> str:
    return f"""You are a Mediator agent. Your role is to judge the validity of the Pro and Con arguments, then determine the final answer from this document.

Question: {query}
Document: {document}

Mediator judgments from other documents in the previous round:
{other_mediator_results}

Pro agent's argument (this round):
{pro_response}

Con agent's argument (this round):
{con_response}

Evaluate both arguments, taking into account what other documents' mediators concluded:
1. Is the Pro's evidence actually present in the document and does it support the answer?
2. Are the Con's objections valid, or are they nitpicking? (Objections about data being "outdated" are NOT valid.)
3. This document may refer to a DIFFERENT entity that shares the same name. If the document clearly answers the question for its specific entity, that answer is VALID.

Important: Your answer must be SHORT and CONCISE (e.g., a name, number, year, short phrase). Copy the exact wording from the Pro's answer without paraphrasing.

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
- CRITICAL: The question may be AMBIGUOUS — different documents may refer to DIFFERENT entities with the SAME name (e.g., different people, places, or things). Each entity's answer is independently valid. Include ALL distinct valid answers.
- Only exclude answers where the mediator said 'Unknown' or gave Low confidence with weak reasoning.
- Each answer must use the exact wording as it appears in the mediator results. Do not paraphrase, abbreviate, or expand.
- Answers must be SHORT (e.g., a name, number, year, short phrase). If a mediator gave a long answer, extract only the key fact.

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
