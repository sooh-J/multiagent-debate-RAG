"""
RAGuard(claim fact-checking) 전용 프롬프트

RAGuard 데이터셋은 정답이 단일 "True" 또는 "False" 두 값 중 하나다.
RAMDocs는 동명이인/동일명 모호성으로 정답이 여러 개 가능한 반면,
RAGuard는 single-answer fact verification이라 multi-answer를 권장하면 EM이 0이 된다.

본 모듈은 RAMDocs용 프롬프트(prompts/madamrag.py, prompts/v3.py, prompts/single_llm.py)에
대응되는 RAGuard 변형을 제공한다. 공통 변경:
  - 답을 단일 "True" 또는 "False"로 강제 (Aggregator는 ["True"] 또는 ["False"])
  - 모든 단계에서 doc-only 제약 명시 — 모델의 사전 지식 / 실세계 팩트 / 외부 정보 사용 금지
    (정치 fact-checking 영역에서 LLM이 internal knowledge로 답하는 것을 차단)
  - Pro/Con/Mediator: 문서에 직접적 evidence 없으면 "No ... evidence found" 또는 Unknown/Low 강제
    (noise 문서에서 억지로 의견을 만들어내는 것을 막아 false bias를 줄이는 목적)

기존 RAMDocs용 함수는 원본 모듈에 그대로 보존되어 있다.
pipeline은 dataset 인자에 따라 둘 중 하나를 선택해 사용한다.
"""


# 모든 프롬프트에 공통 삽입되는 doc-only 제약 문구
_DOC_ONLY = (
    "Use ONLY the information explicitly stated in the document(s) provided below. "
    "Do NOT rely on your prior knowledge about the topic, real-world facts, "
    "or any external sources. If the document does not contain enough information, "
    "say so explicitly rather than guessing from background knowledge."
)


# ──────────────────────────────────────────────────────────────────────────────
# Aggregators
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Single LLM
# ──────────────────────────────────────────────────────────────────────────────

# 대응: prompts/single_llm.py::single_llm_prompt
# 사용처: pipelines/single_llm.py
def single_llm_prompt_raguard(query: str, documents: list[str]) -> str:
    docs_text = "\n\n".join(
        f"Document {i+1}: {doc}" for i, doc in enumerate(documents)
    )
    return f"""You are a fact-checker evaluating a claim against retrieved documents.

The claim has EXACTLY ONE correct answer: either "True" or "False".

{_DOC_ONLY}

Read all documents carefully:
- Some documents may directly support the claim, some may refute it, others may be unrelated or misleading.
- Weigh documents by evidence quality (direct, sourced quotes outweigh opinion or unsupported assertions).
- Output list must contain EXACTLY one element: ["True"] or ["False"]. Never both. Never empty.

Please follow the format: 'All Correct Answers: ["True"]. Explanation: {{}}.'
                     or:  'All Correct Answers: ["False"]. Explanation: {{}}.'

The following is an example:
Question: Is the following claim true or false? Answer with "True" or "False".

Claim: "Country X has the lowest GDP in Asia."
Document 1: World Bank 2023 GDP ranking shows Country X mid-rank in Asia, ahead of several smaller economies.
Document 2: Opinion blog: "Country X is the poorest country in Asia." (no data cited)
Document 3: Unrelated discussion about Country X's tourism industry.
All Correct Answers: ["False"]. Explanation: Document 1 provides authoritative World Bank data directly contradicting the claim. Document 2 is unsourced opinion. Document 3 is irrelevant.

Question: {query}
{docs_text}
"""


# ──────────────────────────────────────────────────────────────────────────────
# MadamRAG agent (also v4 Round 2+)
# ──────────────────────────────────────────────────────────────────────────────

# 대응: prompts/madamrag.py::agent_initial_prompt
# 사용처: pipelines/madamrag.py (Round 1), pipelines/v4.py (간접: Round 2+ 의 agent_debate)
def agent_initial_prompt_raguard(query: str, document: str) -> str:
    return f"""You are an agent reading a single document to evaluate a fact-checking claim.

The claim has EXACTLY ONE correct answer: either "True" or "False".

{_DOC_ONLY}

Based ONLY on this document, decide:
- "True"     — the document directly supports the claim
- "False"    — the document directly refutes the claim
- "Unknown"  — the document does not contain enough information about this claim
              (this is a valid and expected answer for irrelevant or off-topic documents)

Question: {query}
Document: {document}

Please follow the format: 'Answer: {{True / False / Unknown}}. Explanation: {{step-by-step reasoning citing the document}}.'
"""


# 대응: prompts/madamrag.py::agent_debate_prompt
# 사용처: pipelines/madamrag.py (Round 2+), pipelines/v4.py (Round 2+)
def agent_debate_prompt_raguard(query: str, document: str, history: str) -> str:
    return f"""You are an agent reading a single document to evaluate a fact-checking claim.

The claim has EXACTLY ONE correct answer: either "True" or "False".

{_DOC_ONLY}

The following responses are from other agents who read different documents:
{history}

Based ONLY on YOUR document (other agents' answers are for context, not for you to copy):
- "True"     — your document directly supports the claim
- "False"    — your document directly refutes the claim
- "Unknown"  — your document does not contain enough information

Question: {query}
Document: {document}

Please follow the format: 'Answer: {{True / False / Unknown}}. Explanation: {{reasoning citing your document}}.'
"""


# ──────────────────────────────────────────────────────────────────────────────
# v4 Round 1: Pro / Con / Mediator
# ──────────────────────────────────────────────────────────────────────────────

# 대응: prompts/v3.py::pro_prompt
# 사용처: pipelines/v4.py (Round 1)
def pro_prompt_raguard(query: str, document: str) -> str:
    return f"""You are a Pro (supportive) agent. Your role is to argue that THIS document directly supports the claim.

The claim has EXACTLY ONE correct answer: either "True" or "False".

{_DOC_ONLY}

Find and present the strongest direct evidence from this document supporting the claim's verdict (whether True or False — argue whichever direction the document genuinely supports).

CRITICAL:
- If the document does NOT contain direct, specific evidence about this claim,
  reply: 'Verdict: No supporting evidence found. Reasoning: {{why the document is irrelevant or off-topic}}.'
  Do NOT invent or stretch weak connections — being honest here is essential.
- Quote exact sentences from the document as evidence. Do not paraphrase.

Question: {query}
Document: {document}

Please follow the format:
'Verdict: {{True / False}}. Evidence: {{exact quote from document}}. Argument: {{why this evidence supports the verdict}}.'
or, if the document is irrelevant:
'Verdict: No supporting evidence found. Reasoning: {{why}}.'
"""


# 대응: prompts/v3.py::con_prompt
# 사용처: pipelines/v4.py (Round 1)
def con_prompt_raguard(query: str, document: str) -> str:
    return f"""You are a Con (opposing) agent. Your role is to critically examine whether this document actually provides reliable evidence about the claim.

The claim has EXACTLY ONE correct answer: either "True" or "False".

{_DOC_ONLY}

Critique focus:
1. Does the document directly discuss the claim, or is it tangential / off-topic?
2. Is the document's content actually evidence, or is it opinion / unsupported assertion / link-only / boilerplate?
3. Could a Pro agent's reading be a misinterpretation of what the document actually says?

CRITICAL:
- If the document does NOT contain direct, reliable evidence relevant to the claim
  (e.g., it is an unrelated Reddit post, opinion piece without data, or only tangentially mentions the topic),
  reply: 'Verdict: No reliable evidence found. Reasoning: {{why the document is unreliable or off-topic}}.'
  Do NOT invent objections to a document that simply has no evidence.

Question: {query}
Document: {document}

Please follow the format:
'Verdict: {{Unreliable / Insufficient / Misleading / No Answer}}. Reasoning: {{specific issues found in the document}}.'
or:
'Verdict: No reliable evidence found. Reasoning: {{why}}.'
"""


# 대응: prompts/v3.py::mediator_prompt
# 사용처: pipelines/v4.py (Round 1)
def mediator_prompt_raguard(query: str, document: str, pro_response: str, con_response: str) -> str:
    return f"""You are a Mediator agent. Your role is to judge the Pro and Con arguments and produce a final answer for THIS document about the claim.

The claim has EXACTLY ONE correct answer: either "True" or "False" (or "Unknown" if this document does not contribute).

{_DOC_ONLY}

Question: {query}
Document: {document}

Pro agent's argument:
{pro_response}

Con agent's argument:
{con_response}

Decision rule:
- If Pro provides direct evidence (exact quote) that genuinely supports a verdict, AND Con's objections are weak,
  → Answer: True or False (whichever Pro identified), Confidence: High or Medium.
- If Pro and Con disagree on interpretation but the document is genuinely on-topic,
  → pick the more strongly supported side, Confidence: Medium.
- If Pro reported "No supporting evidence found", or Con reported "No reliable evidence found",
  or BOTH sides have weak/vague arguments,
  → Answer: Unknown, Confidence: Low. (This is the correct answer for noise / off-topic documents.)

Important: Be willing to answer Unknown / Low confidence freely. A noisy off-topic document SHOULD get Unknown — guessing causes downstream errors.

Please follow the format:
'Answer: {{True / False / Unknown}}. Confidence: {{High / Medium / Low}}. Explanation: {{which arguments you accepted and why}}.'
"""
