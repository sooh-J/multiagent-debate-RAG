"""
RAGuard 전용 프롬프트 v2 — "v1 + surgical Aggregator fix"

설계 원칙:
- v1 (prompts/raguard.py) 의 Pro/Con/Mediator/single_llm/agent_* 안전장치는 모두 유지
  (이전 NEW v2 rollback 실험에서 안전장치 제거 시 Qwen T-acc 0%, LLAMA T-acc 절반 폭락 확인)
- Aggregator 의 "may be misled by misleading documents" 라인만 제거 + "do not re-evaluate" 강조
  (failure analysis 의 Mode A — Aggregator override 39% — 만 정밀 타겟)
- 모든 examples 에 True / False 균형 (v1 은 False 만)

Failure mode 정량 분석 (LLAMA v4 v1 의 94 wrong cases):
  Mode A 39% — Aggregator 가 Mediator 과반 override (← 본 v2' 의 fix target)
  Mode C 33% — Mediator 가 correct doc 에서 verdict 놓침
  Mode D 31% — Mediator 출력 과반이 Unknown (sparse signal)
  Mode B 20% — Pro 가 noise doc 에서 verdict 환각

v2' 는 Mode A 만 정밀 타겟. C/D 는 Mediator 단계 변경이 필요해서 다음 실험으로 미룸
(Mediator 의 Unknown FREELY 안전장치를 깨지 않고 C/D 개선하는 방법 = TODO).

후속 실험(TODO #3): Mediator confidence 를 logit 기반 p(True) 로 대체.
  자세한 동기/계획은 RAGUARD_V2.md 참고.
"""


# 모든 프롬프트에 공통 삽입되는 doc-only 제약 문구 (v1 과 동일)
_DOC_ONLY = (
    "Use ONLY the information explicitly stated in the document(s) provided below. "
    "Do NOT rely on your prior knowledge about the topic, real-world facts, "
    "or any external sources. If the document does not contain enough information, "
    "say so explicitly rather than guessing from background knowledge."
)


# ──────────────────────────────────────────────────────────────────────────────
# Aggregators — v1 base + "may be misled" 제거 + "do not re-evaluate" 추가 + balanced few-shot
# ──────────────────────────────────────────────────────────────────────────────

def aggregator_prompt_raguard(query: str, agent_responses: list[str]) -> str:
    responses_text = "\n\n".join(
        f"Agent {i+1}: {r}" for i, r in enumerate(agent_responses)
    )
    return f"""You are an aggregator synthesizing multiple agents' answers to a fact-checking claim.

The claim has EXACTLY ONE correct answer: either "True" or "False".
Combine the agents' verdicts into a single answer:
- Weigh agents' reasoning quality, not just majority vote.
- DO NOT re-evaluate the documents yourself. Your job is to combine the agents' verdicts, not to second-guess them based on your own reading.
- Exclude agents that answered "Unknown".
- The output list must contain exactly ONE element: ["True"] or ["False"]. Never both.

Please follow the format: 'All Correct Answers: ["True"]. Explanation: {{}}.'
                     or:  'All Correct Answers: ["False"]. Explanation: {{}}.'

The following are examples:

Example 1 (gold=False):
Question: Is the following claim true or false? Answer with "True" or "False".

Claim: "Country X has the lowest GDP in Asia."
Agent responses:
Agent 1: Answer: False. Explanation: The document provides World Bank data showing Country X's GDP is mid-rank in Asia, not the lowest.
Agent 2: Answer: False. Explanation: The document lists several Asian countries with lower GDP than Country X.
Agent 3: Answer: Unknown. Explanation: The document does not discuss GDP rankings.
All Correct Answers: ["False"]. Explanation: Agents 1 and 2 both refute the claim from their documents. Agent 3 is uninformative.

Example 2 (gold=True):
Question: Is the following claim true or false? Answer with "True" or "False".

Claim: "Country Y has more than 100 million citizens."
Agent responses:
Agent 1: Answer: True. Explanation: The document quotes the World Bank 2022 census reporting Country Y's population at 218 million.
Agent 2: Answer: Unknown. Explanation: My document is a travel blog and does not mention population.
Agent 3: Answer: True. Explanation: The document is a UN brief listing Country Y among the most populous nations.
All Correct Answers: ["True"]. Explanation: Agents 1 and 3 both support the claim with demographic data. Agent 2 is uninformative.

Question: {query}
Agent responses:
{responses_text}
"""


def aggregator_with_confidence_prompt_raguard(query: str, mediator_results: list[str]) -> str:
    results_text = "\n\n".join(
        f"Document {i+1} Mediator: {r}" for i, r in enumerate(mediator_results)
    )
    return f"""You are an Aggregator synthesizing mediator judgments for a fact-checking claim.

Each mediator has evaluated Pro (supportive) and Con (opposing) arguments for one document, and provides an answer with a confidence level.

The claim has EXACTLY ONE correct answer: either "True" or "False".

Synthesize all mediator results into a SINGLE final answer:
- Trust High-confidence answers more than Low-confidence ones.
- DO NOT re-evaluate the documents yourself. Your job is to combine the mediators' verdicts, not to second-guess them based on your own reading.
- Exclude mediators that answered "Unknown".
- The output list must contain exactly ONE element: ["True"] or ["False"]. Never both.

Please follow the format: 'All Correct Answers: ["True"]. Explanation: {{}}.'
                     or:  'All Correct Answers: ["False"]. Explanation: {{}}.'

The following are examples:

Example 1 (gold=False):
Question: Is the following claim true or false? Answer with "True" or "False".

Claim: "Country X has the lowest GDP in Asia."
Document 1 Mediator: Answer: False. Confidence: High. Explanation: Pro provided direct quote with World Bank data; Con's objections were weak.
Document 2 Mediator: Answer: False. Confidence: Medium. Explanation: Pro showed a comparison table; Con noted minor methodology questions.
Document 3 Mediator: Answer: Unknown. Confidence: Low. Explanation: Document does not discuss GDP rankings.
All Correct Answers: ["False"]. Explanation: Documents 1 and 2 reach False with reasonable confidence. Document 3 is uninformative.

Example 2 (gold=True):
Question: Is the following claim true or false? Answer with "True" or "False".

Claim: "Country Y has more than 100 million citizens."
Document 1 Mediator: Answer: True. Confidence: High. Explanation: Pro directly quoted the 2022 World Bank census (218M citizens).
Document 2 Mediator: Answer: Unknown. Confidence: Low. Explanation: Travel blog with no demographic data.
Document 3 Mediator: Answer: True. Confidence: Medium. Explanation: UN statistical brief supporting the claim.
All Correct Answers: ["True"]. Explanation: Documents 1 and 3 reach True with reasonable confidence. Document 2 is uninformative.

Question: {query}
Mediator results:
{results_text}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Single LLM — v1 그대로 + True 예시 추가 (balanced few-shot)
# ──────────────────────────────────────────────────────────────────────────────

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

The following are examples:

Example 1 (gold=False):
Question: Is the following claim true or false? Answer with "True" or "False".

Claim: "Country X has the lowest GDP in Asia."
Document 1: World Bank 2023 GDP ranking shows Country X mid-rank in Asia, ahead of several smaller economies.
Document 2: Opinion blog: "Country X is the poorest country in Asia." (no data cited)
Document 3: Unrelated discussion about Country X's tourism industry.
All Correct Answers: ["False"]. Explanation: Document 1 provides authoritative World Bank data directly contradicting the claim. Document 2 is unsourced opinion. Document 3 is irrelevant.

Example 2 (gold=True):
Question: Is the following claim true or false? Answer with "True" or "False".

Claim: "Country Y has more than 100 million citizens."
Document 1: World Bank 2022 census report: Country Y's population is 218 million.
Document 2: Travel blog about Country Y's busy markets (no demographic data).
Document 3: 2023 UN population brief listing Country Y as one of the top 10 most populous nations worldwide.
All Correct Answers: ["True"]. Explanation: Documents 1 and 3 provide direct, authoritative demographic data supporting the claim. Document 2 is unrelated.

Question: {query}
{docs_text}
"""


# ──────────────────────────────────────────────────────────────────────────────
# MadamRAG agent (also v4 Round 2+) — v1 그대로 (안전장치 유지)
# ──────────────────────────────────────────────────────────────────────────────

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
# v4 Round 1: Pro / Con / Mediator — v1 그대로 (CRITICAL escape 살림, Unknown FREELY 살림)
# ──────────────────────────────────────────────────────────────────────────────

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
