"""
RAGuard 전용 프롬프트 v2 — noise-aware Pro/Con + balanced few-shot (True/False 모두)

prompts/raguard.py 의 v2 파생. 기존 v1 prompts 는 보존, 본 모듈은 ablation 용.

v1 → v2 변경 사항:

1) Pro / Con — "Step 0: is this document on-topic?" 를 prompt 맨 앞에 명시
   - v1: noise escape ("No evidence found") 가 CRITICAL 블록에 숨어 있어 모델이
         noise doc 에서도 verdict 를 stretch하는 경향이 있었음
   - v2: 첫 단계가 unrelated/off-topic 판단. relevant 일 때만 verdict 로 진행

2) Few-shot 예시 균형 — 기존 모든 예시가 gold=False 였음
   - v2: Country X(False) + Country Y(True) 양쪽 예시를 모두 노출
   - 영향 함수: single_llm / aggregator / aggregator_with_confidence

3) Mediator — Unknown 옵션을 더 명시 (Decision rule 첫 줄로 끌어올림)

3 (TODO 미적용): Mediator confidence 를 logit 기반 p(True) 로 대체
   → 별도 실험 (이 파일은 prompt 변경만)
"""


# 모든 프롬프트에 공통 삽입되는 doc-only 제약 문구 (v1 과 동일)
_DOC_ONLY = (
    "Use ONLY the information explicitly stated in the document(s) provided below. "
    "Do NOT rely on your prior knowledge about the topic, real-world facts, "
    "or any external sources. If the document does not contain enough information, "
    "say so explicitly rather than guessing from background knowledge."
)


# ──────────────────────────────────────────────────────────────────────────────
# Aggregators — balanced few-shot (True + False example)
# ──────────────────────────────────────────────────────────────────────────────

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

The following are examples:

Example 1 (gold=False):
Question: Is the following claim true or false? Answer with "True" or "False".

Claim: "Country X has the lowest GDP in Asia."
Agent responses:
Agent 1: Answer: False. Explanation: The document provides World Bank data showing Country X's GDP is mid-rank in Asia, not the lowest.
Agent 2: Answer: True. Explanation: The document is an opinion blog stating X is poorest, but does not cite economic data.
Agent 3: Answer: Unknown. Explanation: The document does not discuss GDP rankings.
All Correct Answers: ["False"]. Explanation: Agent 1 cited authoritative data directly contradicting the claim. Agent 2's evidence is weak (opinion blog). Agent 3 is uninformative.

Example 2 (gold=True):
Question: Is the following claim true or false? Answer with "True" or "False".

Claim: "Country Y has more than 100 million citizens."
Agent responses:
Agent 1: Answer: True. Explanation: The document quotes the World Bank 2022 census reporting Country Y's population at 218 million.
Agent 2: Answer: Unknown. Explanation: My document is a travel blog about Country Y's markets; it doesn't mention population figures.
Agent 3: Answer: True. Explanation: The document is a 2023 UN statistics brief listing Country Y as one of the world's largest populations.
All Correct Answers: ["True"]. Explanation: Agents 1 and 3 cite authoritative demographic sources directly supporting the claim. Agent 2 is uninformative.

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
- Some mediators may be misled by misleading documents — weigh evidence quality.
- The output list must contain exactly ONE element: ["True"] or ["False"]. Never both.
- Do NOT include 'Unknown' in the final answer — pick the more likely True/False.

Please follow the format: 'All Correct Answers: ["True"]. Explanation: {{}}.'
                     or:  'All Correct Answers: ["False"]. Explanation: {{}}.'

The following are examples:

Example 1 (gold=False):
Question: Is the following claim true or false? Answer with "True" or "False".

Claim: "Country X has the lowest GDP in Asia."
Document 1 Mediator: Answer: False. Confidence: High. Explanation: Pro provided direct quote with World Bank data; Con's objections were weak.
Document 2 Mediator: Answer: True. Confidence: Low. Explanation: Pro cited an opinion blog; Con pointed out lack of data.
Document 3 Mediator: Answer: Unknown. Confidence: Low. Explanation: Document does not discuss GDP rankings.
All Correct Answers: ["False"]. Explanation: Document 1 has high-confidence direct evidence contradicting the claim. Document 2 is low-confidence with weak source. Document 3 is uninformative.

Example 2 (gold=True):
Question: Is the following claim true or false? Answer with "True" or "False".

Claim: "Country Y has more than 100 million citizens."
Document 1 Mediator: Answer: True. Confidence: High. Explanation: Pro directly quoted World Bank 2022 census showing 218M citizens; Con's objections were vague.
Document 2 Mediator: Answer: Unknown. Confidence: Low. Explanation: Document is a travel blog with no demographic data.
Document 3 Mediator: Answer: True. Confidence: Medium. Explanation: Pro cited a 2023 UN brief; Con noted minor uncertainty on counting methodology.
All Correct Answers: ["True"]. Explanation: Documents 1 and 3 give high/medium confidence evidence supporting the claim. Document 2 is uninformative.

Question: {query}
Mediator results:
{results_text}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Single LLM — balanced few-shot
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
# MadamRAG agent (also v4 Round 2+) — Unknown 강조 (기능적으로 v1 과 동일)
# ──────────────────────────────────────────────────────────────────────────────

def agent_initial_prompt_raguard(query: str, document: str) -> str:
    return f"""You are an agent reading a single document to evaluate a fact-checking claim.

The claim has EXACTLY ONE correct answer: either "True" or "False".

{_DOC_ONLY}

Step 1 (do this first): Decide whether THIS document is actually on-topic for the claim.
- If the document is unrelated, off-topic, or contains no information about the claim →
    Answer: Unknown.   (This is a valid and expected answer.)

Step 2 (only if the document IS on-topic):
- "True"   — the document directly supports the claim
- "False"  — the document directly refutes the claim

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

Step 1 (do this first): Decide whether YOUR document is actually on-topic for the claim.
- If your document is unrelated, off-topic, or contains no information about the claim →
    Answer: Unknown.   (Do not be pressured by other agents' answers — irrelevance is a valid answer.)

Step 2 (only if your document IS on-topic):
- "True"   — your document directly supports the claim
- "False"  — your document directly refutes the claim

Question: {query}
Document: {document}

Please follow the format: 'Answer: {{True / False / Unknown}}. Explanation: {{reasoning citing your document}}.'
"""


# ──────────────────────────────────────────────────────────────────────────────
# v4 Round 1: Pro / Con / Mediator — Step 0 unrelated check at the TOP
# ──────────────────────────────────────────────────────────────────────────────

def pro_prompt_raguard(query: str, document: str) -> str:
    return f"""You are a Pro (supportive) agent. Your role is to argue that THIS document directly supports the claim's verdict (either True or False, whichever the document genuinely supports).

The claim has EXACTLY ONE correct answer: either "True" or "False".

{_DOC_ONLY}

STEP 0 (do this FIRST, before anything else):
Determine whether this document is on-topic for the claim.
- "On-topic" means: the document discusses the specific subject/event/entity in the claim and provides factual information (data, quote, source, or evidence) that bears on whether the claim is True or False.
- "Off-topic / noise" means: the document is about a different subject, is a tangential mention only, is pure opinion without data, is a link or boilerplate, or simply does not address the claim.

If OFF-TOPIC / NOISE, stop here and reply EXACTLY:
  'Verdict: No supporting evidence found. Reasoning: {{why the document does not address the claim}}.'

Do NOT invent or stretch weak connections to make a noise document look relevant.
This is the most important rule: noise documents must be reported as noise, not argued for.

STEP 1 (only if the document IS on-topic):
Find and quote the strongest direct evidence from the document supporting a verdict. Quote exact sentences — do not paraphrase.

Question: {query}
Document: {document}

Please follow ONE of these formats:
(on-topic case)
  'Verdict: {{True / False}}. Evidence: {{exact quote from document}}. Argument: {{why this evidence supports the verdict}}.'
(off-topic / noise case)
  'Verdict: No supporting evidence found. Reasoning: {{why}}.'
"""


def con_prompt_raguard(query: str, document: str) -> str:
    return f"""You are a Con (opposing) agent. Your role is to critically examine whether this document actually provides reliable evidence about the claim.

The claim has EXACTLY ONE correct answer: either "True" or "False".

{_DOC_ONLY}

STEP 0 (do this FIRST, before critiquing):
Determine whether this document is on-topic and reliable for the claim.
- "On-topic & reliable" means: the document discusses the claim's specific subject and provides verifiable evidence (data, quote, source).
- "Off-topic / unreliable" means: unrelated subject, opinion without data, anonymous forum post, link-only / boilerplate, or only tangentially mentions the topic.

If OFF-TOPIC / UNRELIABLE, stop here and reply EXACTLY:
  'Verdict: No reliable evidence found. Reasoning: {{why the document is unreliable or off-topic}}.'

Do NOT manufacture objections to a document that simply has no evidence — saying "no evidence" is itself a valid and important Con verdict for noise documents.

STEP 1 (only if the document IS on-topic):
Critique what the Pro side might claim:
1. Could the document's content be misinterpreted as evidence when it is not?
2. Is the source / methodology weak (opinion vs. data, anecdote vs. sourced quote)?
3. Does the document only tangentially mention the topic without taking a position?

Question: {query}
Document: {document}

Please follow ONE of these formats:
(on-topic case)
  'Verdict: {{Unreliable / Insufficient / Misleading / No Answer}}. Reasoning: {{specific issues found in the document}}.'
(off-topic case)
  'Verdict: No reliable evidence found. Reasoning: {{why}}.'
"""


def mediator_prompt_raguard(query: str, document: str, pro_response: str, con_response: str) -> str:
    return f"""You are a Mediator agent. Your role is to judge the Pro and Con arguments and produce a final answer for THIS document about the claim.

The claim has EXACTLY ONE correct answer: either "True" or "False" — BUT this single document may not be enough to decide. In that case, Answer: Unknown is the correct judgment for THIS document; the Aggregator will combine signals across documents.

{_DOC_ONLY}

Question: {query}
Document: {document}

Pro agent's argument:
{pro_response}

Con agent's argument:
{con_response}

Decision rule (apply in order):
1. If EITHER Pro said "No supporting evidence found" OR Con said "No reliable evidence found",
   → Answer: Unknown, Confidence: Low. (This is the correct verdict for off-topic / noise documents — do not try to rescue.)
2. If Pro provides a direct, exact quote that supports a verdict AND Con's objections are weak/vague,
   → Answer: True or False (whichever Pro identified), Confidence: High or Medium.
3. If Pro and Con disagree on interpretation but the document is genuinely on-topic with real evidence on both sides,
   → pick the more strongly supported side, Confidence: Medium.
4. If both sides are weak/vague,
   → Answer: Unknown, Confidence: Low.

Important: Answer Unknown FREELY when the document is noisy or off-topic. A confident wrong answer downstream is worse than an honest Unknown here.

Please follow the format:
'Answer: {{True / False / Unknown}}. Confidence: {{High / Medium / Low}}. Explanation: {{which arguments you accepted and why}}.'
"""
