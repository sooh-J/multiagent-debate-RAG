"""
Proposed Method 프롬프트 정의

2단계 구조:
  1단계 — 문서 내부 토론 (Extractor → Skeptic → Resolver)
  2단계 — 문서 간 집계 (Global Aggregator)
"""


# ── 1단계: 문서 내부 토론 ─────────────────────────────────────────────────────

def extractor_prompt(query: str, document: str) -> str:
    return f"""You are an Extractor agent. Your job is to find answer candidates and supporting evidence from the document.

Question: {query}
Document: {document}

Extract all possible answer candidates from this document. For each candidate, quote the exact sentence(s) from the document that support it.
Please follow the format:
'Answer: {{}}. Evidence: {{exact quote from document}}.'"""


def skeptic_prompt(query: str, document: str, extractor_response: str) -> str:
    return f"""You are a Skeptic agent. Your job is to verify whether the Extractor's answer is accurate, or if it involves exaggeration, misreading, or unsupported inference.

Question: {query}
Document: {document}

Extractor's response:
{extractor_response}

Critically examine the Extractor's answer:
1. Does the evidence actually support the answer?
2. Is the answer an exaggeration or misinterpretation of the document?
3. Is there any important context the Extractor missed?

Please follow the format:
'Verdict: {{Supported/Unsupported/Partially Supported}}. Reasoning: {{}}.'"""


def resolver_prompt(query: str, document: str, extractor_response: str, skeptic_response: str) -> str:
    return f"""You are a Resolver agent. Your job is to synthesize the Extractor's findings and the Skeptic's critique to produce a final, reliable summary for this document.

Question: {query}
Document: {document}

Extractor's response:
{extractor_response}

Skeptic's response:
{skeptic_response}

Based on both perspectives, determine the final answer from this document. If the Skeptic found valid issues, adjust the answer accordingly. If the document cannot answer the question, say 'Unknown'.
Please follow the format:
'Answer: {{}}. Explanation: {{}}.'"""


# ── 2단계: 문서 간 집계 ───────────────────────────────────────────────────────

def global_aggregator_prompt(query: str, document_summaries: list[str]) -> str:
    summaries_text = "\n\n".join(
        f"Document {i+1} Summary: {s}" for i, s in enumerate(document_summaries)
    )
    return f"""You are a Global Aggregator synthesizing verified summaries from multiple documents.

Each summary below has already been through an internal verification process (extraction → skeptical review → resolution). Trust verified answers more than unverified ones.

If there are multiple valid answers (e.g., the question is ambiguous and refers to different entities), provide all correct answers. If no document provides a reliable answer, reply 'unknown'.
Each answer must use the exact wording as it appears in the document summaries. Do not paraphrase, abbreviate, or expand.
Please follow the format: 'All Correct Answers: []. Explanation: {{}}.'

The following are examples:
Question: In which year was Michael Jordan born?
Document summaries:
Document 1 Summary: Answer: 1963. Explanation: The document states Michael Jeffrey Jordan was born on February 17, 1963. Evidence directly supports this.
Document 2 Summary: Answer: 1956. Explanation: The document refers to Michael Irwin Jordan, an American scientist, born on February 25, 1956.
Document 3 Summary: Answer: Unknown. Explanation: The Skeptic found that the year 1998 cited by the Extractor contradicts other reliable sources, so this document's answer is unreliable.
All Correct Answers: ["1963", "1956"]. Explanation: Document 1 refers to the basketball player Michael Jeffrey Jordan (born 1963). Document 2 refers to the scientist Michael Irwin Jordan (born 1956). Document 3's answer was flagged as unreliable during verification.

Question: {query}
Document summaries:
{summaries_text}
"""
