"""
MadamRAG 프롬프트 정의

Ref: https://github.com/HanNight/RAMDocs/blob/main/run_madam_rag.py
     — agent_response(), aggregate_responses() 내부 프롬프트
"""


# Ref: agent_response() — history 없을 때 프롬프트
def agent_initial_prompt(query: str, document: str) -> str:
    return f"""You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

Answer the question based only on this document. Provide your answer and a step-by-step reasoning explanation.
Please follow the format: 'Answer: {{}}. Explanation: {{}}.''"""


# Ref: agent_response() — history 있을 때 프롬프트
def agent_debate_prompt(query: str, document: str, history: str) -> str:
    return f"""You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

The following responses are from other agents as additional information.
{history}

Answer the question based on the document and other agents' responses.
Please follow the format: 'Answer: {{}}. Explanation: {{}}.''"""


# Ref: aggregate_responses() 내부 프롬프트
def aggregator_prompt(query: str, agent_responses: list[str]) -> str:
    responses_text = "\n\n".join(
        f"Agent {i+1}: {r}" for i, r in enumerate(agent_responses)
    )
    return f"""You are an aggregator synthesizing multiple agents' answers.

If there are multiple answers, please provide all possible correct answers and also provide a step-by-step reasoning explanation. If there is no correct answer, please reply 'unknown'.
Please follow the format: 'All Correct Answers: []. Explanation: {{}}.'

The following are examples:
Question: In which year was Michael Jordan born?
Agent responses:
Agent 1: Answer: 1963. Explanation: The document clearly states that Michael Jeffrey Jordan was born on February 17, 1963.
Agent 2: Answer: 1956. Explanation: The document states that Michael Irwin Jordan was born on February 25, 1956.
Agent 3: Answer: 1998. Explanation: According to the document, Michael Jeffrey Jordan was born on February 17, 1998.
Agent 4: Answer: Unknown. Explanation: The document does not include information about his birth year.
All Correct Answers: ["1963", "1956"]. Explanation: Agent 1 is talking about the basketball player Michael Jeffrey Jordan, born in 1963. Agent 2 is talking about another person named Michael Jordan, an American scientist, born in 1956. Agent 3 provides an incorrect year. Agent 4 provides no useful information.

Question: {query}
Agent responses:
{responses_text}
"""
