"""
Single LLM 프롬프트 정의

토론 없이 단일 LLM이 모든 문서를 한번에 보고 답변.
"""


def single_llm_prompt(query: str, documents: list[str]) -> str:
    docs_text = "\n\n".join(
        f"Document {i+1}: {doc}" for i, doc in enumerate(documents)
    )
    return f"""You are an expert in retrieval question answering.
You will be provided a question with multiple documents. Please answer the question based on the documents.
If there are multiple answers, please provide all possible correct answers and also provide a step-by-step reasoning explanation. If there is no correct answer, please reply 'unknown'.
Please follow the format: "All Correct Answers: []. Explanation: {{}}"

The following are examples:
Question: In which year was Michael Jordan born?
Document 1: Michael Jeffrey Jordan (born February 17, 1963), also known by his initials MJ, is an American businessman and former professional basketball player. He played 15 seasons in the National Basketball Association (NBA) between 1984 and 2003, winning six NBA championships with the Chicago Bulls. He was integral in popularizing basketball and the NBA around the world in the 1980s and 1990s, becoming a global cultural icon.
Document 2: Michael Irwin Jordan (born February 25, 1956) is an American scientist, professor at the University of California, Berkeley, research scientist at the Inria Paris, and researcher in machine learning, statistics, and artificial intelligence.
Document 3: Michael Jeffrey Jordan was born at Cumberland Hospital in Brooklyn, New York City, on February 17, 1998, to bank employee Deloris (nee Peoples) and equipment supervisor James R. Jordan Sr. He has two older brothers, James Jr. and Larry, as well as an older sister named Deloris and a younger sister named Roslyn. Jordan and his siblings were raised Methodist.
Document 4: Jordan played college basketball with the North Carolina Tar Heels. As a freshman, he was a member of the Tar Heels’ national championship team in 1982. Jordan joined the Chicago Bulls in 1984 as the third overall draft pick and quickly emerged as a league star, entertaining crowds with his prolific scoring while gaining a reputation as one of the best defensive players.
All Correct Answers: ["1963", "1956"]. Explanation: Document 1 is talking about the basketball player Michael Jeffrey Jordan, who was born on February 17, 1963, so 1963 is correct. Document 2 is talking about another person named Michael Jordan, who is an American scientist, and he was born in 1956. Therefore, the answer 1956 from Document 2 is also correct. Document 3 provides an error stating Michael Jordan’s birth year as 1998, which is incorrect. Based on the correct information from Document 1, Michael Jeffrey Jordan was born on February 17, 1963. Document 4 does not provide any useful information.

Question: {query}
{documents}
"""
#     return f"""You are an assistant answering a question based on the provided documents.

# Question: {query}

# {docs_text}

# Answer the question based on the documents above. If there are multiple correct answers, provide all of them.
# If there is no correct answer, please reply 'unknown'.
# Please follow the format: 'All Correct Answers: []. Explanation: {{}}.'

# The following are examples:
# Question: In which year was Michael Jordan born?
# Documents:
# Document 1: Michael Jeffrey Jordan was born on February 17, 1963.
# Document 2: Michael Irwin Jordan was born on February 25, 1956.
# Document 3: Michael Jeffrey Jordan was born on February 17, 1998.
# Document 4: The document does not include information about his birth year.
# All Correct Answers: ["1963", "1956"]. Explanation: Document 1 is talking about the basketball player Michael Jeffrey Jordan, born in 1963. Document 2 is talking about another person named Michael Jordan, an American scientist, born in 1956. Document 3 provides an incorrect year. Document 4 provides no useful information.

# Question: {query}

# {docs_text}
# """
