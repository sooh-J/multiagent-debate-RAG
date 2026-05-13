JUDGE_PROMPT = """You are a strict answer equivalence judge.

Given a question and two answers, determine if Answer A and Answer B refer to exactly the same entity, value, or fact.

Rules:
- Surface form differences are acceptable (e.g., formatting, units, abbreviations)
- Both answers must refer to the same specific entity — partial matches or broader/narrower answers are NOT equivalent
- When in doubt, answer "No"

Question: {question}
Answer A: {gold}
Answer B: {predicted}

Verdict (Yes or No):"""


def judge_prompt(question: str, gold: str, predicted: str) -> str:
    return JUDGE_PROMPT.format(question=question, gold=gold, predicted=predicted)
