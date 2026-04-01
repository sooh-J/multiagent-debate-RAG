"""텍스트 정규화 및 LLM 출력 파싱 유틸리티"""

import re
import string


# Ref: https://github.com/HanNight/RAMDocs/blob/main/run_madam_rag.py — normalize_answer()
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_answer(text: str) -> str:
    """'Answer: {}. Explanation: {}.' 형식에서 answer 부분만 추출 (early stopping 비교용)"""
    match = re.search(r"Answer:\s*(.*?)\.", text)
    return match.group(1).strip() if match else text


def parse_answers(text: str) -> list[str]:
    """
    'All Correct Answers: ["answer1", "answer2", ...]' 에서 정답 리스트 추출
    Aggregator 출력에서 정답 리스트를 파싱
    """
    match = re.search(r"All Correct Answers:\s*\[([^\]]*)\]", text, re.IGNORECASE)
    if not match:
        return []

    raw = match.group(1).strip()
    if not raw:
        return []

    quoted = re.findall(r'"([^"]+)"', raw)
    if quoted:
        return [a.strip() for a in quoted]

    return [
        a.strip().strip("'").strip()
        for a in raw.split(",")
        if a.strip().strip("'").strip()
    ]


def parse_explanation(text: str) -> str:
    """Explanation: {} 부분 추출"""
    match = re.search(r"Explanation:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""
