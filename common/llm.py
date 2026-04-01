"""
LLM 호출 유틸리티

Ref: https://github.com/HanNight/RAMDocs/blob/main/run_madam_rag.py — call_llm()
원본은 HuggingFace transformers pipeline 사용, 여기서는 OpenAI API로 대체.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def call_llm(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()
