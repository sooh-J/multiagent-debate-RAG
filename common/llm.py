"""
LLM 호출 유틸리티

Ref: https://github.com/HanNight/RAMDocs/blob/main/run_madam_rag.py — call_llm()
원본은 HuggingFace transformers pipeline 사용, 여기서는 OpenAI API로 대체.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"

# gpt-4o-mini 가격 (USD per 1M tokens, 2025.05 기준)
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

# 토큰 사용량 누적
_usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


MAX_RETRIES = 3

def call_llm(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            if response.usage:
                _usage["input_tokens"] += response.usage.prompt_tokens
                _usage["output_tokens"] += response.usage.completion_tokens
                _usage["calls"] += 1

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLM] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 ** attempt)


def call_llm_batch(prompts: list[str], model: str = DEFAULT_MODEL, temperature: float = 0.0) -> list[str]:
    """여러 프롬프트를 동시에 호출"""
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        return list(executor.map(lambda p: call_llm(p, model, temperature), prompts))


def get_usage_summary(model: str = DEFAULT_MODEL) -> dict:
    """현재까지 누적된 토큰 사용량과 예상 비용 반환"""
    prices = PRICING.get(model, PRICING["gpt-4o-mini"])
    input_cost = _usage["input_tokens"] / 1_000_000 * prices["input"]
    output_cost = _usage["output_tokens"] / 1_000_000 * prices["output"]
    total_cost = input_cost + output_cost

    return {
        "calls": _usage["calls"],
        "input_tokens": _usage["input_tokens"],
        "output_tokens": _usage["output_tokens"],
        "total_tokens": _usage["input_tokens"] + _usage["output_tokens"],
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(total_cost, 6),
    }


def print_usage_summary(model: str = DEFAULT_MODEL):
    """토큰 사용량과 비용을 출력"""
    s = get_usage_summary(model)
    print(f"\n{'='*50}")
    print("API USAGE SUMMARY")
    print(f"{'='*50}")
    print(f"  Calls        : {s['calls']}")
    print(f"  Input tokens : {s['input_tokens']:,}")
    print(f"  Output tokens: {s['output_tokens']:,}")
    print(f"  Total tokens : {s['total_tokens']:,}")
    print(f"  Cost         : ${s['total_cost_usd']:.4f}")
    print(f"{'='*50}")
