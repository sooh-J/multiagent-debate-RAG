"""
LLM 호출 유틸리티

Ref: https://github.com/HanNight/RAMDocs/blob/main/run_madam_rag.py — call_llm()
원본은 HuggingFace transformers pipeline 사용, 여기서는 OpenAI API로 대체.

Provider 전환 (환경변수):
  LLM_PROVIDER=openai (기본) → OpenAI API
  LLM_PROVIDER=qwen          → 로컬 vLLM 서버 (OpenAI-compatible)
    LLM_BASE_URL=http://localhost:8000/v1 (기본)
    LLM_MODEL=Qwen/Qwen2.5-7B-Instruct (기본)
    LLM_MODEL은 vLLM 서버에 떠있는 모델 ID와 정확히 일치해야 함.
    다른 모델을 띄웠다면 LLM_MODEL 환경변수로 명시할 것.

vLLM 실행 예:
  vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --max-model-len 8192
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

load_dotenv()

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()

if LLM_PROVIDER == "qwen":
    DEFAULT_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    _base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
    _api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    client = OpenAI(base_url=_base_url, api_key=_api_key)
    async_client = AsyncOpenAI(base_url=_base_url, api_key=_api_key)
    # Qwen3 thinking 모드 OFF (4o-mini 동급 비교 목적)
    _EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
else:
    DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    _EXTRA_BODY = {}

# 가격 (USD per 1M tokens). 로컬 모델은 0.
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "Qwen/Qwen3-8B": {"input": 0.0, "output": 0.0},
    "Qwen/Qwen2.5-7B-Instruct": {"input": 0.0, "output": 0.0},
}

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
                extra_body=_EXTRA_BODY,
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


async def async_call_llm(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> str:
    response = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        extra_body=_EXTRA_BODY,
    )
    if response.usage:
        _usage["input_tokens"] += response.usage.prompt_tokens
        _usage["output_tokens"] += response.usage.completion_tokens
        _usage["calls"] += 1

    return response.choices[0].message.content.strip()


def get_usage_summary(model: str = DEFAULT_MODEL) -> dict:
    """현재까지 누적된 토큰 사용량과 예상 비용 반환"""
    prices = PRICING.get(model, {"input": 0.0, "output": 0.0})
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
