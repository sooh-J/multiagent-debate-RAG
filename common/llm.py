"""
LLM 호출 유틸리티

Ref: https://github.com/HanNight/RAMDocs/blob/main/run_madam_rag.py — call_llm()
원본은 HuggingFace transformers pipeline 사용, 여기서는 OpenAI-compatible API로 대체.

엔드포인트는 두 가지 모드를 지원한다:
  1. OpenAI (default)         — OPENAI_API_KEY 만 설정
  2. vLLM OpenAI-compatible   — OPENAI_BASE_URL 도 함께 설정 (예: http://localhost:8000/v1)
                                 vLLM 서버에 직접 붙으려면 OPENAI_API_KEY 는 임의 dummy 값이면 됨.

backward compat:
  - LLM_BASE_URL (PR #10 도입) 도 OPENAI_BASE_URL 의 alias 로 인식
  - LLM_PROVIDER=qwen 일 때 default base_url/model 자동 설정

Qwen3 모델은 chat_template_kwargs.enable_thinking=False 로 thinking 모드를 끔
(gpt-4o-mini / LLAMA-Instruct 와 동급 비교 위해). 모델 이름에 'qwen3' 가 들어가면 자동 적용.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

load_dotenv()

# Provider 분기 (PR #10 호환). LLM_PROVIDER=qwen 이면 vLLM endpoint default 적용.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()

if LLM_PROVIDER == "qwen":
    DEFAULT_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    _BASE_URL = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or "http://localhost:8000/v1"
    )
else:
    DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    _BASE_URL = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")

_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")  # vLLM 은 아무 값이나 OK

client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL) if _BASE_URL \
    else OpenAI(api_key=_API_KEY)
async_client = AsyncOpenAI(api_key=_API_KEY, base_url=_BASE_URL) if _BASE_URL \
    else AsyncOpenAI(api_key=_API_KEY)

# 가격 (USD per 1M tokens). 로컬(vLLM) 모델은 0으로 처리.
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
    # 로컬 vLLM 모델 (cost 0)
    "Qwen/Qwen2.5-7B-Instruct": {"input": 0.0, "output": 0.0},
    "Qwen/Qwen3-8B":            {"input": 0.0, "output": 0.0},
}

# 토큰 사용량 누적
_usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


MAX_RETRIES = 3


def _resolve_model(model: str | None) -> str:
    """None 이면 module-level DEFAULT_MODEL 을 동적으로 반환 (set_default_model 반영용)."""
    return model if model is not None else DEFAULT_MODEL


def _extra_body_for(model: str) -> dict:
    """모델별 vLLM extra_body. Qwen3 의 thinking 모드는 동급 비교를 위해 끔."""
    if "qwen3" in model.lower():
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


def call_llm(prompt: str, model: str | None = None, temperature: float = 0.0) -> str:
    m = _resolve_model(model)
    extra = _extra_body_for(m)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **({"extra_body": extra} if extra else {}),
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


def call_llm_batch(prompts: list[str], model: str | None = None, temperature: float = 0.0) -> list[str]:
    """여러 프롬프트를 동시에 호출"""
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        return list(executor.map(lambda p: call_llm(p, model, temperature), prompts))


async def async_call_llm(prompt: str, model: str | None = None, temperature: float = 0.0) -> str:
    m = _resolve_model(model)
    extra = _extra_body_for(m)
    response = await async_client.chat.completions.create(
        model=m,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        **({"extra_body": extra} if extra else {}),
    )
    if response.usage:
        _usage["input_tokens"] += response.usage.prompt_tokens
        _usage["output_tokens"] += response.usage.completion_tokens
        _usage["calls"] += 1

    return response.choices[0].message.content.strip()


def set_default_model(model: str) -> None:
    """전역 default model 갱신. run_*.py 진입점에서 한 번만 호출."""
    global DEFAULT_MODEL
    DEFAULT_MODEL = model


def model_slug(model: str) -> str:
    """파일명용 short slug. 'meta-llama/Llama-3.1-8B-Instruct' → 'llama-3.1-8b-instruct'"""
    return model.split("/")[-1].lower()


def get_usage_summary(model: str = None) -> dict:
    """현재까지 누적된 토큰 사용량과 예상 비용 반환. 로컬(vLLM) 모델은 cost=0."""
    m = model or DEFAULT_MODEL
    prices = PRICING.get(m, {"input": 0.0, "output": 0.0})
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


def print_usage_summary(model: str = None):
    """토큰 사용량과 비용을 출력"""
    m = model or DEFAULT_MODEL
    s = get_usage_summary(m)
    print(f"\n{'='*50}")
    print("API USAGE SUMMARY")
    print(f"{'='*50}")
    print(f"  Model        : {m}")
    print(f"  Endpoint     : {_BASE_URL or 'OpenAI'}")
    print(f"  Calls        : {s['calls']}")
    print(f"  Input tokens : {s['input_tokens']:,}")
    print(f"  Output tokens: {s['output_tokens']:,}")
    print(f"  Total tokens : {s['total_tokens']:,}")
    print(f"  Cost         : ${s['total_cost_usd']:.4f}")
    print(f"{'='*50}")
