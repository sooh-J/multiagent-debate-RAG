"""
V4 실행 스크립트

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

V4 방법론: 찬/반/중재자 (Round 1만) → 이후 MadamRAG 방식
  - Round 1: 문서별 Pro(찬성), Con(반대), Mediator(중재자)로 신뢰도 검증
  - Round 2+: 원본 MadamRAG 방식으로 전환 (문서 대변인 1명 + 타 에이전트 응답 참고)
  - Round 1 중재자 결과가 Round 2의 "이전 라운드 에이전트 답변"으로 자연스럽게 이어짐
  - 수렴 시 조기 종료 (최대 3라운드)

V3와의 차이:
  V3는 매 라운드 찬/반/중재자를 반복하지만,
  V4는 첫 라운드에서만 찬/반/중재자로 필터링한 뒤 MadamRAG 토론으로 넘어간다.
  → V3 대비 LLM 호출 수가 적고, 라운드 구조 변경의 효과를 ablation으로 비교 가능

Async:
  - 문서 간 / 에이전트 간 API 호출을 병렬로 실행하여 속도 향상

데이터셋 (--dataset):
  - ramdocs           : --n 생략 시 data/ramdocs/full.json (500개 전체)
  - raguard           : --n 생략 시 raguard_preprocessed.json (711개 전체)
  - raguard_balanced  : --n 생략 시 raguard_preprocessed_balanced.json (230개 전체)

기존 결과 파일이 있으면 그 다음 샘플부터 이어서 실행 (resume 내장).
50개마다 중간 저장.

실행 방법:
    conda activate nlp
    python run_v4.py                                  # default: ramdocs 전체 500개
    python run_v4.py --n 20                           # ramdocs 처음 20개
    python run_v4.py --dataset raguard_balanced       # raguard_balanced 전체 230개
    python run_v4.py --dataset raguard_balanced --n 20
    python run_v4.py --model llama-3.1-8b-instruct    # vLLM served model (OPENAI_BASE_URL 필요)

출력 (suffix = "full" if --n 생략 else f"n{N}", default 모델일 땐 slug 미포함):
  - 콘솔: 각 샘플별 예측 결과 및 메트릭 (EM, Precision, Recall, F1)
  - 로그: logs/v4_<dataset>_<suffix>[_<slug>]_YYYYMMDD_HHMM.log
  - 결과: results/v4_<dataset>_<suffix>[_<slug>]_results.json
  예: results/v4_ramdocs_full_results.json, results/v4_raguard_balanced_n20_llama-3.1-8b-instruct_results.json

파이프라인 코드: pipelines/v4.py
프롬프트 정의:  prompts/v3.py (V3와 공유), prompts/madamrag.py (Round 2+)
설정:          configs/v3.py (V3와 공유)
"""

import argparse
import asyncio
import json
import os
import sys

from common.logging import Tee
from common.llm import DEFAULT_MODEL, model_slug, print_usage_summary, set_default_model
from common.metrics import compute_metrics, print_results_table
from data.ramdocs.download import load_ramdocs
from data.raguard.loader import load_raguard
from pipelines.v4 import v4_method


def _load_ramdocs(n: int | None):
    """n=None 이면 full.json (전체 500개), 아니면 sample 사용."""
    if n is None:
        with open("data/ramdocs/full.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return load_ramdocs(n_samples=n)


DATASET_LOADERS = {
    "ramdocs":          lambda n: _load_ramdocs(n),
    "raguard":          lambda n: load_raguard(n_samples=n, balanced=False),
    "raguard_balanced": lambda n: load_raguard(n_samples=n, balanced=True),
}

CHECKPOINT_EVERY = 50


async def run_on_sample(sample: dict, dataset: str) -> dict:
    query = sample["question"]
    doc_texts = [doc["text"] for doc in sample["documents"]]
    doc_meta = [{"type": doc["type"], "answer": doc["answer"]} for doc in sample["documents"]]

    result = await v4_method(query, doc_texts, dataset=dataset)

    predicted_answers = result["final_answer"] if result["final_answer"] else []
    metrics = compute_metrics(predicted_answers, sample["gold_answers"], sample["wrong_answers"])

    return {
        "question": query,
        "disambig_entity": sample["disambig_entity"],
        "gold_answers": sample["gold_answers"],
        "wrong_answers": sample["wrong_answers"],
        "doc_meta": doc_meta,
        "predicted": predicted_answers,
        "explanation": result["final_explanation"],
        "rounds_run": result["rounds_run"],
        "round_history": result["round_history"],
        **metrics,
    }


def _error_placeholder(sample: dict, exc: Exception) -> dict:
    """LLM 호출 실패 시 schema 유지하면서 EM=0으로 기록 (context overflow 등 outlier 대응)"""
    return {
        "question": sample["question"],
        "disambig_entity": sample["disambig_entity"],
        "gold_answers": sample["gold_answers"],
        "wrong_answers": sample["wrong_answers"],
        "doc_meta": [{"type": d["type"], "answer": d["answer"]} for d in sample["documents"]],
        "predicted": [],
        "explanation": "",
        "rounds_run": 0,
        "round_history": [],
        "error": f"{type(exc).__name__}: {exc}",
        **compute_metrics([], sample["gold_answers"], sample["wrong_answers"]),
    }


async def run_on_dataset(ds_sample, existing_results: list, output_path: str, dataset: str) -> list[dict]:
    results = list(existing_results)
    start = len(results)
    total = len(ds_sample)

    if start > 0:
        print(f"이어서 실행: {start}개 완료, {total - start}개 남음")
    else:
        print(f"처음부터 실행: 총 {total}개")

    for i in range(start, total):
        sample = ds_sample[i]
        print(f"\n[{i+1}/{total}] Q: {sample['question']}")
        try:
            out = await run_on_sample(sample, dataset)
        except Exception as e:
            print(f"  !! Sample failed: {type(e).__name__}: {e}")
            out = _error_placeholder(sample, e)
        print(f"  Gold:      {out['gold_answers']}")
        print(f"  Predicted: {out['predicted']}")
        print(f"  EM={out['em']}  P={out['precision']}  R={out['recall']}  F1={out['f1']}")
        results.append(out)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [중간 저장] {i+1}개 결과 → {output_path}")

    n = len(results)
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"  Exact Match : {sum(r['em'] for r in results) / n * 100:.2f}%")
    print(f"  Precision   : {sum(r['precision'] for r in results) / n:.4f}")
    print(f"  Recall      : {sum(r['recall'] for r in results) / n:.4f}")
    print(f"  F1          : {sum(r['f1'] for r in results) / n:.4f}")

    print_results_table(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과가 '{output_path}'에 저장되었습니다.")

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASET_LOADERS), default="ramdocs")
    p.add_argument("--n", type=int, default=None,
                   help="평가 샘플 개수 (생략 시 데이터셋 전체)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="LLM 식별자 (OpenAI 모델 ID 또는 vLLM served name). default 면 출력 파일 이름에 model slug 미포함")
    return p.parse_args()


async def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    set_default_model(args.model)

    suffix = "full" if args.n is None else f"n{args.n}"
    tag = f"{args.dataset}_{suffix}" if args.model == DEFAULT_MODEL \
        else f"{args.dataset}_{suffix}_{model_slug(args.model)}"

    tee = Tee(prefix=f"v4_{tag}")
    sys.stdout = tee

    try:
        ds_sample = DATASET_LOADERS[args.dataset](args.n)
        print(f"{args.dataset} 데이터 로드: {len(ds_sample)}개")

        output_path = f"results/v4_{tag}_results.json"
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            print(f"기존 결과 로드: {len(existing)}개")
        else:
            existing = []

        if len(existing) >= len(ds_sample):
            print("이미 전체 완료 상태입니다. 종료.")
        else:
            await run_on_dataset(ds_sample, existing, output_path, args.dataset)
            print_usage_summary()
    finally:
        tee.close()
        print(f"로그 저장: {tee.filepath}")


if __name__ == "__main__":
    asyncio.run(main())
