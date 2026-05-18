# RAGuard v2 — Prompt Ablation 실험

`prompts/raguard.py` (v1) 의 ablation variant. RAGuard 에서 multi-agent debate 가 single_llm 에 지는 failure mode 를 prompt 수준에서 보완하는 실험.

## 동기

v1 결과 (raguard_balanced 230, 3-model 평가, main README 참고):

| Model | single_llm | madamrag | v4 |
|---|---:|---:|---:|
| GPT-4o-mini | **73.9%** | 65.7% | 64.3% |
| LLAMA-3.1-8B | **67.8%** | 57.0% | 58.7% |
| Qwen2.5-7B | 53.5% | 53.9% | 54.3% |

3 모델 모두 multi-agent < single_llm (-8 ~ -11pp). Per-cell 분석에서 v4 의 가장 큰 약점은 **(gold=True + misinfo doc 존재)** sub-condition (v4 True acc 25.8% vs single_llm 54.8%). Pro voice 가 misinfo doc 도 legitimate evidence 처럼 다뤄 verdict 를 stretch 하기 때문으로 추정.

## v1 → v2 변경 — "RAMDocs-aligned minimal delta"

설계 원칙: RAMDocs (prompts/madamrag.py, prompts/v3.py) prompt 가 표준. RAGuard 데이터셋 특성상 어쩔 수 없는 차이만 적용하고, 나머지는 RAMDocs 구조로 회귀.

### 95개 wrong 케이스 (LLAMA v4 v1) 정량 분석 결과

| Failure mode | 비율 | RAGuard v1 의 어떤 추가가 원인 |
|---|---:|---|
| Mode A: Aggregator 가 Mediator 과반 override | **39%** | Aggregator 의 "Some agents may be misled" 라이센스 |
| Mode C: correct doc 에서 gold verdict 한 번도 안 나옴 | 33% | Mediator 의 "Be willing to answer Unknown FREELY" |
| Mode D: Mediator 출력 과반이 Unknown | 31% | 같은 이유 |
| Mode B: noise doc 에서 Pro 가 verdict 환각 | 20% | RAGuard 구조적 한계 (RAMDocs 도 같음) |

→ A+C+D = 70% 가 RAGuard v1 의 noise 방어용 추가 문구 때문에 발생. RAMDocs 로 회귀하면 개선 기대.

### 유지 (RAGuard 적응, 필수)
1. **출력 format** — `["True"]` / `["False"]` single binary (multi-answer 아님)
2. **doc-only 제약** — `_DOC_ONLY` 블록 (prior knowledge 사용 금지, 모델 간 공정 비교)
3. **Balanced few-shot** — Country X (gold=False) + Country Y (gold=True) 양쪽 노출 (v1 은 False 만)
4. **Entity disambiguation 문구 제거** — RAGuard claim 에는 동명이인 같은 ambiguity 없음

### 롤백 (RAMDocs 톤으로 회귀)
1. **Aggregator** — "Some agents may be misled by misleading documents" 문구 **삭제** → Mediator 결과 존중
2. **Aggregator with confidence** — 같은 "Some mediators may be misled" 삭제
3. **Mediator** — "Be willing to answer Unknown FREELY" 삭제 → Unknown 은 evidence 없을 때만
4. **Pro/Con** — 강한 `CRITICAL: No supporting evidence found` 블록 **삭제 / 약화** → 자연스러운 1줄 escape 만 (Pro 의 마지막에 "if you cannot find evidence, briefly state so")

## 실행

기존 코드 그대로, `--dataset` 만 `_v2` suffix 붙이면 됨 (raguard_v2 / raguard_balanced_v2). 같은 데이터를 다른 prompt 분기로 라우팅:

```bash
# GPT-4o-mini (default)
python run_v4.py --dataset raguard_balanced_v2

# LLAMA via vLLM
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
  python run_v4.py --dataset raguard_balanced_v2 --model llama-3.1-8b-instruct

# Qwen via vLLM
OPENAI_BASE_URL=http://localhost:8003/v1 OPENAI_API_KEY=EMPTY \
  python run_v4.py --dataset raguard_balanced_v2 --model qwen-7b-instruct
```

3 method (`single_llm` / `madamrag` / `v4`) 모두 동일.

## 결과 파일 패턴

```
results/single_llm_raguard_balanced_v2_full_results.json                                # GPT-4o-mini
results/madamrag_raguard_balanced_v2_full_llama-3.1-8b-instruct_results.json            # LLAMA
results/v4_raguard_balanced_v2_full_qwen-7b-instruct_results.json                       # Qwen
```

v1 (`raguard_balanced_full*`) 결과 파일과 충돌 없음 → 같은 분석 코드로 직접 비교 가능.

## 비교 분석 방법

핵심 측정 포인트:

1. **(gold=True + misinfo) cell 의 True acc 개선 여부** — v2 가 노린 가장 큰 fix. v4 의 25.8% (v1) 가 얼마나 회복되는지가 1차 지표.
2. **Class bias 변화** — v1 에서 v4 의 False bias 가 v2 에서 완화되는지 (T-acc 와 F-acc 격차)
3. **Method 간 ordering** — single_llm < madamrag < v4 가 v2 에서 회복되는지

비교 코드는 [3-model 분석 섹션] 의 per-bucket breakdown 그대로 재사용 가능. v1/v2 결과 dict 만 양쪽 로드.

## 후속 실험 (TODO)

### Experiment 3 — Mediator confidence 를 logit 기반 p(True) 로 대체 (v4 Round 1)

**문제 의식**: 현재 Mediator 의 "Confidence: High/Medium/Low" 는 self-rating discrete label. logprob 기반 확률을 쓰면 model 의 실제 belief 를 정량적으로 반영해서 Aggregator 가 weighted aggregation 가능.

**구현 방향**:
- vLLM/OpenAI 둘 다 `logprobs=True` 지원 (vLLM 은 거의 무료 — 이미 계산됨)
- Mediator 응답 후 calibration query 1번 더 호출:
  ```
  "Given the Pro/Con arguments above for this single document, output ONLY one token: True or False."
  ```
  → 그 토큰의 logprob 에서 `p(True)`, `p(False)` 추출
- `common/llm.py` 에 `call_llm_with_logprobs(prompt, target_tokens=["True","False"]) -> (text, {token: prob})` 추가
- `pipelines/v4.py` 의 mediator step 에서 calibration query 결과를 함께 저장:
  - 기존 필드: `mediator_response` (label-based)
  - 신규 필드: `mediator_p_true` (continuous, 0~1)
- 새 Aggregator variant: `aggregator_with_prob_prompt_raguard` — weighted vote using `p_true`
- ablation: label-based vs prob-based aggregation 비교

**보존 원칙**: 기존 v4 pipeline / Aggregator 그대로 두고 새 함수/variant 로 추가. `--dataset raguard_balanced_v3` 같은 alias 로 라우팅.

**예상 작업 시간**: 반나절 ~ 하루.

**측정 포인트**:
- prob-based aggregation EM vs label-based EM
- Calibration: predicted `p_true` 분포 vs 실제 정답 분포 (ECE 등)
- LLAMA vs Qwen 의 calibration quality 차이 (Qwen 은 "default-False" 강해서 p_true 가 0 쪽 몰릴 가능성)

### 추가 아이디어 (낮은 우선순위)

- **Retrieval filter preprocessing** (relevance top-k) — RAGuard 의 52% noise doc 사전 제거 후 모든 method 에 동일 적용. multi-agent 가 의미 있는 doc 만 보게 해서 debate 효용 증가 가능성
- **Confidence-weighted self-consistency** (다중 sampling + voting) — 같은 prompt 를 여러 번 sample 한 후 majority vote / weighted by logprob
