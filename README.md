# Multi-Agent Debate RAG

RAG 환경에서 충돌하는 정보(ambiguity, misinformation)를 다루기 위한 Multi-Agent Debate 실험 프로젝트.

## 방법론 요약

| 방법론 | 설명 | 에이전트 구조 | 라운드 |
|--------|------|---------------|--------|
| **Single LLM** (Baseline) | 토론 없이 모든 문서를 한 프롬프트로 전달 | 단일 LLM | 1 |
| **MadamRAG** (Baseline) | 문서 대변인 간 토론 | 문서당 Agent 1명 + Aggregator | 최대 3 (수렴 시 조기 종료) |
| **Proposed Method (V2)** | 문서 내부 검증 후 집계 | Extractor → Skeptic → Resolver + Global Aggregator | 1 (토론 없이 1회) |
| **V3** | 찬/반/중재자 매 라운드 반복 | Pro → Con → Mediator + Aggregator | 최대 3 (수렴 시 조기 종료) |
| **V4** | V3 (Round 1) + MadamRAG (Round 2+) | Round 1: Pro/Con/Mediator, Round 2+: MadamRAG Agent | 최대 3 (수렴 시 조기 종료) |

## 결과 (RAMDocs full, 500 samples)

| 방법론 | LLM | EM | Precision | Recall | F1 |
|--------|-----|----|-----------|--------|-----|
| MadamRAG | Qwen2.5-7B-Instruct | 18.20% | 0.6457 | 0.4683 | 0.5112 |
| V4       | gpt-4o-mini         | 29.20% | 0.6701 | 0.6783 | 0.6464 |
| V4       | Qwen2.5-7B-Instruct | 20.20% | 0.6700 | 0.5117 | 0.5494 |

결과 파일 패턴: `results/<method>_<dataset>_<suffix>[_<model-slug>]_results.json`
- gpt-4o-mini (default) → slug 미포함 (예: `madamrag_ramdocs_full_results.json`)
- 그 외 모델 → slug 포함 (예: `v4_ramdocs_full_llama-3.1-8b-instruct_results.json`)
- 구버전 `_qwen` 접미사 파일은 PR #10 이전 형식 (호환을 위해 그대로 둠).

## Baseline: MadamRAG

[MADAM-RAG](https://arxiv.org/abs/2504.13079) (Multi-Agent Debate for Ambiguity and Misinformation in RAG)를 baseline으로 사용한다.

- 원본 구현: https://github.com/HanNight/RAMDocs/blob/main/run_madam_rag.py
- 데이터셋: [HanNight/RAMDocs](https://huggingface.co/datasets/HanNight/RAMDocs) (500개)

### 원본 대비 변경 사항

| 항목 | 원본 (RAMDocs) | 본 프로젝트 |
|------|----------------|-------------|
| LLM | HuggingFace transformers pipeline (로컬 모델) | OpenAI API (`gpt-4o-mini`) |
| Aggregator 출력 파싱 | 별도 파싱 없음 | `parse_answers()`, `parse_explanation()` 추가 |
| 평가 지표 | 없음 | EM, Precision, Recall, F1 + 노이즈 오염률 |
| Early stopping | 전체 응답 문자열 비교 | Answer 부분만 추출하여 비교 (`extract_answer()`) |

### 프로젝트 구조

```
├── common/                     # 공통 유틸리티 (방법론 간 공유)
│   ├── llm.py                  # OpenAI API 호출 (retry, 병렬 batch 지원)
│   ├── parsing.py              # 텍스트 정규화 및 파싱
│   ├── metrics.py              # 평가 지표 (EM, F1 등)
│   └── logging.py              # 로그 출력 (Tee)
├── prompts/                    # 방법론별 프롬프트
│   ├── madamrag.py             # MadamRAG 프롬프트
│   ├── single_llm.py           # Single LLM 프롬프트
│   ├── proposed_method.py      # Proposed Method (V2) 프롬프트 — Extractor/Skeptic/Resolver
│   ├── v3.py                   # V3/V4 공통 프롬프트 — Pro/Con/Mediator
│   ├── raguard.py              # RAGuard 전용 변형 프롬프트 — binary 답변 + doc-only 제약
│   └── llm_judge.py            # LLM-as-a-Judge 채점 프롬프트
├── pipelines/                  # 방법론별 파이프라인
│   ├── madamrag.py             # MadamRAG 토론 파이프라인 (에이전트 병렬 호출)
│   ├── single_llm.py           # Single LLM 파이프라인
│   ├── proposed_method.py      # Proposed Method (V2) 파이프라인 — 문서 내부 토론 + 글로벌 집계
│   ├── v3.py                   # V3 파이프라인 — 찬/반/중재자 매 라운드 반복
│   └── v4.py                   # V4 파이프라인 — V3(Round 1) + MadamRAG(Round 2+)
├── configs/                    # 방법론별 설정
│   ├── madamrag.py             # MAX_ROUNDS 등
│   ├── single_llm.py           # Single LLM 설정
│   ├── proposed_method.py      # Proposed Method 설정
│   └── v3.py                   # V3/V4 공통 설정 (MAX_ROUNDS = 3)
├── data/
│   ├── ramdocs/                # RAMDocs 데이터셋 (다운로드 + 샘플)
│   ├── faitheval/              # FaithEval 데이터셋 (다운로드 + 샘플)
│   └── raguard/                # RAGuard 데이터셋 — download.py + preprocess.py + loader.py, data/raguard/README.md 참고
├── results/                    # 실험 결과 저장
├── run_single_llm.py           # Single LLM 실행
├── run_madamrag.py             # MadamRAG 실행
├── run_proposed_method.py      # Proposed Method (V2) 실행
├── run_v3.py                   # V3 실행
├── run_v4.py                   # V4 실행
├── eval_llm_judge.py           # LLM-as-a-Judge 재채점
└── original.py                 # 원본 단일 파일 (참고용)
```

### 새로운 방법론 추가 방법

1. `prompts/<method_name>.py` — 프롬프트 정의
2. `pipelines/<method_name>.py` — 파이프라인 구현 (`common/` 유틸 활용)
3. `configs/<method_name>.py` — 설정값
4. `run_<method_name>.py` — 실행 스크립트 작성

## 실행

```bash
conda activate nlp

# 데이터 다운로드 (최초 1회)
python -m common.data
python -m data.raguard.download         # RAGuard 사용 시
python -m data.raguard.preprocess       # RAMDocs 스키마로 변환 + balanced 버전 생성

# 파이프라인 실행 — default: ramdocs 전체 (500개), --n 으로 sample 가능
python run_single_llm.py                                    # Single LLM baseline
python run_madamrag.py                                      # MadamRAG baseline
python run_proposed_method.py                               # Proposed Method (V2)
python run_v3.py                                            # V3
python run_v4.py                                            # V4

# 데이터셋 전환 (single_llm / madamrag / v4 한정 — --dataset 인자 지원)
python run_v4.py --dataset raguard_balanced                 # raguard_balanced 전체 (230개)
python run_v4.py --dataset raguard --n 50                   # raguard 전처리본에서 50개
python run_madamrag.py --dataset raguard_balanced --n 20    # sample 평가

# 출력 파일 패턴: results/<method>_<dataset>_<suffix>_results.json
#   suffix = "full" (--n 생략) 또는 f"n{N}"

# LLM-as-a-Judge 재채점
python eval_llm_judge.py results/madamrag_results.json
python eval_llm_judge.py results/single_llm_results.json
```

### Open-source 모델 (vLLM)으로 실행

OpenAI-compatible API를 제공하는 vLLM 서버를 띄우면 동일한 실행 스크립트로 LLAMA / Qwen 등 어떤 instruct 모델도 평가 가능. **권장 인터페이스: `OPENAI_BASE_URL` + `--model` 인자** (모델 무관, 출력 파일에 slug 자동 부착).

1. vLLM 서버 (별도 터미널, GPU 1장당 1 instance)
   ```bash
   # LLAMA-3.1-8B-Instruct (예시, HF gating 통과 필요)
   CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
     --model meta-llama/Llama-3.1-8B-Instruct \
     --served-model-name llama-3.1-8b-instruct \
     --port 8000 \
     --max-model-len 16384 \
     --max-num-seqs 8 \
     --gpu-memory-utilization 0.92

   # 또는 로컬 path Qwen
   CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
     --model /data_seoul/models/Qwen2.5-7B-Instruct \
     --served-model-name qwen-7b-instruct \
     --port 8001 \
     --max-model-len 16384 \
     --max-num-seqs 8 --gpu-memory-utilization 0.92
   ```

2. 환경 변수로 endpoint 지정 (클라이언트 셸):
   ```bash
   export OPENAI_BASE_URL=http://localhost:8000/v1
   export OPENAI_API_KEY=EMPTY      # vLLM 은 인증 안 함, 아무 문자열이나 OK
   ```

3. `--model` 인자로 served name 명시 (결과 파일에 자동으로 slug 붙음):
   ```bash
   python run_single_llm.py --dataset raguard_balanced --model llama-3.1-8b-instruct
   python run_madamrag.py   --dataset raguard_balanced --model llama-3.1-8b-instruct
   python run_v4.py         --dataset raguard_balanced --model llama-3.1-8b-instruct
   ```

> Qwen3 모델은 `common/llm.py`가 자동으로 `enable_thinking=False` 를 보내 thinking 모드를 끔 (다른 모델과 동급 비교 목적).
>
> **Backward compat (PR #10)**: `LLM_PROVIDER=qwen` 환경변수도 여전히 동작 (default base_url/model 자동 설정). 새 코드는 `OPENAI_BASE_URL` + `--model` 쪽 권장.
>
> 자세한 환경 setup / troubleshooting / 4-GPU 병렬 실행은 [Reproduction](#reproduction--새-모델로-평가-돌리기) 참조.

중간에 끊겨도 결과 JSON이 있으면 이어서 재개된다 (50개마다 체크포인트 저장).

## 진행 상황

### 데이터셋 확장

- **RAGuard 추가** (`data/raguard/`)
  - 출처: [UCSC-IRKM/RAGuard](https://huggingface.co/datasets/UCSC-IRKM/RAGuard) — 정치 발언 fact-checking
  - 원본: claim 2,648 / document 16,331
  - **전처리 완료** (RAMDocs 스키마로 통일):
    - `raguard_preprocessed.json`: claim 711 / document 4,593 (verdict True 115 : False 596, unbalanced)
    - `raguard_preprocessed_balanced.json`: claim 230 / document 1,495 (True 115 : False 115, stratified by original_verdict)
  - 컬럼 정의·전처리 규칙 전체는 [data/raguard/README.md](data/raguard/README.md) 참고
- **미해결 / 결정 필요**
  - balanced(230) vs unbalanced full(711) 중 어느 쪽으로 평가할지. unbalanced로 가면 F1·balanced accuracy 필수.
  - 본문 없는 doc 약 25%를 일단 폐기하고 있음. 필요 시 Reddit `.json` 크롤링으로 보강 가능.

### RAGuard 평가 셋업

- **`--dataset` 인자 추가**: `run_single_llm.py` / `run_madamrag.py` / `run_v4.py` 모두 `--dataset {ramdocs, raguard, raguard_balanced}` 지원. default는 `ramdocs`라 기존 동작 보존.
- **RAGuard 전용 프롬프트 분기** (`prompts/raguard.py`): RAMDocs(open-domain QA + 모호성)와 RAGuard(single binary fact-check)의 task 특성이 달라, dataset 인자에 따라 pipeline에서 프롬프트를 분기. 총 8개 RAGuard 변형 함수.

  | 단계 | RAMDocs용 (원본) | RAGuard용 (raguard.py) |
  |---|---|---|
  | Single LLM | `single_llm_prompt` | `single_llm_prompt_raguard` |
  | MadamRAG agent (initial/debate) | `agent_initial_prompt`, `agent_debate_prompt` | `..._raguard` |
  | V4 Round 1 Pro/Con/Mediator | `pro_prompt`, `con_prompt`, `mediator_prompt` | `..._raguard` |
  | V4 Round 1 Aggregator | `aggregator_with_confidence_prompt` | `..._raguard` |
  | madamrag/V4 Round 2+ Aggregator | `aggregator_prompt` | `aggregator_prompt_raguard` |

- **RAGuard 분기의 공통 변경**:
  1. 답을 단일 `"True"` 또는 `"False"`로 강제 (`["True"]` 또는 `["False"]`, 절대 둘 다 X)
  2. **doc-only 제약 명시** — `"Use ONLY the information explicitly stated in the document(s). Do NOT rely on your prior knowledge."` 모든 단계에 삽입. 모델이 정치 fact-checking을 internal knowledge로 답하는 것을 차단해서 공정한 비교 보장.
  3. Pro/Con/Mediator: 문서에 직접적 evidence가 없으면 `"No supporting evidence found"` / `Unknown`/`Low confidence`로 답하도록 강제. noise 문서에서 억지 의견을 만들지 않게 해서 multi-agent의 false bias 완화.

#### 평가 결과 (raguard_balanced, 230개 전체, gpt-4o-mini)

| Method | EM | Balanced Acc | True acc | False acc | Unknown 답변 |
|---|---:|---:|---:|---:|---:|
| **single_llm** | **73.9%** | **73.9%** | 58.3% | 89.6% | 0 |
| MadamRAG (baseline) | 65.7% | 65.7% | 47.8% | 83.5% | 17 |
| **V4 (proposed)** | 64.3% | 64.3% | 34.8% | 93.9% | 2 |

**Key findings:**

- **single_llm이 multi-agent baseline 모두를 능가** (+8.2%p vs MadamRAG, +9.6%p vs V4). doc-only 제약을 모든 method에 적용했음에도 격차 유지 → multi-agent debate가 RAGuard 같은 binary fact-check 형 task에는 fundamental하게 잘 맞지 않음.
- **V4 < MadamRAG (-1.4%p)**. 우리 proposed method가 baseline에 뒤짐.
- **V4의 극단적 false bias**:
  - True 정답을 34.8%만 식별 — "항상 False" baseline(50%)보다 낮음
  - False 정답은 93.9%로 가장 높음
  - 모든 답을 False로 미는 경향이 V4 > MadamRAG > single_llm 순으로 강함

**원인 추정**: V4의 Pro/Con 구조가 noise/misleading 문서에서도 양쪽 논거를 강제로 만들어내고, Mediator가 그 결과를 더 강하게 부정 쪽으로 결론 → Aggregator 입력 자체가 False 편향. RAGuard는 동명이인 disambig가 없고 정답이 단일 binary라, V4의 다중 entity 답변 통합 능력이 발휘되지 않음.

#### Ablation: GPT-4o-mini contamination 검증

문서 없이 claim만 입력했을 때:

| Setup | EM | True acc | False acc |
|---|---:|---:|---:|
| **no-doc (claim only)** | **78.3%** | 82.6% | 73.9% |
| single_llm (doc-given) | 73.9% | 58.3% | 89.6% |

→ **doc 없이 답한 게 doc 줘서 답한 것보다 4.4%p 높음**. Explanation에 "This statement is attributed to President Biden...", "Various analyses from the Tax Policy Center..." 같은 학습된 사실 인용 다수 → GPT-4o-mini가 RAGuard claim을 internal knowledge로 답함 (**PolitiFact contamination 의심**).

→ 본 평가의 multi-agent < single_llm 격차는 task 부적합보다 **single_llm의 internal knowledge 누출**에 기인한 부분이 큼. → LLAMA-8B, Qwen-7B 로 검증 (아래 [3-Model 비교](#3-model-비교-결과-raguard_balanced-230)).

**추후 검증 방향**:
- ~~LLAMA 등 open-source 작은 모델에서 격차 재측정~~ ✅ 완료 — 같은 multi-agent < single_llm 패턴 재현 (아래 참조)
- V4 구조 보강 (doc credibility-aware Pro/Con / confidence weighted voting / retrieval filter 등)

### Open-source 모델 평가 (`feature/llama`)

GPT-4o-mini의 internal knowledge 누출 가능성을 차단하고 동일 조건에서 multi-agent 효과를 재측정하기 위해, vLLM의 OpenAI-compatible 서버를 통해 LLAMA / Qwen 등 open-source 모델로도 동일 pipeline 실행이 가능하다.

#### 3-Model 비교 결과 (raguard_balanced 230)

3개 모델(GPT-4o-mini / LLAMA-3.1-8B-Instruct / Qwen2.5-7B-Instruct) × 3 method(single_llm / madamrag / v4)를 동일한 `prompts/raguard.py` 분기로 평가:

| Model | Method | EM | Bal Acc | True acc | False acc | predT | predF | empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **GPT-4o-mini** | single_llm | **73.9%** | 73.9% | 58.3% | 89.6% | 79 | 151 | 0 |
| | madamrag | 65.7% | 65.7% | 47.8% | 83.5% | 67 | 146 | 0 |
| | v4 | 64.3% | 64.3% | 34.8% | 93.9% | 47 | 181 | 0 |
| **LLAMA-3.1-8B** | single_llm | **67.8%** | 67.8% | 56.5% | 79.1% | 81 | 134 | 15 |
| | madamrag | 57.0% | 57.0% | 62.6% | 51.3% | 128 | 101 | 1 |
| | v4 | 58.7% | 58.7% | 60.9% | 56.5% | 119 | 110 | 1 |
| **Qwen2.5-7B** | single_llm | 53.5% | 53.5% | 9.6% | 97.4% | 14 | 216 | 0 |
| | madamrag | 53.9% | 53.9% | 10.4% | 97.4% | 14 | 213 | 0 |
| | v4 | 54.3% | 54.3% | 10.4% | 98.3% | 14 | 214 | 0 |

##### 주요 발견

**1. Multi-agent < single_llm — 3-model 공통 패턴**
- GPT-4o-mini, LLAMA-8B 둘 다 single_llm 대비 multi-agent가 **-8 ~ -11pp** 손해
- 같은 방향 + 비슷한 크기 → 한 모델 contamination이 원인이 아님. **RAGuard binary fact-check task가 multi-agent debate의 design assumption (multi-answer ambiguity)에 fundamental하게 안 맞음**.

**2. Failure mode가 모델별로 다름** (질적 차이)

|  | single_llm baseline | multi-agent 효과 | 메커니즘 |
|---|---|---|---|
| GPT-4o-mini | False 약간 우세 (T 58 / F 90) | **False bias 증폭** (v4 T-acc 34.8%) | 기존 prior 확신 강화 |
| LLAMA-8B | 비슷한 패턴 (T 57 / F 79) | **bias 뒤집힘** (T 60+ / F 50+) | predictions 흔들림 (noise) |

→ Multi-agent debate가 모델의 prior를 amplify하거나 dilute하지, 정보를 더 정확하게 만들지 않음 (양쪽 모델에서 다른 메커니즘으로 같은 결론).

**3. Qwen2.5-7B는 "default-False" 모델 — 비교 부적합**
- True 예측 14/230 (6%) — 클래스 거의 무시
- 3 method 다 BalAcc ~54%, 차이 < 0.8pp → method 비교 의미 없음
- 강한 RLHF refusal bias로 인한 model collapse 추정. 분석에서 제외.

**4. GPT-4o-mini의 False acc 우위 (+10.5pp vs LLAMA single_llm)는 contamination 의심**
- True acc는 거의 동일 (58.3% vs 56.5%), False acc만 격차 큼 (89.6% vs 79.1%)
- 이전 [no-doc ablation](#ablation-gpt-4o-mini-contamination-검증) (EM 78.3%) 과 일관 → PolitiFact 학습 가능성

##### Doc composition × outcome 분석 (GPT-4o-mini, n=230)

RAGuard sample의 doc 구성 (avg 6.5 doc/sample): correct 36.1%, noise 52.4%, misinfo 11.6%. (gold class) × (misinfo 유무) 4-cell:

| 조건 | n | single_llm | madamrag | v4 |
|---|---:|---:|---:|---:|
| misinfo 있음, gold=True | 62 | 54.8% | 45.2% | **25.8%** |
| **misinfo 있음, gold=False** | 25 | 60.0% | 52.0% | **76.0%** ← v4 압승 |
| misinfo 없음, gold=True | 53 | 62.3% | 50.9% | 45.3% |
| misinfo 없음, gold=False | 90 | 97.8% | 92.2% | 98.9% |

**해석**: V4는 (gold=False + misinfo 있음) sub-condition에서 single_llm 대비 **+16pp 압승**. 이건 multi-agent debate의 design이 실제 적용되는 케이스 (거짓 주장 + 헷갈리는 doc). 반대로 (gold=True + misinfo)에선 Pro voice가 misinfo doc도 legitimate evidence처럼 다뤄서 **-29pp 폭망** — method의 진짜 약점은 doc credibility 평가 부족.

##### 결론 및 paper narrative

1. **Multi-agent debate doesn't fit binary fact-check** — 3-model 동일 결과로 강력한 negative finding
2. **RAGuard는 method의 boundary case로 활용** — 어디서 안 통하는지 보여주는 negative 사례
3. **V4의 design 정당성은 RAMDocs (multi-answer ambiguous QA)에서 검증** — 본 README 하단 V4 +2.2pp 결과 참고
4. **Future work**: doc credibility-aware Pro/Con prompting, retrieval filter preprocessing (모든 method 동일 적용)

---

#### Reproduction — 새 모델로 평가 돌리기

**환경 (`nlp` conda env 기준)**

```bash
conda activate nlp

# torch + vllm + transformers stack (NVIDIA driver 535.x / CUDA 12.2 호환)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.7.3
pip install "transformers==4.48.3" "tokenizers<0.22" "huggingface_hub==0.36.2"
```

> driver 535.x는 CUDA 12.x runtime forward-compatible. torch 2.11+cu130 같은 최신 wheel은 driver 업그레이드 필요 → cu121 stack으로 핀.
> `huggingface_hub` orphan dist-info 남는 이슈 발생 시 `site-packages/huggingface_hub-*.dist-info` 중 옛 버전 디렉터리 수동 제거.

**LLAMA 모델 — HF 인증 (Meta gating 필요)**

```bash
# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct 에서 access request (보통 즉시 승인)
# https://huggingface.co/settings/tokens 에서 Read 권한 토큰 발급
huggingface-cli login   # 토큰 붙여넣기
```

**vLLM 서버 띄우기 — 4-GPU 병렬 평가용**

GPU 1장당 vLLM 인스턴스 1개. 8B 모델은 fp16 ~16GB라 A5000 24GB 1장에 fit. 3개 method를 다른 포트에 매핑해서 동시 실행:

```bash
# 권장: tmux로 long-running 세션 보호
tmux new -s vllm

# 각 window에서 (Ctrl+B C로 새 window 추가, CUDA_VISIBLE_DEVICES와 --port만 다르게)
# GPU 0 → port 8000 (single_llm 용)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --served-model-name llama-3.1-8b-instruct \
  --port 8000 \
  --max-model-len 16384 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.92

# GPU 1 → port 8001 (madamrag 용)
# GPU 2 → port 8002 (v4 용)
# 같은 명령에서 CUDA_VISIBLE_DEVICES 와 --port 만 1/8001, 2/8002 로 바꿈
```

> `--max-model-len 16384` 권장: madamrag/v4 debate history 누적 시 worst case 16K 도달. `8192`로 띄우면 후반 샘플에서 BadRequest 400 발생.
> `--max-num-seqs 8` 으로 KV cache OOM 방지 (A5000 24GB 기준).
> 로컬 path도 지원: `--model /data_seoul/models/Qwen2.5-7B-Instruct` 처럼 절대 경로 가능.

각 서버에서 `Uvicorn running on http://0.0.0.0:800X` 줄 뜨면 ready.

**클라이언트 실행 — endpoint env var + `--model` 인자**

```bash
# 각 client 터미널에서
conda activate nlp
cd /path/to/multiagent-debate-RAG
export OPENAI_BASE_URL=http://localhost:8000/v1   # vLLM 서버 주소 (포트 method별 다름)
export OPENAI_API_KEY=EMPTY                       # vLLM 은 인증 안 함, dummy

# 3 method 병렬 (각각 별도 터미널)
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
  python run_single_llm.py --dataset raguard_balanced --model llama-3.1-8b-instruct
OPENAI_BASE_URL=http://localhost:8001/v1 OPENAI_API_KEY=EMPTY \
  python run_madamrag.py   --dataset raguard_balanced --model llama-3.1-8b-instruct
OPENAI_BASE_URL=http://localhost:8002/v1 OPENAI_API_KEY=EMPTY \
  python run_v4.py         --dataset raguard_balanced --model llama-3.1-8b-instruct
```

> **env var 비어있으면 기존 OpenAI 동작 그대로 유지** — `common/llm.py`가 `OPENAI_BASE_URL` 존재 여부로 분기. gpt-4o-mini 평가는 영향 없음.
> `--served-model-name` 과 client 의 `--model` 값을 **동일하게** 맞춰야 vLLM이 요청 받음.

**예상 소요 시간** (raguard_balanced 230개, A5000 1장)
- single_llm: ~15분
- madamrag: ~1-2시간
- v4: ~1.5-3시간

**Resume / fault tolerance**
- madamrag / v4: 50개마다 checkpoint 자동 저장 + 다시 같은 명령 돌리면 그 다음부터 resume (`run_*.py` 내장)
- single_llm: 끝에 한 번만 저장 → 중간에 죽으면 처음부터 (다만 단일 호출이라 짧음)
- 16384 토큰 초과한 outlier 샘플은 자동 catch (`predicted=[]`, `error="..."` 로 placeholder 기록) → 전체 run 안 죽음. 최종 결과 파일에 fail 샘플 명시.

**출력 파일 패턴**

```
results/single_llm_raguard_balanced_full_results.json                          # gpt-4o-mini (default, slug 미포함)
results/single_llm_raguard_balanced_full_llama-3.1-8b-instruct_results.json    # LLAMA
results/single_llm_raguard_balanced_full_qwen-7b-instruct_results.json         # Qwen
```

**Troubleshooting**
- `RuntimeError: NVIDIA driver too old (12020)` → torch wheel이 newer CUDA용. 위 cu121 stack 재설치
- `libnccl.so.2: cannot open` → ghost CUDA lib 충돌. cu13 suffix 없는 `nvidia-*` 패키지 모두 제거 후 cu121 deps 재설치
- `400 BadRequest, max context length 16384` → 일부 outlier 샘플이 16K 초과. error placeholder 자동 기록되니 무시 OK. 정확히 보고 싶으면 `--max-model-len 32768 --enforce-eager --max-num-seqs 4` 로 키움
- `KeyError: 'n_gold_hit'` (구버전 checkpoint와 섞임) → `_error_placeholder()`가 `compute_metrics([], ...)` 호출하는지 확인. 옛 checkpoint 있으면 한 번 patch 필요
- HF 401 / 403 → access 신청 안 됐거나 토큰 만료. `huggingface-cli login` 다시

- 기존 평가는 `sample_100.json` (RAMDocs test split 500개 중 random 100개) 기준이었음
- n=100, EM≈0.3 기준 95% 신뢰구간 ±9%p는 메서드 간 비교용으로 너무 noisy → 전체 500개로 측정 시 ±4%p 수준
- 입력 파일을 `data/ramdocs/full.json` (전체 500개)로 통일하고 `run_v4.py` / `run_madamrag.py`에 다음 로직 내장:
  - 50개마다 중간 저장 (checkpoint)
  - 출력 파일이 존재하면 그 다음 샘플부터 이어서 실행 (resume)
- 완료된 PR
  - [#7](https://github.com/sooh-J/multiagent-debate-RAG/pull/7) — V4 500개 (merged)
  - [#8](https://github.com/sooh-J/multiagent-debate-RAG/pull/8) — MadamRAG 500개

#### V4 vs MadamRAG 비교 (둘 다 500개, 같은 RAMDocs test split)

| Method | EM | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| MadamRAG (baseline) | 27.00% | 0.6582 | 0.6433 | 0.6240 |
| **V4 (proposed)** | **29.20%** | 0.6701 | 0.6783 | **0.6464** |
| Δ (V4 − baseline) | +2.20%p | +0.0119 | +0.0350 | +0.0224 |

## 환경

```bash
pip install -r requirements.txt
```

- API 키: 프로젝트 루트에 `.env` 파일을 생성하고 아래 내용을 작성
  ```
  OPENAI_API_KEY=sk-your-api-key
  ```
