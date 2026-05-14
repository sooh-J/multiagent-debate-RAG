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
│   └── raguard/                # RAGuard 데이터셋 (다운로드 + 샘플) — 스키마 통합 미정, data/raguard/README.md 참고
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

# 파이프라인 실행 (100 samples)
python run_single_llm.py         # Single LLM baseline
python run_madamrag.py           # MadamRAG baseline
python run_proposed_method.py    # Proposed Method (V2)
python run_v3.py                 # V3
python run_v4.py                 # V4

# LLM-as-a-Judge 재채점
python eval_llm_judge.py results/madamrag_results.json
python eval_llm_judge.py results/single_llm_results.json
```

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

## 환경

```bash
pip install -r requirements.txt
```

- API 키: 프로젝트 루트에 `.env` 파일을 생성하고 아래 내용을 작성
  ```
  OPENAI_API_KEY=sk-your-api-key
  ```
