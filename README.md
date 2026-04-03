# Multi-Agent Debate RAG

RAG 환경에서 충돌하는 정보(ambiguity, misinformation)를 다루기 위한 Multi-Agent Debate 실험 프로젝트.

## 방법론 요약

| 방법론 | 설명 | 에이전트 구조 | 라운드 |
|--------|------|---------------|--------|
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
│   ├── llm.py                  # OpenAI API 호출
│   ├── parsing.py              # 텍스트 정규화 및 파싱
│   ├── metrics.py              # 평가 지표 (EM, F1 등)
│   ├── logging.py              # 로그 출력 (Tee)
│   └── data.py                 # 데이터셋 로드/캐싱
├── prompts/                    # 방법론별 프롬프트
│   ├── madamrag.py             # MadamRAG 프롬프트
│   ├── proposed_method.py      # Proposed Method (V2) 프롬프트 — Extractor/Skeptic/Resolver
│   └── v3.py                   # V3/V4 공통 프롬프트 — Pro/Con/Mediator
├── pipelines/                  # 방법론별 파이프라인
│   ├── madamrag.py             # MadamRAG 토론 파이프라인
│   ├── proposed_method.py      # Proposed Method (V2) 파이프라인 — 문서 내부 토론 + 글로벌 집계
│   ├── v3.py                   # V3 파이프라인 — 찬/반/중재자 매 라운드 반복
│   └── v4.py                   # V4 파이프라인 — V3(Round 1) + MadamRAG(Round 2+)
├── configs/                    # 방법론별 설정
│   ├── madamrag.py             # MAX_ROUNDS 등
│   ├── proposed_method.py      # Proposed Method 설정
│   └── v3.py                   # V3/V4 공통 설정 (MAX_ROUNDS = 3)
├── data/
│   ├── full/                   # 전체 데이터셋 (git 미추적)
│   └── sample/                 # 실험용 샘플 (sample_3, sample_20, sample_100)
├── results/                    # 실험 결과 저장
├── run_madamrag.py             # MadamRAG 실행
├── run_proposed_method.py      # Proposed Method (V2) 실행
├── run_v3.py                   # V3 실행
├── run_v4.py                   # V4 실행
└── original.py                 # 원본 단일 파일 (참고용)
```

### 새로운 방법론 추가 방법

1. `prompts/<method_name>.py` — 프롬프트 정의
2. `pipelines/<method_name>.py` — 파이프라인 구현 (`common/` 유틸 활용)
3. `configs/<method_name>.py` — 설정값
4. `run_<method_name>.py` — 실행 스크립트 작성

## 실행

> 현재는 20개 샘플로 toy experiment 진행 중.

```bash
conda activate nlp

# 데이터 다운로드 (최초 1회)
python -m common.data

# MadamRAG baseline 실행
python run_madamrag.py

# Proposed Method (V2) 실행
python run_proposed_method.py

# V3 실행 (찬/반/중재자 매 라운드)
python run_v3.py

# V4 실행 (V3 Round 1 + MadamRAG Round 2+)
python run_v4.py
```

## 환경

```bash
pip install -r requirements.txt
```

- API 키: 프로젝트 루트에 `.env` 파일을 생성하고 아래 내용을 작성
  ```
  OPENAI_API_KEY=sk-your-api-key
  ```
