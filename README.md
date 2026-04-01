# Multi-Agent Debate RAG

RAG 환경에서 충돌하는 정보(ambiguity, misinformation)를 다루기 위한 Multi-Agent Debate 실험 프로젝트.

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
├── methods/
│   └── madamrag/               # Baseline 방법론
│       ├── prompts.py          # 프롬프트 정의 (수정 시 여기만 변경)
│       ├── pipeline.py         # 토론 파이프라인
│       └── config.py           # MAX_ROUNDS 등 설정
├── data/
│   ├── full/                   # 전체 데이터셋 (git 미추적)
│   └── sample/                 # Toy experiment용 샘플 (git 추적)
├── results/                    # 실험 결과 저장
├── run_madamrag.py             # 실행 진입점
└── original.py                 # 원본 단일 파일 (참고용)
```

### 새로운 방법론 추가 방법

1. `methods/<method_name>/` 폴더 생성
2. `prompts.py` — 프롬프트 정의
3. `pipeline.py` — 파이프라인 구현 (`common/` 유틸 활용)
4. `run_<method_name>.py` — 실행 스크립트 작성

## 실행

> 현재는 20개 샘플로 toy experiment 진행 중.

```bash
conda activate nlp

# 데이터 다운로드 (최초 1회)
python -m common.data

# MadamRAG baseline 실행
python run_madamrag.py
```

## 환경

```bash
pip install -r requirements.txt
```

- API 키: 프로젝트 루트에 `.env` 파일을 생성하고 아래 내용을 작성
  ```
  OPENAI_API_KEY=sk-your-api-key
  ```
