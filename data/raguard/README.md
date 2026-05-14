# UCSC-IRKM/RAGuard

정치 발언 fact-checking 데이터셋. RAG 환경에서 검색된 문서가 주장(claim)을 뒷받침/오도/무관한지 평가하는 용도.

- 출처: https://huggingface.co/datasets/UCSC-IRKM/RAGuard
- 라이선스: MIT
- 규모: claim 2,648개 / document 16,331개

## 사용

```bash
# 1) 원본 다운로드 + 조인 (최초 1회)
python -m data.raguard.download

# 2) 전처리 (RAMDocs 스키마로 변환)
python -m data.raguard.preprocess
```

`load_dataset("UCSC-IRKM/RAGuard")` 한 번에 불러올 수 없는 데이터셋이라 (`claims.csv`와 `documents.csv`의 스키마가 달라 `DatasetGenerationCastError` 발생), `huggingface_hub.hf_hub_download`으로 두 CSV를 각각 받은 뒤 `Claim ID` 기준으로 조인합니다.

## 폴더 구조

```
raguard/
├── download.py                       # 다운로드·조인 스크립트
├── preprocess.py                     # RAMDocs 스키마 변환·필터링 스크립트
├── raw/                              # HF에서 받은 원본 CSV (확인용)
│   ├── claims.csv
│   └── documents.csv
├── full/
│   ├── raguard.json                              # Claim ID 조인본 (raw에 가까움, 디버깅·재전처리용)
│   ├── raguard_preprocessed.json                 # 전처리·스키마 변환 완료 (711 claims)
│   └── raguard_preprocessed_balanced.json        # ↑에 verdict balance 적용 (230 claims, True 115 : False 115)
└── sample/
    ├── sample_20.json
    ├── sample_20_preprocessed.json               # 전처리본 샘플
    └── sample_20_preprocessed_balanced.json      # balanced 샘플
```

## 데이터 스키마

### raw/claims.csv

| 컬럼 | 타입 | 의미 |
|---|---|---|
| `Claim ID` | int | 주장 고유 ID. `documents.csv`의 동일 컬럼과 조인 키 |
| `Claim` | str | 검증 대상이 되는 정치 발언/주장 |
| `Verdict` | bool | 이항 정답 라벨 (True / False). 본 프로젝트의 gold label |
| `Document IDs` | str (list-like) | 이 claim에 연결된 document ID 리스트 (문자열 형태) |
| `Document Labels` | str (list-like) | 위 문서들의 라벨 리스트 (supporting/misleading/unrelated) |
| `Original Verdict` | str | PolitiFact 원본 6단계 라벨 (true / mostly-true / half-true / mostly-false / false / pants-on-fire) |

### raw/documents.csv

| 컬럼 | 타입 | 의미 |
|---|---|---|
| `Document ID` | int | 문서 고유 ID |
| `Title` | str | 문서 제목 |
| `Full Text` | str | 문서 본문. **약 25%(4,132/16,331)는 `[Link Post]` 또는 빈 문자열** (Reddit 링크만 있고 본문 미수집) |
| `Claim ID` | int | 이 문서가 검색된 claim의 ID. `claims.csv`와 조인 키 |
| `Document Label` | str | claim과의 관계: `supporting` / `misleading` / `unrelated` |
| `Link` | str | 원본 문서 URL (주로 Reddit) |

### full/raguard.json (조인된 형태)

한 row = 한 claim + 거기에 연결된 모든 documents.

```json
{
  "claim_id":         "2620",
  "claim":            "...",
  "verdict":          "True" | "False",
  "original_verdict": "mostly-false",
  "document_ids":     "[16118, 16119, ...]",   // 원본 문자열 그대로
  "document_labels":  "['unrelated', ...]",    // 원본 문자열 그대로
  "documents": [
    {
      "Document ID":    "16118",
      "Title":          "...",
      "Full Text":      "...",
      "Claim ID":       "2620",
      "Document Label": "unrelated",
      "Link":           "https://..."
    }
  ]
}
```

## 원본 분포 (`raguard.json`, 전처리 전)

```
verdict           : True 1,333  /  False 1,315
original_verdict  : true 325, mostly-true 485, half-true 523,
                    mostly-false 447, false 629, pants-on-fire 239
Document Label    : unrelated 11,834  /  misleading 1,812  /  supporting 2,685
Full Text 없음    : 4,132 / 16,331 (약 25%)
```

## 전처리 (`preprocess.py`)

본 프로젝트는 RAMDocs와 같은 인터페이스로 RAGuard를 사용하기 위해 다음과 같이 변환·필터링합니다.

### 변환 규칙

1. **doc 본문 유효성 필터** — 다음 중 하나라도 해당하는 doc은 폐기
   - `Full Text`가 비어있거나 공백만
   - placeholder: `[Link Post]`, `[deleted]`, `[removed]`, `[deleted by user]`, `[removed by reddit]` (대소문자 무시)
   - URL/마크다운 링크를 모두 걷어낸 후 남는 영숫자 글자 수가 **30자 미만**
   - **`title`은 optional, `body`(=`Full Text`)는 필수** — body가 무효면 title이 의미 있어도 폐기
2. 위 결과로 `supporting` 문서가 **하나도 없는 claim 폐기**
3. 본문이 비어있는 문서를 외부에서 크롤링하지 않음 — 그냥 폐기
4. `Document Label` → `type` 매핑
   | RAGuard | → | RAMDocs |
   |---|---|---|
   | `supporting` | → | `correct` |
   | `misleading` | → | `misinfo` |
   | `unrelated` | → | `noise` |
5. 문서별 `answer` 합성
   - `correct` → `verdict` 그대로 (`"True"`/`"False"`)
   - `misinfo` → `verdict` 반전
   - `noise`   → `"unknown"`
6. `original_verdict == "half-true"` 인 claim 제외 (이항 분류 평가에 부적절한 ambiguous case)
7. 유효 doc 수가 **2개 미만**인 claim 제외
8. **(balanced 버전 한정)** verdict 분포 균형화
   - True 그룹은 모두 보존
   - False 그룹은 `original_verdict`(false / mostly-false / pants-on-fire) 비율을 유지하면서 stratified random sampling
   - 결과: True 115 + False 115 = **230 claims**

### 변환된 스키마 (`raguard_preprocessed.json`)

RAMDocs와 동일한 형태:

```json
{
  "question": "Is the following claim true or false? Answer with \"True\" or \"False\".\n\nClaim: \"...\"",
  "documents": [
    {"text": "Title\n\nBody...", "type": "correct" | "misinfo" | "noise",
     "answer": "True" | "False" | "unknown"}
  ],
  "disambig_entity": [],
  "gold_answers": ["True"] ,
  "wrong_answers": ["False"]
}
```

### 전처리 결과 통계

```
입력      : 2,648 claims / 16,331 docs
폐기 (doc): invalid body 3,861
폐기 (claim): half-true 523, supporting 0개 1,401, doc<2 13
출력      : 711 claims / 4,593 docs
```

전처리 후 분포 (`raguard_preprocessed.json`):

```
verdict           : True 115  /  False 596    ⚠️ unbalanced (약 1:5)
doc type          : correct 1,795 / misinfo 350 / noise 2,448
docs/claim        : avg 6.46  (min 2, max 16)
supporting/claim  : avg 2.52  (min 1, max 9)
```

True/False 그룹별 세부 분포:

| Gold | n | original_verdict 분포 | doc/claim avg | doc type 비율 (correct/misinfo/noise) |
|---|---|---|---|---|
| True  | 115 | true 39 / mostly-true 76 | 6.43 | 27.7% / 18.1% / 54.2% |
| False | 596 | false 289 / mostly-false 178 / pants-on-fire 129 | 6.46 | 41.3% / 5.6% / 53.1% |

> **단순 accuracy 평가는 무의미.** baseline이 항상 "False"만 답해도 ~84%가 나옵니다. 평가 시 **F1 / balanced accuracy / per-class precision·recall**을 메인 메트릭으로 사용하세요. 또는 아래 balanced 버전 사용.

### Balanced 버전 분포 (`raguard_preprocessed_balanced.json`)

```
verdict           : True 115  /  False 115   (1:1)
total             : 230 claims / 1,495 docs
```

False 그룹 stratified sampling 결과 — `original_verdict` 비율 유지됨:

| original_verdict | 원본 (596) | balanced (115) |
|---|---|---|
| `false` | 289 (48.5%) | 56 (48.7%) |
| `mostly-false` | 178 (29.9%) | 34 (29.6%) |
| `pants-on-fire` | 129 (21.6%) | 25 (21.7%) |

## RAMDocs 스키마와의 차이 (참고)

| 항목 | RAMDocs | RAGuard 원본 | RAGuard 전처리 후 |
|---|---|---|---|
| Task | open-domain QA + 모호성/misinfo 견고성 | claim fact-checking | (RAMDocs 인터페이스로 통일) |
| 입력 | `question` (질문) | `claim` (주장) | `question` (claim wrapping prompt) |
| 정답 | `gold_answers` (list), `wrong_answers` (list) | `verdict` (True/False 단일) | `gold_answers=[verdict]`, `wrong_answers=[~verdict]` |
| 문서별 answer | 있음 | 없음 | 합성 (위 규칙 5) |
| 문서 라벨 | `correct` / `misinfo` / `noise` | `supporting` / `misleading` / `unrelated` | `correct` / `misinfo` / `noise` |
| 부가 정보 | `disambig_entity` | `original_verdict`, `Link`, `Title` | (전처리에서 drop) |
