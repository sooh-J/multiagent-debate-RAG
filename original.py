from openai import OpenAI
from datasets import load_dataset, DatasetDict

OPENAI_API_KEY = ""

ds = load_dataset("HanNight/RAMDocs")
print(ds)

# 첫 번째 split에서 상위 20개 추출하여 변수에 저장
for split in ds:
    ds_sample = ds[split].select(range(min(20, len(ds[split]))))
    print(f"크기: {len(ds_sample)}")
    print(f"컬럼: {ds_sample.column_names}")
    print()
    for i, row in enumerate(ds_sample):
        print(f"[샘플 {i}]")
        for k, v in row.items():
            val = str(v)[:200]
            print(f"  {k}: {val}")
        print()
    break

"""
MADAM-RAG: Multi-Agent Debate for Ambiguity and Misinformation in RAG
논문: "Retrieval-Augmented Generation with Conflicting Evidence" (arXiv:2504.13079)

구조:
  - 각 문서마다 에이전트 1개 (문서 대변인)
  - 최대 T=3 라운드 토론
  - 모든 에이전트 답변이 이전 라운드와 동일하면 early stopping
  - Aggregator가 매 라운드 후 요약 생성 → 다음 라운드 에이전트 입력으로 전달
"""

import os
import re
import json
import sys
import string
from openai import OpenAI


class Tee:
    """stdout을 터미널과 파일에 동시에 출력"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.file = open(filepath, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.terminal

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4o-mini"
MAX_ROUNDS = 3  # T = 3


# ── 정규화 함수 (공식 코드와 동일) ────────────────────────────────────────────
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_answer(text: str) -> str:
    """'Answer: {}. Explanation: {}.' 형식에서 answer 부분만 추출 (early stopping용)"""
    match = re.search(r"Answer:\s*(.*?)\.", text)
    return match.group(1).strip() if match else text


# ── 프롬프트 함수 ──────────────────────────────────────────────────────────────
def agent_initial_prompt(query: str, document: str) -> str:
    return f"""You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

Answer the question based only on this document. Provide your answer and a step-by-step reasoning explanation.
Please follow the format: 'Answer: {{}}. Explanation: {{}}.''"""


# [수정] 시그니처 변경: prev_answers/prev_explanation → history(str)
def agent_debate_prompt(query: str, document: str, history: str) -> str:
    return f"""You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

The following responses are from other agents as additional information.
{history}

Answer the question based on the document and other agents' responses.
Please follow the format: 'Answer: {{}}. Explanation: {{}}.''"""


def aggregator_prompt(query: str, agent_responses: list[str]) -> str:
    responses_text = "\n\n".join(
        f"Agent {i+1}: {r}" for i, r in enumerate(agent_responses)
    )
    return f"""You are an aggregator synthesizing multiple agents' answers.

If there are multiple answers, please provide all possible correct answers and also provide a step-by-step reasoning explanation. If there is no correct answer, please reply 'unknown'.
Please follow the format: 'All Correct Answers: []. Explanation: {{}}.'

The following are examples:
Question: In which year was Michael Jordan born?
Agent responses:
Agent 1: Answer: 1963. Explanation: The document clearly states that Michael Jeffrey Jordan was born on February 17, 1963.
Agent 2: Answer: 1956. Explanation: The document states that Michael Irwin Jordan was born on February 25, 1956.
Agent 3: Answer: 1998. Explanation: According to the document, Michael Jeffrey Jordan was born on February 17, 1998.
Agent 4: Answer: Unknown. Explanation: The document does not include information about his birth year.
All Correct Answers: ["1963", "1956"]. Explanation: Agent 1 is talking about the basketball player Michael Jeffrey Jordan, born in 1963. Agent 2 is talking about another person named Michael Jordan, an American scientist, born in 1956. Agent 3 provides an incorrect year. Agent 4 provides no useful information.

Question: {query}
Agent responses:
{responses_text}
"""


# ── LLM 호출 ──────────────────────────────────────────────────────────────────
def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


# ── 파싱 함수 ──────────────────────────────────────────────────────────────────
def parse_answers(text: str) -> list[str]:
    """
    'All Correct Answers: ["answer1", "answer2", ...]' 에서 정답 리스트 추출
    - 따옴표로 감싸진 경우 우선 처리 → 3,506 같은 숫자 파싱 버그 방지
    """
    match = re.search(r"All Correct Answers:\s*\[([^\]]*)\]", text, re.IGNORECASE)
    if not match:
        return []

    raw = match.group(1).strip()
    if not raw:
        return []

    # 따옴표로 감싸진 경우 우선 처리
    quoted = re.findall(r'"([^"]+)"', raw)
    if quoted:
        return [a.strip() for a in quoted]

    # fallback: 콤마로 분리
    return [
        a.strip().strip("'").strip()
        for a in raw.split(",")
        if a.strip().strip("'").strip()
    ]


def parse_explanation(text: str) -> str:
    """Explanation: {} 부분 추출"""
    match = re.search(r"Explanation:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


# ── 메인 파이프라인 ────────────────────────────────────────────────────────────
def madam_rag(query: str, documents: list[str]) -> dict:
    """
    Args:
        query:     사용자 질문
        documents: 검색된 문서 리스트 (각 문서가 에이전트 1개에 매핑)

    Returns:
        {
          "final_answer":      list[str],
          "final_explanation": str,
          "rounds_run":        int,
          "round_history":     list
        }
    """
    n_agents = len(documents)
    prev_agent_outputs = [""] * n_agents  # [수정] 이전 라운드 에이전트 응답 저장용
    prev_summary: list[str] = []
    prev_explanation = ""
    round_history = []

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{'='*50}")
        print(f"Round {round_num}")
        print('='*50)

        # ── Step 1: 각 에이전트가 답변 생성 ──────────────────────────────────
        current_answers = []
        for i, doc in enumerate(documents):
            if round_num == 1:
                prompt = agent_initial_prompt(query, doc)
            else:
                # [수정] current_answers(현재 라운드 진행 중) 대신
                #        prev_agent_outputs(이전 라운드 완성본) 사용
                history = "\n".join([
                    f"Agent {j+1}: {prev_agent_outputs[j]}"
                    for j in range(n_agents) if j != i
                ])
                prompt = agent_debate_prompt(query, doc, history)

            answer = call_llm(prompt)
            current_answers.append(answer)
            print(f"\n[Agent {i+1}]\n{answer}")

        # ── Step 2: Early stopping 체크 ───────────────────────────────────────
        if round_num > 1:
            # [수정] 전체 응답 문자열이 아닌 answer 부분만 추출해서 비교
            pred_normalized = [normalize_answer(extract_answer(a)) for a in current_answers]
            prev_normalized = [normalize_answer(extract_answer(a)) for a in prev_agent_outputs]

            flag = True
            for p, q in zip(pred_normalized, prev_normalized):
                if p not in q and q not in p:
                    flag = False
                    break

            if flag:
                print(f"\n>> Early stopping at round {round_num} (all agents converged)")
                round_history.append({
                    "round": round_num,
                    "agent_responses": current_answers,
                    "aggregator_answer": prev_summary,
                    "aggregator_explanation": prev_explanation,
                    "early_stopped": True,
                })
                return {
                    "final_answer": prev_summary,
                    "final_explanation": prev_explanation,
                    "rounds_run": round_num,
                    "round_history": round_history,
                }

        # ── Step 3: Aggregator가 이번 라운드 요약 ─────────────────────────────
        agg_prompt = aggregator_prompt(query, current_answers)
        agg_output = call_llm(agg_prompt)

        agg_answer = parse_answers(agg_output)       # list[str]
        agg_explanation = parse_explanation(agg_output)

        print(f"\n[Aggregator]\nANSWER: {agg_answer}\nEXPLANATION: {agg_explanation}")

        round_history.append({
            "round": round_num,
            "agent_responses": current_answers,
            "aggregator_answer": agg_answer,
            "aggregator_explanation": agg_explanation,
            "early_stopped": False,
        })

        # 다음 라운드 준비
        prev_agent_outputs = current_answers  # [수정] 이전 라운드 응답 업데이트
        prev_summary = agg_answer
        prev_explanation = agg_explanation

    # ── 최대 라운드 도달 → 마지막 aggregator 답변 반환 ───────────────────────
    print(f"\n>> Reached max rounds (T={MAX_ROUNDS})")
    return {
        "final_answer": prev_summary,
        "final_explanation": prev_explanation,
        "rounds_run": MAX_ROUNDS,
        "round_history": round_history,
    }


# ── 평가 함수 ──────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    """소문자 + 앞뒤 공백 제거 (간단한 정규화)"""
    return text.lower().strip()


def compute_metrics(predicted_answers, gold_answers, wrong_answers):
    # normalize_answer로 구두점/관사 제거
    pred_norm  = [normalize_answer(a) for a in predicted_answers]
    gold_norm  = [normalize_answer(a) for a in gold_answers]
    wrong_norm = [normalize_answer(a) for a in wrong_answers]

    # substring 매칭으로 tp/wrong 판정
    def is_match(a, b):
        return a in b or b in a

    tp = sum(1 for p in pred_norm if any(is_match(p, g) for g in gold_norm))
    has_wrong = any(is_match(p, w) for p in pred_norm for w in wrong_norm)

    # EM: 모든 gold와 매칭되는 pred가 있고, wrong 없음
    em = (
        all(any(is_match(g, p) for p in pred_norm) for g in gold_norm) and
        len(pred_norm) == len(gold_norm) and
        not has_wrong
    )

    precision = tp / len(pred_norm) if pred_norm else 0.0
    recall    = tp / len(gold_norm) if gold_norm else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    wrong_in_pred = [p for p in pred_norm if any(is_match(p, w) for w in wrong_norm)]

    return {
        "em":                int(em),
        "precision":         round(precision, 4),
        "recall":            round(recall, 4),
        "f1":                round(f1, 4),
        "predicted_answers": predicted_answers,
        "wrong_in_pred":     wrong_in_pred,
    }


def run_on_sample(sample: dict) -> dict:
    """
    ds_sample의 단일 row를 받아 MADAM-RAG 실행

    sample 컬럼:
      question        : str
      documents       : [{'text': str, 'type': str, 'answer': str}, ...]
      disambig_entity : [str, ...]
      gold_answers    : [str, ...]
      wrong_answers   : [str, ...]
    """
    query = sample["question"]

    doc_texts = [doc["text"] for doc in sample["documents"]]
    doc_meta  = [{"type": doc["type"], "answer": doc["answer"]} for doc in sample["documents"]]

    result = madam_rag(query, doc_texts)

    # final_answer는 이미 list[str] → parse_answers 불필요
    predicted_answers = result["final_answer"] if result["final_answer"] else []
    metrics = compute_metrics(predicted_answers, sample["gold_answers"], sample["wrong_answers"])

    return {
        "question":        query,
        "disambig_entity": sample["disambig_entity"],
        "gold_answers":    sample["gold_answers"],
        "wrong_answers":   sample["wrong_answers"],
        "doc_meta":        doc_meta,
        "predicted":       predicted_answers,
        "explanation":     result["final_explanation"],
        "rounds_run":      result["rounds_run"],
        "round_history":   result["round_history"],
        **metrics,  # em, precision, recall, f1
    }


def run_on_dataset(ds_sample) -> list[dict]:
    """ds_sample 전체 순회"""
    results = []
    for i, sample in enumerate(ds_sample):
        print(f"\n[{i+1}/{len(ds_sample)}] Q: {sample['question']}")
        out = run_on_sample(sample)
        print(f"  Gold:      {out['gold_answers']}")
        print(f"  Predicted: {out['predicted']}")
        print(f"  EM={out['em']}  P={out['precision']}  R={out['recall']}  F1={out['f1']}")
        results.append(out)

    n = len(results)
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"  Exact Match : {sum(r['em']        for r in results) / n * 100:.2f}%")
    print(f"  Precision   : {sum(r['precision'] for r in results) / n:.4f}")
    print(f"  Recall      : {sum(r['recall']    for r in results) / n:.4f}")
    print(f"  F1          : {sum(r['f1']        for r in results) / n:.4f}")

    print_results_table(results)

    # 결과를 JSON 파일로 저장
    output_path = "madam_rag_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과가 '{output_path}'에 저장되었습니다.")

    return results

def print_results_table(results: list[dict]):
    print("\n" + "="*90)
    print("EXPERIMENT RESULTS")
    print("="*90)
    print(f"{'#':<4} {'Question':<40} {'Predicted':<20} {'Gold':<20} {'EM':>3} {'P':>6} {'R':>6} {'F1':>6} {'Wrong':>6}")
    print("-"*90)
    
    for i, r in enumerate(results):
        q         = r['question'][:38] + ".." if len(r['question']) > 38 else r['question']
        pred      = str(r['predicted'])[:18] + ".." if len(str(r['predicted'])) > 18 else str(r['predicted'])
        gold      = str(r['gold_answers'])[:18] + ".." if len(str(r['gold_answers'])) > 18 else str(r['gold_answers'])
        em        = "✓" if r['em'] else "✗"
        wrong_flag = "⚠" if r['wrong_in_pred'] else "-"
        
        print(f"{i:<4} {q:<40} {pred:<20} {gold:<20} {em:>3} {r['precision']:>6.2f} {r['recall']:>6.2f} {r['f1']:>6.2f} {wrong_flag:>6}")
    
    print("="*90)
    n = len(results)
    print(f"{'AVERAGE':<65} {sum(r['em'] for r in results)/n*100:>3.0f}% {sum(r['precision'] for r in results)/n:>6.2f} {sum(r['recall'] for r in results)/n:>6.2f} {sum(r['f1'] for r in results)/n:>6.2f}")
    print("="*90)
    
    # 노이즈 있는 샘플만 따로
    noisy = [r for r in results if r['wrong_answers']]
    clean = [r for r in results if not r['wrong_answers']]
    
    if noisy:
        print(f"\n  노이즈 있는 샘플 ({len(noisy)}개) EM: {sum(r['em'] for r in noisy)/len(noisy)*100:.1f}%  wrong 오염률: {sum(1 for r in noisy if r['wrong_in_pred'])/len(noisy)*100:.1f}%")
    if clean:
        print(f"  노이즈 없는 샘플 ({len(clean)}개) EM: {sum(r['em'] for r in clean)/len(clean)*100:.1f}%")

if __name__ == "__main__":
    LOG_PATH = "madam_rag_log.txt"
    tee = Tee(LOG_PATH)
    sys.stdout = tee

    try:
        # 단일 샘플 테스트
        result = run_on_sample(ds_sample[0])
        print(result)

        # 전체 데이터셋 실행
        all_results = run_on_dataset(ds_sample)
    finally:
        tee.close()
        print(f"전체 로그가 '{LOG_PATH}'에 저장되었습니다.")