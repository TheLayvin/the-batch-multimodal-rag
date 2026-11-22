import json
import re
import time
from google import genai

from src.config import GEMINI_API_KEY, GENERATION_MODEL
from src.rag_engine import ask_rag

EVAL_DATASET = [
    {
        "query": "How does DeepSeek-R1 compare to OpenAI o1 on benchmarks and API pricing?",
        "expected_article_id": "deepseek-r1-an-affordable-rival-to-openais-o1",
    },
    {
        "query": "What is DeepLearning.AI Pro and why was it launched?",
        "expected_article_id": "introducing-deeplearning-ai-pro",
    },
    {
        "query": "What are the details of Anthropic's S-1 filing, IPO valuation, and revenue?",
        "expected_article_id": "issue-378",
    },
    {
        "query": "What is OpenAI Operator and what model powers its web actions?",
        "expected_article_id": "issue-286",
    },
    {
        "query": "What is the difference between verifiable tasks and judgement tasks for LLMs?",
        "expected_article_id": "issue-375",
    },
    {
        "query": "What is an agent harness and what SDK did OpenWorker release?",
        "expected_article_id": "issue-379",
    },
]

def compute_retrieval_metrics(
    retrieved_texts: list[dict],
    retrieved_images: list[dict],
    expected_article_id: str,
) -> dict:
    text_articles_ids = [hit["metadata"].get("article_id") for hit in retrieved_texts]
    image_article_ids = [hit["metadata"].get("article_id") for hit in retrieved_images]

    text_hit = 1.0 if expected_article_id in text_articles_ids else 0.0
    reciprocal_rank = 0.0

    for rank, art_id in enumerate(text_articles_ids, start=1):
        if art_id == expected_article_id:
            reciprocal_rank = 1 / rank
            break

    visual_hit = 1.0 if expected_article_id in image_article_ids else 0.0

    return {
        "text_hit": text_hit,
        "reciprocal_rank": reciprocal_rank,
        "visual_hit": visual_hit,
    }

def judge_answer_with_llm(
    client: genai.Client,
    query: str,
    context_texts: list[dict],
    answer: str,
) -> dict:
    context_str = "\n---\n".join(hit["content"] for hit in context_texts)

    judge_prompt = f"""You are an impartial AI Evaluation Judge for a RAG system.
        Evaluate the Generated Answer based on the User Question and Retrieved Context.
        User Question: 
        {query}
        Retrieved Context:
        {context_str}
        Generated Answer:
        {answer}
        Score on a 1 to 5 scale (5 is best):
        - faithfulness: Are all facts in the answer grounded in the context (or obvious chart readings from the same article) with no hallucinations?
        - relevance: Does the answer directly and clearly address the user's question?
        Respond ONLY with a valid JSON object in this exact format:
        {{"faithfulness": 5, "relevance": 5, "reason": "One short sentence explaining the score."}}"""

    interaction = client.interactions.create(
        model=GENERATION_MODEL,
        generation_config={"temperature": 0.0},
        input=judge_prompt,
    )

    raw_output = interaction.output_text.strip() if interaction.output_text else "{}"
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return {
                "faithfulness": float(parsed.get("faithfulness", 0)),
                "relevance": float(parsed.get("relevance", 0)),
                "reason": parsed.get("reason", ""),
            }
        except json.JSONDecodeError:
            pass
    
    return {"faithfulness": 0.0, "relevance": 0.0, "reason": "Failed to parse judge JSON."}

def run_evaluation() -> dict:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in .env file!")

    client = genai.Client(api_key=GEMINI_API_KEY)
    results = []

    print(f"Running Multimodal RAG Evaluation on {len(EVAL_DATASET)} test queries\n")

    for idx, sample in enumerate(EVAL_DATASET, start=1):
        query = sample["query"]
        expected_id = sample["expected_article_id"]
        print(f"[{idx}/{len(EVAL_DATASET)}] Testing: {query}")

        rag_out = ask_rag(query, top_k_text=4, top_k_images=3, min_image_similarity=0.45)

        ret_metrics = compute_retrieval_metrics(
            retrieved_texts=rag_out["retrieved_texts"],
            retrieved_images=rag_out["retrieved_images"],
            expected_article_id=expected_id,
        )

        judge_metrics = judge_answer_with_llm(
            client=client,
            query=query,
            context_texts=rag_out["retrieved_texts"],
            answer=rag_out["answer"],
        )

        row = {
            "query": query,
            "expected_article_id": expected_id,
            **ret_metrics,
            **judge_metrics,
        }

        results.append(row)

        print(
            f"   -> Text Hit: {int(row['text_hit'])} | RR: {row['reciprocal_rank']:.2f} | "
            f"Visual Hit: {int(row['visual_hit'])} | "
            f"Faithfulness: {row['faithfulness']}/5 | Relevance: {row['relevance']}/5"
        )
        time.sleep(1.0)

    n = len(results)
    summary = {
        "total_queries": n,
        "text_hit_rate_at_4": round(sum(r["text_hit"] for r in results) / n, 4),
        "mrr_at_4": round(sum(r["reciprocal_rank"] for r in results) / n, 4),
        "visual_hit_rate_at_3": round(sum(r["visual_hit"] for r in results) / n, 4),
        "avg_faithfulness": round(sum(r["faithfulness"] for r in results) / n, 2),
        "avg_relevance": round(sum(r["relevance"] for r in results) / n, 2),
        "details": results,
    }

    print("\n" + "=" * 55)
    print("FINAL EVALUATION REPORT")
    print("=" * 55)
    print(f"Text Hit Rate @ 4:    {summary['text_hit_rate_at_4'] * 100:.1f}%")
    print(f"MRR @ 4:              {summary['mrr_at_4']:.3f}")
    print(f"Visual Hit Rate @ 3:  {summary['visual_hit_rate_at_3'] * 100:.1f}%")
    print(f"Avg Faithfulness:     {summary['avg_faithfulness']} / 5.0")
    print(f"Avg Answer Relevance: {summary['avg_relevance']} / 5.0")
    print("=" * 55)
    return summary

if __name__ == "__main__":
    run_evaluation()
