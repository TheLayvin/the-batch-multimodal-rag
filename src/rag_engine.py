import base64
import io
from PIL import Image
from google import genai

from src.config import BASE_DIR, GEMINI_API_KEY, GENERATION_MODEL
from src.vector_store import search_multimodal

SYSTEM_INSTRUCTION = """You are an expert AI Research Assistant for 'The Batch' newsletter by DeepLearning.AI.
Your task is to answer the user's question accurately using ONLY the provided article excerpts and attached images.
Rules:
1. Base your answer strictly on the provided text context and visual media (charts, diagrams, photos).
2. If the provided images contain charts, benchmarks, or numbers relevant to the question, explicitly extract and reference those visual details.
3. Cite the article titles when referencing facts (e.g., according to "Article Title").
4. If the retrieved context does not contain enough information to answer the question, state clearly that you could not find that information in The Batch knowledge base. Do NOT hallucinate or use outside knowledge.
5. Keep the answer well-structured, clear, and concise."""

def _encode_image_base64(rel_path: str) -> str | None:
    full_path = BASE_DIR / rel_path
    if not full_path.exists():
        return None
    try:
        with Image.open(full_path) as img:
            img = img.convert("RGB")
            img.thumbnail((1024, 1024))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[WARN] Could not load image {rel_path}: {e}")
        return None

def _collect_unique_sources(text_hits: list[dict], image_hits: list[dict]) -> list[dict]:
    sources_by_id = {}
    for hit in text_hits + image_hits:
        meta = hit["metadata"]
        article_id = meta.get("article_id", "")
        if article_id and article_id not in sources_by_id:
            sources_by_id[article_id] = {
                "article_id": article_id,
                "title": meta.get("title", "Untitled"),
                "url": meta.get("url", ""),
            }
    return list(sources_by_id.values())

def ask_rag(
    query: str,
    top_k_text: int = 4,
    top_k_images: int = 3,
    min_image_similarity: float = 0.50,
) -> dict:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in .env file!")
    
    search_results = search_multimodal(
        query=query,
        top_k_text=top_k_text,
        top_k_images=top_k_images,
    )
    text_hits = search_results["texts"]
    raw_image_hits = search_results["images"]

    image_hits = [
        img for img in raw_image_hits
        if img["similarity"] >= min_image_similarity
    ]

    context_blocks = []
    for idx, hit in enumerate(text_hits, start=1):
        title = hit["metadata"].get("title", "Untitled")
        context_blocks.append(
            f"[Text Excerpt {idx} | Source: {title} | Similarity: {hit['similarity']}]\n"
            f"{hit['content']}"
        )
    formatted_context = "\n\n---\n\n".join(context_blocks)

    user_prompt = (
        f"User Question: {query}\n\n"
        f"Retrieved Text Context:\n{formatted_context}\n\n"
        "Please answer the question based on the text excerpts above and the attached images."
    )

    input_parts = [{"type": "text", "text": user_prompt}]
    attached_images = []

    for img_hit in image_hits:
        rel_path = img_hit["metadata"].get("image_path", "")
        b64_data = _encode_image_base64(rel_path)
        if b64_data:
            input_parts.append({
                "type": "image",
                "data": b64_data,
                "mime_type": "image/jpeg",
            })
            attached_images.append(img_hit)

    client = genai.Client(api_key=GEMINI_API_KEY)
    interaction = client.interactions.create(
        model=GENERATION_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config={"temperature": 0.2},
        input=input_parts,
    )

    answer = interaction.output_text.strip() if interaction.output_text else "No response generated."
    sources = _collect_unique_sources(text_hits, attached_images)

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "retrieved_texts": text_hits,
        "retrieved_images": attached_images,
    }

if __name__ == "__main__":
    sample_query = "How does DeepSeek-R1 compare to OpenAI o1 on benchmarks and cost?"
    print(f"Query: {sample_query}\n")
    result = ask_rag(sample_query)
    print("AI Answer:")
    print(result["answer"])
    print("\nSources:")
    for src in result["sources"]:
        print(f"  - {src['title']} ({src['url']})")
    print("\nAttached Images:")
    for img in result["retrieved_images"]:
        print(f"  - [{img['similarity']}] {img['metadata']['image_path']}")