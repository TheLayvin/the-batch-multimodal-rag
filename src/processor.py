import base64
import json
import mimetypes
import time
import io
from PIL import Image
from pathlib import Path

from google import genai
from tqdm import tqdm

from src.config import (
    BASE_DIR,
    RAW_DATA_DIR,
    GEMINI_API_KEY,
    GENERATION_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

def get_gemini_client() -> genai.Client | None:
    if not GEMINI_API_KEY:
        print("[WARN] GEMINI_API_KEY is not set in .env! Using alt_text only for images.")
        return None
    
    return genai.Client(api_key=GEMINI_API_KEY)

def chunk_article_text(article: dict) -> list[dict]:
    text = article.get("text", "")
    title = article.get("title", "Untitled")
    article_id = article.get("article_id", "unknown")
    url = article.get("url", "")

    article_images = [img["local_path"] for img in article.get("images", [])]
    primary_image = article_images[0] if article_images else ""

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_paragraphs = []
    current_length = 0
    chunk_index = 1

    for para in paragraphs:
        para_len = len(para)

        if current_length + para_len > CHUNK_SIZE and current_paragraphs:
            chunk_body = "\n\n".join(current_paragraphs)
            enriched_text = f"Article:{title}\n\n{chunk_body}"

            chunks.append({
                "id": f"{article_id}_text_{chunk_index}",
                "type": "text",
                "article_id": article_id,
                "title": title,
                "url": url,
                "content": enriched_text,
                "image_path": primary_image,
            })
            chunk_index += 1

            overlap_paragraphs = []
            overlap_len = 0
            for prev_para in reversed(current_paragraphs):
                if overlap_len + len(prev_para) <= CHUNK_OVERLAP or not overlap_paragraphs:
                    overlap_paragraphs.insert(0, prev_para)
                    overlap_len += len(prev_para)
                else:
                    break
            
            current_paragraphs = overlap_paragraphs
            current_length = overlap_len
        
        current_paragraphs.append(para)
        current_length += para_len

    if current_paragraphs:
        chunk_body = "\n\n".join(current_paragraphs)
        enriched_text = f"Article: {title}\n\n{chunk_body}"
        chunks.append({
            "id": f"{article_id}_text_{chunk_index}",
            "type": "text",
            "article_id": article_id,
            "title": title,
            "url": url,
            "content": enriched_text,
            "image_path": primary_image,
        })
    
    return chunks

def describe_image_with_gemini(
    client: genai.Client | None,
    image_info: dict,
    article_title: str,
) -> str:
    alt_text = image_info.get("alt_text", "").strip()
    local_rel_path = image_info.get("local_path", "")
    full_img_path = BASE_DIR / local_rel_path
    
    if client is None or not full_img_path.exists():
        fallback = alt_text if alt_text else f"Illustration from article: {article_title}"
        return f"Image from '{article_title}'. Description: {fallback}"

    try:
        with Image.open(full_img_path) as img:
            img = img.convert("RGB")
            img.thumbnail((1024, 1024))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            b64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        prompt = (
            f"This image is from an AI news article titled '{article_title}'. "
            f"Existing caption/alt-text: '{alt_text}'. "
            "Provide a concise, dense, factual description (3-4 sentences) optimized for search retrieval. "
            "If it is a chart, graph, or table, explicitly list the model names, benchmarks, axes, and key findings. "
            "If it is a photo or diagram, describe the key subjects and technical concepts shown."
        )

        interaction = client.interactions.create(
            model=GENERATION_MODEL,
            input=[
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "data": b64_image,
                    "mime_type": "image/jpeg",
                },
            ],
        )

        vlm_caption = interaction.output_text.strip() if interaction.output_text else ""
    except Exception as e:
        print(f"[WARN] Gemini vision caption failed for {full_img_path.name}: {e}")
        vlm_caption = ""
    
    combined_parts = [f"Visual media from article: {article_title}."]
    if alt_text:
        combined_parts.append(f"Original caption: {alt_text}.")
    if vlm_caption:
        combined_parts.append(f"Visual analysis: {vlm_caption}")
    
    return " ".join(combined_parts)

def process_all_articles(max_vlm_images: int = 15) -> list[dict]:
    input_file = RAW_DATA_DIR / "articles.json"
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}. Run 'python -m src.scraper' first!")
    
    with open(input_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    client = get_gemini_client()
    all_documents = []
    vlm_calls_count = 0

    print(f"Processing {len(articles)} articles (text chunking + image analysis)")

    for article in tqdm(articles, desc="Processing articles"):
        text_chunks = chunk_article_text(article)
        all_documents.extend(text_chunks)
    
        title = article.get("title", "Untitled")
        article_id = article.get("article_id", "unknown")
        url = article.get("url", "")

        for img_info in article.get("images", []):
            use_client = client if vlm_calls_count < max_vlm_images else None
            description = describe_image_with_gemini(use_client, img_info, title)

            if use_client is not None:
                vlm_calls_count += 1
                time.sleep(0.5)
            
            all_documents.append({
                "id": img_info["image_id"],
                "type": "image",
                "article_id": article_id,
                "title": title,
                "url": url,
                "content": description,
                "image_path": img_info["local_path"],
            })
        
    output_file = RAW_DATA_DIR / "processed_chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=2)
    
    text_count = sum(1 for doc in all_documents if doc["type"] == "text")
    image_count = sum(1 for doc in all_documents if doc["type"] == "image")

    print(f"Done! Created {len(all_documents)} total searchable items:")
    print(f"Text chunks: {text_count}")
    print(f"Image documents: {image_count} ({vlm_calls_count} analyzed with Gemini Vision)")
    print(f"Saved to: {output_file}")

    return all_documents

if __name__ == "__main__":
    process_all_articles(max_vlm_images=15)

