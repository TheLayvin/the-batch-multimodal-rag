import json
import time
from google import genai
from google.genai import types
import chromadb
from tqdm import tqdm
from google.genai import errors

from src.config import (
    RAW_DATA_DIR,
    CHROMA_DB_DIR,
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
)

COLLECTION_NAME = "the_batch_multimodal"

def get_genai_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in .env file!")
    return genai.Client(api_key=GEMINI_API_KEY)

def get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection

def generate_embeddings(
    client: genai.Client,
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    max_retries: int = 5,
) -> list[list[float]]:
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )

            return [emb.values for emb in response.embeddings]
        except errors.APIError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 25
                print(f"Rate limit (429) hit. Waiting {wait_time}s before retry ({attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError("Exceeded max retries for embedding generation.")

def build_vector_index(batch_size: int = 50) -> None:
    input_file = RAW_DATA_DIR / "processed_chunks.json"
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}. Run 'python -m src.processor' first!")
    
    with open(input_file, "r", encoding="utf-8") as f:
        documents = json.load(f)
    
    genai_client = get_genai_client()
    collection = get_chroma_collection()

    existing_ids = set(collection.get(include=[])["ids"])
    documents = [doc for doc in documents if doc["id"] not in existing_ids]
    if not documents:
        print(f"All items are already indexed! Total in ChromaDB: {collection.count()}")
        return

    print(f"Indexing {len(documents)} items into ChromaDB (batch_size={batch_size})")

    for i in tqdm(range(0, len(documents), batch_size), desc="Indexing batches"):
        batch = documents[i : i + batch_size]

        ids = [doc["id"] for doc in batch] 
        contents = [doc["content"] for doc in batch]
        metadatas = [
            {
                "type": doc["type"],
                "article_id": doc["article_id"],
                "title": doc["title"],
                "url": doc["url"],
                "image_path": doc["image_path"],
            }
            for doc in batch
        ]

        embeddings = generate_embeddings(
            client=genai_client,
            texts=contents,
            task_type="RETRIEVAL_DOCUMENT",
        )

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )
        time.sleep(0.3)

    print(f"Indexing completed! Total items in ChromaDB: {collection.count()}")

def search_multimodal(
    query: str,
    top_k_text: int = 4,
    top_k_images: int = 3,
) -> dict:
    genai_client = get_genai_client()
    collection = get_chroma_collection()

    query_embedding = generate_embeddings(
        client=genai_client,
        texts=[query],
        task_type="RETRIEVAL_QUERY",
    )[0]

    text_results = collection.query(
        query_embeddings = [query_embedding],
        n_results=top_k_text,
        where={"type": "text"}
    )

    image_results = collection.query(
        query_embeddings = [query_embedding],
        n_results=top_k_images,
        where={"type": "image"}
    )

    return {
        "texts": _format_chroma_results(text_results),
        "images": _format_chroma_results(image_results),
    }

def _format_chroma_results(raw_results: dict) -> list[dict]:
    formatted = []
    if not raw_results or not raw_results["ids"] or not raw_results["ids"][0]:
        return formatted
    
    ids = raw_results["ids"][0]
    docs = raw_results["documents"][0]
    metas = raw_results["metadatas"][0]
    dists = raw_results["distances"][0]

    for doc_id, content, meta, dist in zip(ids, docs, metas, dists):
        formatted.append({
            "id": doc_id,
            "content": content,
            "metadata": meta,
            "similarity": round(1.0 - dist, 4),
        })
        
    return formatted

if __name__ == "__main__":
    build_vector_index()

    test_query = "DeepSeek reasoning model performance and benchmarks"
    print(f"Test Search: '{test_query}'")
    results = search_multimodal(test_query, top_k_text=2, top_k_images=2)
    print("Top Text Matches:")
    for item in results["texts"]:
        print(f"  - [{item['similarity']}] {item['metadata']['title']} ({item['id']})")
    print("Top Image Matches:")
    for item in results["images"]:
        print(f"  - [{item['similarity']}] {item['metadata']['image_path']}")


