import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.config import THE_BATCH_BASE_URL, RAW_DATA_DIR, IMAGES_DIR

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
}

def get_article_links(max_articles: int = 10) -> list[str]:
    seed_urls = [
        THE_BATCH_BASE_URL,
        urljoin(THE_BATCH_BASE_URL, "tag/research/"),
        urljoin(THE_BATCH_BASE_URL,  "tag/business/"),
    ]

    excluded_prefixes = (
        "/the-batch/tag/",
        "/the-batch/about",
        "/the-batch/search",
        "/the-batch/page/",
    )

    article_urls = []
    seen = set()

    for seed_url in seed_urls:
        try:
            response = requests.get(seed_url, headers=HEADERS, timeout=15)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"[WARN] Не вдалося відкрити {seed_url}: {e}")
            continue
        
        soup = BeautifulSoup(response.text, "html.parser")

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()

            if not href.startswith("/the-batch/") or href == "/the-batch/":
                continue

            if any(href.startswith(prefix) for prefix in excluded_prefixes):
                continue

            full_url = urljoin("https://www.deeplearning.ai", href)
            full_url = full_url.split("#")[0].rstrip("/")

            if full_url not in seen:
                seen.add(full_url)
                article_urls.append(full_url)

            if(len(article_urls) >= max_articles):
                return article_urls
    
    return article_urls

def extract_real_image_url(img_src: str) -> str | None:
    if not img_src:
        return None

    if "/_next/image" in img_src:
        parsed = urlparse(img_src)
        query_params = parse_qs(parsed.query)
        if "url" in query_params:
            img_src = unquote(query_params["url"][0])

    ignore_patterns = ["/dlai/assets/", ".svg", "dlai-logo", "badge-app-store", "badge-google-play"]
    if any(pattern in img_src.lower() for pattern in ignore_patterns):
        return None

    if img_src.startswith("http"):
        return img_src

    return None


def download_image(img_url: str, article_slug: str, img_index: int) -> Path | None:
    try:
        response = requests.get(img_url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        parsed_path = urlparse(img_url).path
        ext = Path(parsed_path).suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:
            ext = ".png"
        
        filename = f"{article_slug}_img_{img_index}{ext}"
        local_path = IMAGES_DIR / filename
        
        with open(local_path, "wb") as f:
            f.write(response.content)
        return local_path
    except requests.RequestException as e:
        print(f"[WARN] Не вдалося скачати картинку {img_url}: {e}")
        return None

def parse_article(url: str) -> dict | None:
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] Помилка завантаження статті {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    
    article_slug = url.rstrip("/").split("/")[-1]

    h1_tag = soup.find("h1")
    title = h1_tag.get_text(strip=True) if h1_tag else article_slug.replace("-", " ").title()

    content_container = soup.find("article") or soup.find("main") or soup.body
    if not content_container:
        return None

    text_blocks = []
    for elem in content_container.find_all(["h2", "h3", "p", "li"]):
        text = elem.get_text(" ", strip=True)
        if len(text) < 25:
            continue
        if elem.name in ["h2", "h3"]:
            text_blocks.append(f"\n## {text}\n")
        else:
            text_blocks.append(text)
    clean_text = "\n\n".join(text_blocks)

    images_data = []
    seen_ing_urls = set()
    img_counter = 1
    for img_tag in content_container.find_all("img"):
        raw_src = img_tag.get("src") or img_tag.get("srcset", "").split(" ")[0]
        real_img_url = extract_real_image_url(raw_src)

        if not real_img_url or real_img_url in seen_ing_urls:
            continue
        
        seen_ing_urls.add(real_img_url)
        alt_text = img_tag.get("alt", "").strip()

        local_path = download_image(real_img_url, article_slug, img_counter)
        if local_path:
            images_data.append({
                "image_id": f"{article_slug}_img_{img_counter}",
                 "local_path": str(local_path.relative_to(RAW_DATA_DIR.parent.parent)),
                "source_url": real_img_url,
                "alt_text": alt_text,
            })
            img_counter += 1

    return {
        "article_id": article_slug,
        "title": title, 
        "url": url,
        "text": clean_text,
        "images": images_data,
    }

def run_scraper(max_articles: int = 8) -> list[dict]:
    print(f"Searching for articles on {THE_BATCH_BASE_URL}...")
    urls = get_article_links(max_articles=max_articles)
    print(f"Found {len(urls)} articles to download.")

    articles = []
    for url in tqdm(urls, desc="Scraping The Batch articles"):
        article_data = parse_article(url)
        if article_data and len(article_data["text"]) > 200:
            articles.append(article_data)
        time.sleep(0.5)
    
    output_file = RAW_DATA_DIR / "articles.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    total_images = sum(len(a["images"]) for a in articles)
    print(f"Done! Saved {len(articles)} articles and {total_images} images.")
    print(f"Data saved to: {output_file}")
    return articles

if __name__ == "__main__":
    run_scraper(max_articles=8)