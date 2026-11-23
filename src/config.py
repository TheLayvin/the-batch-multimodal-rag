import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
IMAGES_DIR = DATA_DIR / "images"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GENERATION_MODEL = "gemini-3.5-flash-lite"

EMBEDDING_MODEL = "gemini-embedding-001"

THE_BATCH_BASE_URL = "https://www.deeplearning.ai/the-batch/"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")