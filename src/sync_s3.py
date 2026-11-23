import argparse
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from src.config import BASE_DIR, DATA_DIR, CHROMA_DB_DIR, S3_BUCKET_NAME

SYNC_DIRS = [DATA_DIR, CHROMA_DB_DIR]

def upload_to_s3(bucket_name: str) -> None:
    s3 = boto3.client("s3")
    uploaded_count = 0

    for local_dir in SYNC_DIRS:
        if not local_dir.exists():
            print(f"[WARN] Directory does not exist, skipping: {local_dir}")
            continue

        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                s3_key = file_path.relative_to(BASE_DIR).as_posix()
                print(f"Uploading s3://{bucket_name}/{s3_key}...")
                s3.upload_file(str(file_path), bucket_name, s3_key)
                uploaded_count += 1

    print(f"Upload complete! Total files uploaded: {uploaded_count}")


def download_from_s3(bucket_name: str) -> None:
    if CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir()):
        print("Local ChromaDB already populated. Skipping S3 download.")
        return

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    downloaded_count = 0

    print(f"Syncing index and media from s3://{bucket_name}/ ...")
    for prefix in ["data/", "chroma_db/"]:
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                if s3_key.endswith("/"):
                    continue

                local_path = BASE_DIR / s3_key
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket_name, s3_key, str(local_path))
                downloaded_count += 1

    print(f"Sync complete! Downloaded {downloaded_count} files from S3.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync RAG data and ChromaDB with AWS S3")
    parser.add_argument("--upload", action="store_true", help="Upload local data to S3")
    parser.add_argument("--download", action="store_true", help="Download data from S3")
    args = parser.parse_args()

    if not S3_BUCKET_NAME:
        print("[INFO] S3_BUCKET_NAME is not set. Skipping S3 sync.")
    elif args.upload:
        upload_to_s3(S3_BUCKET_NAME)
    elif args.download:
        try:
            download_from_s3(S3_BUCKET_NAME)
        except ClientError as e:
            print(f"[ERROR] Failed to sync from S3: {e}")
            raise
    else:
        print("Specify --upload or --download")