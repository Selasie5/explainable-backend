import os
import hashlib
import logging
import joblib
import pandas as pd
from fastapi import UploadFile, HTTPException, status
from typing import Tuple
import traceback
from utils.logger import logger
ALLOWED_EXTENSIONS = {
    "csv": ["text/csv", "application/vnd.ms-excel"],
    "pkl": ["application/x-pickle", "application/octet-stream"]
}


def validate_file_type(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename)[1].lower().strip(".")
    content_type = file.content_type

    logger.debug(f"Validating file: {file.filename}, extension: {ext}, content_type: {content_type}")

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension: {ext}. Allowed extensions are {list(ALLOWED_EXTENSIONS.keys())}"
        )

    if content_type not in ALLOWED_EXTENSIONS[ext]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported MIME type: {content_type} for extension {ext}"
        )
    
    return ext

# --- File Hashing for Caching ---
def compute_file_hash(file: UploadFile) -> str:
    hash_sha256 = hashlib.sha256()
    contents = file.file.read()
    hash_sha256.update(contents)
    file.file.seek(0)
    return hash_sha256.hexdigest()

# --- Cache Path Management ---
def get_cache_path(file_hash: str, ext: str) -> str:
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{file_hash}.{ext}")

# --- Save to Cache ---
def save_to_cache(data, cache_path: str):
    joblib.dump(data, cache_path)
    logger.info(f"Saved file to cache: {cache_path}")

# --- Load from Cache ---
def load_from_cache(cache_path: str):
    logger.info(f"Loaded file from cache: {cache_path}")
    return joblib.load(cache_path)

# --- Load and Validate Model/File Content ---
def process_uploaded_file(file: UploadFile, expected_columns: list = None) -> Tuple[object, str]:
    try:
        ext = validate_file_type(file)
        file_hash = compute_file_hash(file)
        cache_path = get_cache_path(file_hash, ext)

        if os.path.exists(cache_path):
            data = load_from_cache(cache_path)
            return data, "from_cache"

        if ext == "csv":
            file.file.seek(0)
            df = pd.read_csv(file.file)
            if expected_columns and not set(expected_columns).issubset(df.columns):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="CSV file does not contain the required columns."
                )
            save_to_cache(df, cache_path)
            return df, "uploaded"
        
        elif ext == "pkl":
            file.file.seek(0)
            model = joblib.load(file.file)
            save_to_cache(model, cache_path)
            return model, "uploaded"

    except Exception as e:
        logger.error(f"Error processing file: {file.filename}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the file."
        )
