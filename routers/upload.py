from typing import List  
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from utils.file_checker import validate_file_type

UploadFileRouter = APIRouter()

@UploadFileRouter.post("/upload-dataset")
async def upload_file(files: List[UploadFile] = File(...)):  
    results = []
    for file in files:
        try:
            ext = validate_file_type(file)
            results.append({
                "filename": file.filename,
                "content_type": file.content_type,
                "extension": ext
            })
        except HTTPException as e:
            raise e
    return results
