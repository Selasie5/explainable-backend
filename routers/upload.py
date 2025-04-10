from fastapi import APIRouter, UploadFile, File, HTTPException, status
from utils.file_checker import validate_file_type

UploadFileRouter = APIRouter()
@UploadFileRouter.post("/upload-dataset")
async def upload_file(file: UploadFile = File(...)):
    try:
        ext = validate_file_type(file)
        return {"filename": file.filename, "content_type": file.content_type, "extension": ext}
    except HTTPException as e:
        raise e
