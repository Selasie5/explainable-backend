from fastapi import UploadFile, HTTPException,status
import os


#Define allowed extensions and MIME type
ALLOWED_EXTENSTIONS = {
    "csv": ["text/csv","application/vnd.ms-excel"],
    "pkl": ["application/x-pickle", "application/octet-stream"]
}

def validate_file_type(file:UploadFile):
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower().strip(".")


    
  #Check MIME type for the file 
    content_type = file.content_type
    print(f"DEBUG — filename: {filename}, extension: {ext}, content_type: {content_type}")

    #Check the vaildidity  of the file extension
    if ext not in ALLOWED_EXTENSTIONS:
        raise HTTPException(
            status_code= status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension: {ext}. Allowed extensions are {list(ALLOWED_EXTENSTIONS.keys())}"
        )

  #Check the validity of the MIME type
    if content_type not in ALLOWED_EXTENSTIONS[ext]:
        raise HTTPException(
            status_code= status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension: {ext}. Allowed extensions are {list(ALLOWED_EXTENSTIONS.keys())}"
        )
    
    return ext
