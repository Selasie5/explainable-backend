from pydantic import BaseModel

class UploadFile(BaseModel):
    file:str
    file_name:str
    file_type:str
