from fastapi import FastAPI
# from routers.upload import UploadFileRouter
from routers.analyze import AnalyzeRouter
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

@app.get("/")
async def root():
    return {"message":"Hello World"}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# app.include_router(UploadFileRouter, prefix="/api/v1", tags=["upload"])
app.include_router(AnalyzeRouter, prefix="/api/v1", tags=["analyze"])
    