from fastapi import File,UploadFile,Form,Body,Response,FastAPI, BackgroundTasks,APIRouter,status,Depends

api = APIRouter()

@api.get("/health")
def health_check():
    return {"status": "ok"}
