from fastapi import APIRouter
router = APIRouter()

@router.post("/api/chat")
def chat(payload: dict):
    return {"reply": ""}
