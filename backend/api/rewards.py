from fastapi import APIRouter
router = APIRouter()

@router.post("/api/rewards/log")
def log_reward():
    return {}
