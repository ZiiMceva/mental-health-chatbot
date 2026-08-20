from fastapi import APIRouter
router = APIRouter()

@router.post("/api/games/recommend")
def recommend():
    return {}

@router.post("/api/games/complete")
def complete():
    return {}
