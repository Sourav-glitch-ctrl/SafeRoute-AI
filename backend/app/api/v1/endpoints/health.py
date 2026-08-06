from fastapi import APIRouter
router = APIRouter()

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "SafeRoute AI",
        "version": "1.0.0"
    }