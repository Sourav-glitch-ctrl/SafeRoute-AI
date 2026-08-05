from fastapi import FastAPI
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    debug=settings.DEBUG
)


@app.get("/")
def root():
    return {
        "message": "Welcome to SafeRoute AI API"
    }