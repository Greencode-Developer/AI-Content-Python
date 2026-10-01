from fastapi import FastAPI
from src.ai_python.api.health import router

app = FastAPI()
app.include_router(router)