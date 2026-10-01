from fastapi import FastAPI

from src.ai_python.presentation.routes import health, generate

app = FastAPI()

app.include_router(health.router)
# app.include_router(generate.router)