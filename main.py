from fastapi import FastAPI

from ai_python.presentation.routes import generate, health

app = FastAPI()

app.include_router(health.router)
app.include_router(generate.router)
