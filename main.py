from dotenv import load_dotenv
from fastapi import FastAPI

from ai_python.presentation.routes import generate, health

load_dotenv()


app = FastAPI()

app.include_router(health.router)
app.include_router(generate.router)
