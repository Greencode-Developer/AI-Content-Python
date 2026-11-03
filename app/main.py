from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from langfuse import get_client
from openinference.instrumentation.google_genai import (
    GoogleGenAIInstrumentor,
)

from app.api import content, health, ideas


# Langfuse
langfuse = get_client()
assert langfuse.auth_check(), "Langfuse authentication failed"

#Google GenAI SDK
GoogleGenAIInstrumentor().instrument()


app = FastAPI(
    title="AI Content Platform",
    description=(
        "API hỗ trợ chủ shop nhỏ sản xuất nội dung mạng xã hội bằng AI. "
        "Scope hiện tại: Facebook (fanpage), sinh ý tưởng và bài đăng."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


app.include_router(health.router)
app.include_router(content.router)
app.include_router(ideas.router)