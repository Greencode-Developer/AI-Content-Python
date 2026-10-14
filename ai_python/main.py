from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from langfuse import get_client
from openinference.instrumentation.google_genai import (
    GoogleGenAIInstrumentor,
)

from ai_python.api import health, generate


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

API_PREFIX = "/api/v1"

app.include_router(health.router, prefix=API_PREFIX)
app.include_router(generate.router, prefix=API_PREFIX)