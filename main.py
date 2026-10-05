from dotenv import load_dotenv
from fastapi import FastAPI

from ai_python.presentation.routes import generate, health

load_dotenv()

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
