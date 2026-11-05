import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langfuse import get_client
from openinference.instrumentation.google_genai import (
    GoogleGenAIInstrumentor,
)

from app.api import content, health, ideas
from app.core.exception.exception import InvalidLLMOutputError

logger = logging.getLogger(__name__)

#Google GenAI SDK
GoogleGenAIInstrumentor().instrument()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Tracing is optional: a Langfuse outage must not stop idea generation
    try:
        if not get_client().auth_check():
            logger.warning("Langfuse authentication failed, traces will not be sent")
    except Exception:
        logger.warning("Langfuse is unreachable, traces will not be sent", exc_info=True)
    yield


app = FastAPI(
    title="AI Content Platform",
    description=(
        "API hỗ trợ chủ shop nhỏ sản xuất nội dung mạng xã hội bằng AI. "
        "Scope hiện tại: Facebook (fanpage), sinh ý tưởng và bài đăng."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.exception_handler(InvalidLLMOutputError)
async def invalid_llm_output_handler(_: Request, exc: InvalidLLMOutputError) -> JSONResponse:
    # 502: the request was fine, the upstream model returned unusable output
    return JSONResponse(status_code=502, content={"detail": str(exc)})


app.include_router(health.router)
app.include_router(content.router)
app.include_router(ideas.router)
