"""
AWS Bedrock provider — Amazon Nova Micro (model rẻ nhất trên Bedrock).

Giá (on-demand, us-east-1):
  Input : $0.000035 / 1K tokens  (~$0.035 / 1M tokens)
  Output: $0.000140 / 1K tokens  (~$0.140 / 1M tokens)

Auth (theo thứ tự ưu tiên của boto3):
  1. Biến môi trường: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
  2. ~/.aws/credentials  (aws configure)
  3. IAM Role gắn trên EC2/Lambda/ECS (zero-credential khi deploy trên AWS — tiết kiệm nhất)

Region mặc định: us-east-1 (Nova Micro available, giá thấp nhất).
Có thể override qua biến môi trường AWS_BEDROCK_REGION.

boto3 chỉ hỗ trợ sync I/O — ta wrap bằng asyncio.to_thread để không block event loop.
"""

import asyncio
import logging
import os

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Model ID rẻ nhất hiện có trên Amazon Bedrock (Nova family)
_MODEL_ID = "amazon.nova-micro-v1:0"

# Tên model hiển thị để ghi vào trường ai_model_used
MODEL_NAME = "amazon-nova-micro-v1"

# Retry tự động khi gặp throttling (Bedrock on-demand có rate limit thấp hơn Provisioned)
_BOTO_CONFIG = Config(
    retries={"max_attempts": 3, "mode": "adaptive"},
)


class BedrockAIClient:
    """AIClient implementation dùng AWS Bedrock Converse API."""

    def __init__(
        self,
        model_id: str = _MODEL_ID,
        temperature: float = 0.7,
        # Claude 4.5+ rejects a request that sets both temperature and topP: pass None for Claude
        top_p: float | None = 0.9,
        max_tokens: int = 2048,
    ) -> None:
        self._model_id = model_id
        self._inference_config = {"maxTokens": max_tokens, "temperature": temperature}
        if top_p is not None:
            self._inference_config["topP"] = top_p

        region = os.getenv("AWS_BEDROCK_REGION", "us-east-1")
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=_BOTO_CONFIG,
        )

    # ── Public ───────────────────────────────────────────────────────────────

    async def generate(self, prompt: str) -> str:
        """
        Gọi Bedrock Converse API bất đồng bộ.
        boto3 là sync — dùng asyncio.to_thread để chạy trên thread pool
        mà không block event loop của FastAPI.
        """
        return await asyncio.to_thread(self._invoke, prompt)

    # ── Private ──────────────────────────────────────────────────────────────

    def _invoke(self, prompt: str) -> str:
        """Gọi Converse API (sync) — chạy trong thread pool."""
        response = self._client.converse(
            modelId=self._model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            inferenceConfig=self._inference_config,
        )

        # Converse API trả về: output.message.content[0].text
        return response["output"]["message"]["content"][0]["text"]
