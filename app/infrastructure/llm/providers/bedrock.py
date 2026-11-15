import asyncio
import json
import logging
import os

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from langfuse import get_client

logger = logging.getLogger(__name__)

_MODEL_ID = "amazon.nova-micro-v1:0"

MODEL_NAME = "amazon-nova-micro-v1"

# Tool the model must call when the caller asks for JSON (see _invoke)
_JSON_TOOL = "return_json"

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
        self.model = model_id
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

    async def generate(self, prompt: str, json_schema: dict | None = None) -> str:
        """
        Gọi Bedrock Converse API bất đồng bộ.
        boto3 là sync — dùng asyncio.to_thread để chạy trên thread pool
        mà không block event loop của FastAPI.
        """
        # Unlike Gemini, boto3 has no Langfuse instrumentation, so the call is traced here
        with get_client().start_as_current_observation(
            as_type="generation",
            name="bedrock-converse",
            model=self.model,
            model_parameters=self._inference_config,
            input=prompt,
        ) as generation:
            text, usage = await asyncio.to_thread(self._invoke, prompt, json_schema)
            generation.update(
                output=text,
                usage_details={"input": usage.get("inputTokens", 0), "output": usage.get("outputTokens", 0)},
            )
        return text

    # ── Private ──────────────────────────────────────────────────────────────

    def _invoke(self, prompt: str, json_schema: dict | None) -> tuple[str, dict]:
        """Gọi Converse API (sync) — chạy trong thread pool. Trả về text và số token đã dùng."""
        request = {
            "modelId": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            "inferenceConfig": self._inference_config,
        }
        if json_schema is not None:
            # Forcing a tool call makes Bedrock return the tool input as parsed JSON:
            # written as plain text, Nova Micro sometimes drops the commas between fields
            request["toolConfig"] = {
                "tools": [{"toolSpec": {
                    "name": _JSON_TOOL,
                    "description": "Return the answer",
                    "inputSchema": {"json": json_schema},
                }}],
                "toolChoice": {"tool": {"name": _JSON_TOOL}},
            }

        response = self._client.converse(**request)
        content = response["output"]["message"]["content"]
        usage = response.get("usage", {})

        if json_schema is not None:
            tool_input = next((block["toolUse"]["input"] for block in content if "toolUse" in block), None)
            # No tool call (e.g. the answer hit maxTokens): "" fails JSON parsing, so the caller retries
            return (json.dumps(tool_input, ensure_ascii=False) if tool_input is not None else ""), usage

        # Converse API trả về: output.message.content[0].text
        return content[0]["text"], usage
