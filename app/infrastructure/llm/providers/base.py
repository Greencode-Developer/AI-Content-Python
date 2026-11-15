from typing import Protocol


class AIClient(Protocol):
    # Id of the model the provider calls (e.g. "amazon.nova-micro-v1:0"), returned to the caller
    model: str

    async def generate(self, prompt: str, json_schema: dict | None = None) -> str:
        """Generate text based on the given prompt.

        With json_schema, the provider is forced to return a JSON object with that structure.
        """
        ...