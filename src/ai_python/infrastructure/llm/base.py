from typing import Protocol


class AIClient(Protocol):

    async def generate(self, prompt: str) -> str:
        """Generate text based on the given prompt."""
        ...