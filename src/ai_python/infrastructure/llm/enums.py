from enum import Enum

class AIProvider(str, Enum):
    BEDROCK = "bedrock"
    GEMINI = "gemini"