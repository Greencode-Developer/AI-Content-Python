import os

# Tests must not send traces: importing app.main loads the real Langfuse keys from .env.
# Set before any app module creates the Langfuse client; load_dotenv does not override it
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
