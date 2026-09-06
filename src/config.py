"""
config.py — single place that reads environment configuration into a typed
object, so no other module has to touch os.getenv() directly.

Import and use:  from config import settings
"""

import os  # for reading raw environment variables
from functools import lru_cache  # to build the settings object exactly once per process
from pathlib import Path  # for repo-root-anchored paths

from dotenv import load_dotenv  # loads key=value pairs from the local .env file
from pydantic import BaseModel, Field  # typed, validated settings container

load_dotenv()  # populate os.environ from .env before we read anything below

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../hackathon — this file is in src/
DATA_DIR = PROJECT_ROOT / "data"                    # curated dataset + built vector index live here


class Settings(BaseModel):
    """Validated view of the environment. Constructed once via get_settings()."""

    # --- LLM provider ---------------------------------------------------------
    llm_backend: str = Field(default="groq")            # "groq" for dev, "bedrock" for the demo build
    groq_api_key: str = Field(default="")               # required when llm_backend == "groq"
    groq_extract_model: str = Field(default="llama-3.1-8b-instant")      # cheap model for intake extraction / classification
    groq_reasoning_model: str = Field(default="llama-3.3-70b-versatile")  # stronger model for ranking rationale / message drafting

    # --- AWS Bedrock --------------------------------------------------------
    aws_region: str = Field(default="us-east-1")        # region for Bedrock Converse + Titan embeddings
    bedrock_extract_model: str = Field(default="anthropic.claude-haiku-4-5-20251001-v1:0")     # Haiku 4.5 — extraction/classification
    bedrock_reasoning_model: str = Field(default="anthropic.claude-sonnet-4-5-20250929-v1:0")  # Sonnet 4.5 — nuance/drafting

    # --- Retrieval --------------------------------------------------------
    embedding_backend: str = Field(default="local")    # "local" (MiniLM) or "bedrock" (Titan V2)

    # --- Guardrails -----------------------------------------------------
    max_iterations: int = Field(default=6, ge=1)       # hard cap on orchestration loop iterations, enforced from state

    # --- Paths (not from env — derived) --------------------------------
    services_path: Path = Field(default=DATA_DIR / "services.json")   # the curated dataset the index is built from
    index_path: Path = Field(default=DATA_DIR / "vector_index")       # persisted FAISS index location


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read the environment once and return a cached, validated Settings object."""
    return Settings(
        llm_backend=os.getenv("LLM_BACKEND", "groq"),                       # provider switch
        groq_api_key=os.getenv("GROQ_API_KEY", ""),                          # dev provider key
        groq_extract_model=os.getenv("GROQ_EXTRACT_MODEL", "llama-3.1-8b-instant"),
        groq_reasoning_model=os.getenv("GROQ_REASONING_MODEL", "llama-3.3-70b-versatile"),
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        bedrock_extract_model=os.getenv("BEDROCK_EXTRACT_MODEL", "anthropic.claude-haiku-4-5-20251001-v1:0"),
        bedrock_reasoning_model=os.getenv("BEDROCK_REASONING_MODEL", "anthropic.claude-sonnet-4-5-20250929-v1:0"),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "local"),
        max_iterations=int(os.getenv("MAX_ITERATIONS", "6")),                # str -> int; Settings validates it is >= 1
    )


settings = get_settings()  # module-level convenience handle for the common `from config import settings` import
