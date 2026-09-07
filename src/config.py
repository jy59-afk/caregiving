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
    groq_extract_model: str = Field(default="openai/gpt-oss-20b")    # cheap/fast model for intake extraction / classification
    groq_reasoning_model: str = Field(default="openai/gpt-oss-120b")  # stronger model for ranking rationale / message drafting

    # --- AWS Bedrock --------------------------------------------------------
    # Demo runs in us-east-1: Claude 4.5 is reachable via the `us.` cross-region
    # inference profiles (the org SCP only blocks `ap-northeast-1` / `global.`),
    # there is no meaningful rate limit, and Titan embeddings are available.
    # Claude 4.x/4.5 CANNOT be invoked with the bare `anthropic.` id — the `us.`
    # (or `apac.`/`global.`) profile prefix is required for on-demand.
    aws_region: str = Field(default="us-east-1")        # region for Bedrock Converse + embeddings
    bedrock_extract_model: str = Field(default="us.anthropic.claude-haiku-4-5-20251001-v1:0")     # Haiku 4.5 — intake extraction
    bedrock_reasoning_model: str = Field(default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")  # Sonnet 4.5 — ranking rationale / drafting
    bedrock_embed_model: str = Field(default="cohere.embed-english-v3")                           # 1024-dim; recall@3 0.80 vs Titan's 0.50 on this corpus

    # --- Retrieval --------------------------------------------------------
    embedding_backend: str = Field(default="local")    # "local" (MiniLM, 384-dim) or "bedrock" (Cohere Embed v3, 1024-dim)

    # --- Guardrails -----------------------------------------------------
    max_iterations: int = Field(default=6, ge=1)       # hard cap on orchestration loop iterations, enforced from state

    # --- Model-role tuning --------------------------------------------
    # The Explain & Draft node uses the strong ("reasoning") model by default.
    # Set DRAFT_MODEL_ROLE=extract to run drafting on the cheap/fast model
    # instead — useful if a backend rate-limits the strong model, leaving only
    # the rerank on it. Not needed on us-east-1 Bedrock (no meaningful limit).
    draft_model_role: str = Field(default="reasoning")  # "reasoning" | "extract"

    # --- Paths (not from env — derived) --------------------------------
    services_path: Path = Field(default=DATA_DIR / "services.json")   # the curated dataset the index is built from
    index_path: Path = Field(default=DATA_DIR / "vector_index")       # persisted FAISS index location


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read the environment once and return a cached, validated Settings object."""
    return Settings(
        llm_backend=os.getenv("LLM_BACKEND", "groq"),                       # provider switch
        groq_api_key=os.getenv("GROQ_API_KEY", ""),                          # dev provider key
        groq_extract_model=os.getenv("GROQ_EXTRACT_MODEL", "openai/gpt-oss-20b"),
        groq_reasoning_model=os.getenv("GROQ_REASONING_MODEL", "openai/gpt-oss-120b"),
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        bedrock_extract_model=os.getenv("BEDROCK_EXTRACT_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
        bedrock_reasoning_model=os.getenv("BEDROCK_REASONING_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
        bedrock_embed_model=os.getenv("BEDROCK_EMBED_MODEL", "cohere.embed-english-v3"),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "local"),
        max_iterations=int(os.getenv("MAX_ITERATIONS", "6")),                # str -> int; Settings validates it is >= 1
        draft_model_role=os.getenv("DRAFT_MODEL_ROLE", "reasoning"),         # "extract" to keep drafting off a rate-limited Sonnet
    )


settings = get_settings()  # module-level convenience handle for the common `from config import settings` import
