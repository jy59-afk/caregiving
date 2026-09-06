"""
check_env.py — model-access smoke test.

Confirms that the LLM wrapper in `llm.py` can actually reach the configured
provider: prints which backend/model is active, sends one trivial completion,
prints the reply and token usage, and exits non-zero on any failure so it can
gate a demo setup or CI step.

Run from the repo root:  python src/check_env.py
"""

import sys  # for the process exit code

# Some Groq/Bedrock models reply with characters the legacy Windows console
# codepage can't encode; replace them rather than crash the smoke test on print.
try:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass  # non-reconfigurable stream (e.g. piped) — nothing to do

from config import settings  # to report the active configuration before we call out
from llm import LLMError, generate  # low-level call returns model/backend/usage we want to show


def main() -> int:
    """Return 0 on success, non-zero on failure (used as the process exit code)."""

    # --- 1. Report the configuration the wrapper will use -------------------
    backend = settings.llm_backend.lower()                       # "groq" or "bedrock"
    print(f"LLM_BACKEND        : {settings.llm_backend}")
    if backend == "groq":
        print(f"groq extract model : {settings.groq_extract_model}")
        print(f"groq reasoning     : {settings.groq_reasoning_model}")
        print(f"GROQ_API_KEY set   : {bool(settings.groq_api_key)}")   # never print the key itself
    elif backend == "bedrock":
        print(f"bedrock extract    : {settings.bedrock_extract_model}")
        print(f"bedrock reasoning  : {settings.bedrock_reasoning_model}")
        print(f"AWS_REGION         : {settings.aws_region}")
    else:
        print(f"ERROR: unknown LLM_BACKEND {settings.llm_backend!r} (expected 'groq' or 'bedrock')")
        return 1

    # --- 2. Send one trivial completion on the cheap ("extract") model -----
    print("\nsending one test completion (model_role='extract')...")
    try:
        result = generate(
            [{"role": "user", "content": "Reply with exactly the word: pong"}],  # deterministic, tiny answer
            model_role="extract",                                                 # exercise the cheap model path
            max_tokens=256,  # generous: reasoning-style models (gpt-oss, qwen) spend hidden tokens before the answer
        )
    except LLMError as exc:                                     # our wrapped, provider-agnostic failure
        print(f"FAILED: {exc}")
        return 1
    except Exception as exc:                                    # anything unwrapped is still a failure to surface
        print(f"FAILED (unexpected): {exc!r}")
        return 1

    # --- 3. Show the reply + token accounting -----------------------------
    print(f"\nserved by         : {result.backend} / {result.model}")
    print(f"reply             : {result.text.strip()!r}")
    print(f"prompt tokens     : {result.prompt_tokens}")
    print(f"completion tokens : {result.completion_tokens}")

    if not result.text.strip():                                # a blank reply means something is wrong upstream
        print("FAILED: model returned an empty reply")
        return 1

    print("\nOK - model access is working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # propagate the status so `python src/check_env.py && ...` behaves
