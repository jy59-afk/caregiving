"""
llm.py — the single seam through which every orchestration node talks to a
large language model.

Why this module exists:
  * Provider-agnostic. Nodes call `complete(...)` / `generate(...)` and never
    import `groq` or `boto3` themselves, so switching Groq -> Bedrock is a
    change to `LLM_BACKEND` in `.env`, nothing else.
  * One place for the "every model output is schema-validated before use"
    guardrail (CLAUDE.md). If a caller passes `response_schema`, the raw text
    is parsed and validated here and a hard error is raised on any mismatch —
    so no node can accidentally consume unvalidated model output.
  * `model_role` ("extract" vs "reasoning") maps to the cheap vs strong model
    id per backend, keeping model-selection policy out of the nodes.

Provider SDKs (`groq`, `boto3`) are imported lazily inside the dispatch branch
that needs them — mirroring `ingest.get_embeddings` — so importing this module
never requires a provider to be installed or configured.

Run the smoke test with:  python src/check_env.py
"""

from __future__ import annotations  # allow the `BaseModel | str` style hints on older 3.x if ever needed

import json  # to coerce model text into a dict before schema validation
from dataclasses import dataclass, field  # lightweight typed container for a raw completion
from typing import Any, Literal  # precise types for the public signatures

from pydantic import BaseModel, ValidationError  # schema container + the error we translate into LLMSchemaError

from config import settings  # typed view of .env — backend switch, model ids, keys, region

# Roles a caller can ask for. "extract" -> cheap/fast model (intake parsing,
# classification); "reasoning" -> stronger model (ranking rationale, drafting).
ModelRole = Literal["extract", "reasoning"]


class LLMError(RuntimeError):
    """Base class for every failure originating in this module."""


class LLMBackendError(LLMError):
    """The configured provider could not be reached or returned a transport-level error."""


class LLMSchemaError(LLMError):
    """
    The model replied, but its output could not be parsed as JSON or did not
    conform to the requested pydantic schema. This is the guardrail firing:
    the caller must never see or use the offending text.
    """


@dataclass
class LLMResult:
    """
    One raw completion plus the metadata the observability layer and the
    smoke test want (token usage, which model/backend actually served it).
    `generate()` returns this; `complete()` unwraps it.
    """

    text: str                                   # the model's reply, verbatim
    model: str                                  # concrete model id that produced `text`
    backend: str                                # "groq" or "bedrock"
    prompt_tokens: int = 0                       # tokens billed for the input
    completion_tokens: int = 0                   # tokens billed for the output
    raw_usage: dict[str, Any] = field(default_factory=dict)  # provider-native usage block, for logging


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def generate(
    messages: list[dict[str, str]],
    *,
    model_role: ModelRole,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> LLMResult:
    """
    Low-level entry point: send `messages` to the role-appropriate model and
    return the raw reply + usage. No schema validation here — callers that
    need a validated object use `complete(..., response_schema=...)`.

    messages: a list of {"role": "system"|"user"|"assistant", "content": str}
              in the usual chat-completions shape. The Bedrock branch splits
              out the system turns itself.
    json_mode: hint the provider to emit a single JSON object (used by
               `complete()` when a schema is supplied).
    """
    backend = settings.llm_backend.lower()  # normalise so "Groq"/"BEDROCK" both work

    if backend == "groq":                                   # dev / fast-iteration path
        return _generate_groq(
            messages,
            model=_model_id_for("groq", model_role),        # cheap vs strong groq model
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )

    if backend == "bedrock":                                # demo / deploy path
        return _generate_bedrock(
            messages,
            model=_model_id_for("bedrock", model_role),     # cheap vs strong bedrock model
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )

    # Any other value is a config mistake — fail loudly rather than silently
    # picking a default the operator did not choose.
    raise LLMError(f"Unknown LLM_BACKEND {settings.llm_backend!r}; expected 'groq' or 'bedrock'")


def complete(
    messages: list[dict[str, str]],
    *,
    model_role: ModelRole,
    response_schema: type[BaseModel] | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> BaseModel | str:
    """
    High-level entry point used by the orchestration nodes.

    * `response_schema is None`  -> returns the model's reply as a `str`.
    * `response_schema` given    -> instructs the model to answer with JSON,
      parses it, validates it against the schema, and returns the populated
      model instance. Any parse/validation failure raises `LLMSchemaError` —
      this is the "every model output validated before use" guardrail, and it
      lives here so every node inherits it for free.
    """
    if response_schema is None:                              # plain free-text completion
        return generate(
            messages,
            model_role=model_role,
            max_tokens=max_tokens,
            temperature=temperature,
        ).text.strip()

    # Schema requested: append a system instruction describing the exact JSON
    # contract, then ask the provider for JSON-object output as well.
    schema_messages = _with_schema_instruction(messages, response_schema)

    result = generate(
        schema_messages,
        model_role=model_role,
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=True,                                     # provider-level "return one JSON object"
    )

    payload = _extract_json_object(result.text)             # tolerate ```json fences / stray prose
    try:
        return response_schema.model_validate(payload)      # pydantic enforces the contract
    except ValidationError as exc:                          # guardrail trips — caller must not see result.text
        raise LLMSchemaError(
            f"{response_schema.__name__} validation failed for model output: {exc}"
        ) from exc


# --------------------------------------------------------------------------
# Model-id policy
# --------------------------------------------------------------------------

def _model_id_for(backend: str, role: ModelRole) -> str:
    """Map (backend, role) to the concrete model id configured in .env."""
    table = {
        ("groq", "extract"): settings.groq_extract_model,        # llama-3.1-8b-instant by default
        ("groq", "reasoning"): settings.groq_reasoning_model,    # llama-3.3-70b-versatile by default
        ("bedrock", "extract"): settings.bedrock_extract_model,  # Claude Haiku 4.5
        ("bedrock", "reasoning"): settings.bedrock_reasoning_model,  # Claude Sonnet 4.5
    }
    try:
        return table[(backend, role)]                            # exact match on the (backend, role) pair
    except KeyError:                                             # unreachable via the public API, but explicit
        raise LLMError(f"No model id configured for backend={backend!r} role={role!r}")


# --------------------------------------------------------------------------
# Schema-prompting helpers
# --------------------------------------------------------------------------

def _with_schema_instruction(
    messages: list[dict[str, str]],
    schema: type[BaseModel],
) -> list[dict[str, str]]:
    """
    Return a copy of `messages` with an extra system turn that spells out the
    JSON schema the reply must satisfy. Kept as plain text (not a provider
    "tools" call) so the exact same prompt works on Groq and Bedrock.
    """
    contract = json.dumps(schema.model_json_schema(), indent=2)  # the authoritative shape, straight from pydantic
    instruction = (
        "You must reply with a single JSON object and nothing else — no prose, "
        "no markdown fences. The object must validate against this JSON Schema:\n"
        f"{contract}"
    )
    # Prepend rather than append so a later user turn is the last thing the model reads.
    return [{"role": "system", "content": instruction}, *messages]


def _extract_json_object(text: str) -> Any:
    """
    Best-effort recovery of the JSON object from a model reply. Handles the
    common cases where a model wraps JSON in ```json ... ``` fences or adds a
    sentence before/after. Raises `LLMSchemaError` if nothing parseable is found.
    """
    candidate = text.strip()  # trim surrounding whitespace/newlines

    if candidate.startswith("```"):                     # strip a leading ```json / ``` fence
        candidate = candidate.split("```", 2)[1]        # take the content between the first pair of fences
        if candidate.lstrip().lower().startswith("json"):
            candidate = candidate.lstrip()[4:]          # drop the language tag

    candidate = candidate.strip().strip("`").strip()   # clean up any trailing fence remnants

    try:
        return json.loads(candidate)                    # fast path — the whole reply is valid JSON
    except json.JSONDecodeError:
        pass                                            # fall through to the substring heuristic

    start = candidate.find("{")                         # first brace
    end = candidate.rfind("}")                          # last brace
    if start != -1 and end != -1 and end > start:       # plausible object span
        try:
            return json.loads(candidate[start : end + 1])  # parse just that span
        except json.JSONDecodeError as exc:
            raise LLMSchemaError(f"Model output was not valid JSON: {exc}") from exc

    raise LLMSchemaError("Model output contained no JSON object")


# --------------------------------------------------------------------------
# Groq backend (dev)
# --------------------------------------------------------------------------

def _generate_groq(
    messages: list[dict[str, str]],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
) -> LLMResult:
    """Call Groq's OpenAI-compatible chat-completions endpoint."""
    if not settings.groq_api_key:                        # fail early with a clear message, not a 401 later
        raise LLMBackendError("LLM_BACKEND=groq but GROQ_API_KEY is empty — add it to .env")

    from groq import Groq  # lazy import: only needed on the groq path

    client = Groq(api_key=settings.groq_api_key)         # thin HTTP client, cheap to construct per call at our volume

    kwargs: dict[str, Any] = {                           # assemble the request
        "model": model,
        "messages": messages,                            # already in chat-completions shape
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:                                        # ask Groq to constrain output to one JSON object
        kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat.completions.create(**kwargs)  # the actual network call
    except Exception as exc:                             # groq raises its own exception hierarchy; wrap it
        raise LLMBackendError(f"Groq request failed: {exc}") from exc

    choice = resp.choices[0]                             # we never request n>1
    usage = resp.usage                                   # token accounting for logging / the smoke test

    return LLMResult(
        text=choice.message.content or "",               # content is None only on tool calls, which we don't use
        model=model,
        backend="groq",
        prompt_tokens=getattr(usage, "prompt_tokens", 0),
        completion_tokens=getattr(usage, "completion_tokens", 0),
        raw_usage=usage.model_dump() if hasattr(usage, "model_dump") else dict(usage or {}),
    )


# --------------------------------------------------------------------------
# Bedrock backend (demo / deploy) — Converse API via boto3
# --------------------------------------------------------------------------

def _generate_bedrock(
    messages: list[dict[str, str]],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
) -> LLMResult:
    """
    Call Bedrock's provider-agnostic Converse API. Converse wants system turns
    passed separately from the user/assistant transcript, and each message's
    content wrapped as a list of content blocks — so we translate here.
    """
    import boto3  # lazy import: only needed on the bedrock path

    system_blocks, turns = _split_for_converse(messages)  # (system=[{"text":...}], messages=[{"role","content":[...]}])

    inference_config: dict[str, Any] = {                  # Converse's knobs live in one nested dict
        "maxTokens": max_tokens,
        "temperature": temperature,
    }

    # json_mode: Converse has no universal "JSON object" switch across model
    # families, so we lean on the schema instruction already added by
    # complete(). Nudge it once more here for models that honour a system hint.
    if json_mode:
        system_blocks = [*system_blocks, {"text": "Respond with only a single valid JSON object."}]

    try:
        client = boto3.client("bedrock-runtime", region_name=settings.aws_region)  # region must match the model's
        resp = client.converse(
            modelId=model,
            messages=turns,
            system=system_blocks or [{"text": "You are a helpful assistant."}],  # Converse rejects an empty system list on some models
            inferenceConfig=inference_config,
        )
    except Exception as exc:                              # botocore raises ClientError etc.; wrap for callers
        raise LLMBackendError(f"Bedrock Converse request failed: {exc}") from exc

    # Converse response shape: output.message.content is a list of blocks; the
    # text answer is the first (and, for us, only) {"text": ...} block.
    content_blocks = resp["output"]["message"]["content"]
    text = next((b["text"] for b in content_blocks if "text" in b), "")

    usage = resp.get("usage", {})                         # {"inputTokens", "outputTokens", "totalTokens"}
    return LLMResult(
        text=text,
        model=model,
        backend="bedrock",
        prompt_tokens=usage.get("inputTokens", 0),
        completion_tokens=usage.get("outputTokens", 0),
        raw_usage=dict(usage),
    )


def _split_for_converse(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """
    Translate chat-completions messages into Converse's two-part shape:
      * system turns  -> [{"text": "..."}]
      * user/assistant -> [{"role": "...", "content": [{"text": "..."}]}]
    Consecutive same-role turns are left as-is; Converse accepts them.
    """
    system_blocks: list[dict[str, str]] = []             # collects every system turn's text
    turns: list[dict[str, Any]] = []                     # the user/assistant transcript

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":                             # Converse wants these out-of-band
            system_blocks.append({"text": content})
        else:                                            # "user" / "assistant"
            turns.append({"role": role, "content": [{"text": content}]})

    return system_blocks, turns
