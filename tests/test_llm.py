"""
test_llm.py — unit tests for the model-access wrapper.

No network here. The provider call (`llm.generate`) is monkeypatched to return
a canned reply, so what's actually under test is the layer we own:
  * schema instruction + JSON extraction + pydantic validation in `complete()`
  * the "malformed / non-conforming output raises" guardrail
  * the (backend, role) -> model-id mapping and the unknown-backend guard

Run:  pytest tests/test_llm.py -v
"""

import pytest  # fixtures + raises helper
from pydantic import BaseModel  # to define a tiny schema for the validation tests

import llm  # module under test (src/ is on sys.path via conftest.py)
from llm import LLMResult, LLMSchemaError, complete


# ---------------------------------------------------------------------------
# A minimal schema the "extract" role would realistically produce.
# ---------------------------------------------------------------------------
class CaregiverProfile(BaseModel):
    needs_description: str          # free-text care need
    budget: float | None = None    # S$/session ceiling, optional
    area: str | None = None        # preferred area, optional


def _canned(text: str) -> LLMResult:
    """Build an LLMResult with a fixed reply, standing in for a provider call."""
    return LLMResult(text=text, model="fake-model", backend="fake", prompt_tokens=1, completion_tokens=1)


# ---------------------------------------------------------------------------
# complete() — free-text mode (no schema)
# ---------------------------------------------------------------------------

def test_complete_returns_stripped_text_when_no_schema(monkeypatch):
    """Without a schema, complete() returns the model's reply as a trimmed string."""
    monkeypatch.setattr(llm, "generate", lambda *a, **k: _canned("  hello there \n"))
    out = complete([{"role": "user", "content": "hi"}], model_role="reasoning")
    assert out == "hello there"


# ---------------------------------------------------------------------------
# complete() — schema mode: happy paths
# ---------------------------------------------------------------------------

def test_complete_parses_and_validates_plain_json(monkeypatch):
    """A clean JSON object is parsed and returned as a populated model instance."""
    monkeypatch.setattr(
        llm, "generate",
        lambda *a, **k: _canned('{"needs_description": "day care for mum", "budget": 40}'),
    )
    out = complete([{"role": "user", "content": "..."}], model_role="extract", response_schema=CaregiverProfile)
    assert isinstance(out, CaregiverProfile)
    assert out.needs_description == "day care for mum"
    assert out.budget == 40
    assert out.area is None  # optional field defaulted


def test_complete_recovers_json_from_markdown_fence(monkeypatch):
    """Models often wrap JSON in ```json fences — that must still validate."""
    fenced = '```json\n{"needs_description": "night respite", "area": "Bishan"}\n```'
    monkeypatch.setattr(llm, "generate", lambda *a, **k: _canned(fenced))
    out = complete([{"role": "user", "content": "..."}], model_role="extract", response_schema=CaregiverProfile)
    assert out.area == "Bishan"


def test_complete_recovers_json_with_surrounding_prose(monkeypatch):
    """A stray sentence before/after the object is tolerated via the brace-span heuristic."""
    noisy = 'Sure! Here is the profile:\n{"needs_description": "weekend cover"}\nHope that helps.'
    monkeypatch.setattr(llm, "generate", lambda *a, **k: _canned(noisy))
    out = complete([{"role": "user", "content": "..."}], model_role="extract", response_schema=CaregiverProfile)
    assert out.needs_description == "weekend cover"


def test_complete_requests_json_mode_and_injects_schema(monkeypatch):
    """complete() must ask the provider for json_mode and prepend the schema contract."""
    seen = {}

    def spy(messages, **kwargs):
        seen["messages"] = messages
        seen["kwargs"] = kwargs
        return _canned('{"needs_description": "x"}')

    monkeypatch.setattr(llm, "generate", spy)
    complete([{"role": "user", "content": "hi"}], model_role="extract", response_schema=CaregiverProfile)

    assert seen["kwargs"]["json_mode"] is True                       # provider-level JSON constraint requested
    assert seen["messages"][0]["role"] == "system"                   # schema instruction prepended
    assert "JSON Schema" in seen["messages"][0]["content"]           # ...and it actually describes the contract
    assert seen["messages"][-1]["content"] == "hi"                   # original user turn stays last


# ---------------------------------------------------------------------------
# complete() — schema mode: the guardrail must fire
# ---------------------------------------------------------------------------

def test_complete_raises_when_output_is_not_json(monkeypatch):
    """Non-JSON text under a schema request is a hard error, never returned to the caller."""
    monkeypatch.setattr(llm, "generate", lambda *a, **k: _canned("I could not complete that."))
    with pytest.raises(LLMSchemaError):
        complete([{"role": "user", "content": "..."}], model_role="extract", response_schema=CaregiverProfile)


def test_complete_raises_when_json_violates_schema(monkeypatch):
    """Valid JSON that breaks the schema (missing required field) must raise, not pass through."""
    monkeypatch.setattr(llm, "generate", lambda *a, **k: _canned('{"budget": 40}'))  # no needs_description
    with pytest.raises(LLMSchemaError):
        complete([{"role": "user", "content": "..."}], model_role="extract", response_schema=CaregiverProfile)


def test_complete_raises_when_json_has_wrong_type(monkeypatch):
    """A field of the wrong type is a schema violation too."""
    monkeypatch.setattr(
        llm, "generate",
        lambda *a, **k: _canned('{"needs_description": "x", "budget": "not-a-number"}'),
    )
    with pytest.raises(LLMSchemaError):
        complete([{"role": "user", "content": "..."}], model_role="extract", response_schema=CaregiverProfile)


# ---------------------------------------------------------------------------
# _extract_json_object — the recovery helper, in isolation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', {"a": 1}),                                  # bare object
        ('```json\n{"a": 1}\n```', {"a": 1}),                    # fenced with language tag
        ('```\n{"a": 1}\n```', {"a": 1}),                        # fenced, no tag
        ('text {"a": 1} more text', {"a": 1}),                   # embedded in prose
    ],
)
def test_extract_json_object_recovers_common_shapes(raw, expected):
    assert llm._extract_json_object(raw) == expected


def test_extract_json_object_raises_on_no_object():
    with pytest.raises(LLMSchemaError):
        llm._extract_json_object("there is no json here at all")


# ---------------------------------------------------------------------------
# Model-id policy + backend guard
# ---------------------------------------------------------------------------

def test_model_id_for_maps_backend_and_role():
    """Each (backend, role) pair resolves to the id configured in settings."""
    assert llm._model_id_for("groq", "extract") == llm.settings.groq_extract_model
    assert llm._model_id_for("groq", "reasoning") == llm.settings.groq_reasoning_model
    assert llm._model_id_for("bedrock", "extract") == llm.settings.bedrock_extract_model
    assert llm._model_id_for("bedrock", "reasoning") == llm.settings.bedrock_reasoning_model


def test_generate_rejects_unknown_backend(monkeypatch):
    """An unrecognised LLM_BACKEND is a config error, not a silent default."""
    monkeypatch.setattr(llm.settings, "llm_backend", "openai")  # not a backend this module supports
    with pytest.raises(llm.LLMError):
        llm.generate([{"role": "user", "content": "hi"}], model_role="extract")


def test_groq_backend_errors_without_api_key(monkeypatch):
    """The groq path fails early with a clear message when the key is missing."""
    monkeypatch.setattr(llm.settings, "llm_backend", "groq")
    monkeypatch.setattr(llm.settings, "groq_api_key", "")
    with pytest.raises(llm.LLMBackendError):
        llm.generate([{"role": "user", "content": "hi"}], model_role="extract")
