"""
test_guardrails.py — unit tests for the three hard guardrails.

No network. Covers:
  * the iteration cap boundary (>= max_iterations, read from settings)
  * the tool allow-list is exactly {search_services, draft_message}
  * assert_tool_allowed raises on anything else
  * the source scan finds no send/book/call tool anywhere in src/

Run:  pytest tests/test_guardrails.py -v
"""

import pytest

import guardrails


# ---------------------------------------------------------------------------
# 1. Iteration cap
# ---------------------------------------------------------------------------

def test_iteration_cap_boundary(monkeypatch):
    """Cap trips at exactly max_iterations, not before; missing key counts as 0."""
    monkeypatch.setattr(guardrails.settings, "max_iterations", 3)  # small, explicit cap for the test

    assert guardrails.iteration_cap_reached({"iteration_count": 0}) is False
    assert guardrails.iteration_cap_reached({"iteration_count": 2}) is False
    assert guardrails.iteration_cap_reached({"iteration_count": 3}) is True   # boundary: >=
    assert guardrails.iteration_cap_reached({"iteration_count": 9}) is True
    assert guardrails.iteration_cap_reached({}) is False                       # absent -> 0


def test_iteration_cap_uses_live_settings(monkeypatch):
    """Changing settings.max_iterations changes the verdict (no value is baked in at import)."""
    state = {"iteration_count": 5}
    monkeypatch.setattr(guardrails.settings, "max_iterations", 6)
    assert guardrails.iteration_cap_reached(state) is False
    monkeypatch.setattr(guardrails.settings, "max_iterations", 5)
    assert guardrails.iteration_cap_reached(state) is True


# ---------------------------------------------------------------------------
# 2. Tool allow-list
# ---------------------------------------------------------------------------

def test_allow_list_is_exactly_search_and_draft():
    """The allow-list is a closed set of two read/draft tools — no sender."""
    assert guardrails.ALLOWED_TOOLS == frozenset({"search_services", "draft_message"})


def test_assert_tool_allowed_passes_for_listed_tools():
    guardrails.assert_tool_allowed("search_services")  # no raise
    guardrails.assert_tool_allowed("draft_message")    # no raise


@pytest.mark.parametrize(
    "bad_tool",
    ["send_message", "send_email", "book_service", "make_booking",
     "submit_enquiry", "call_provider", "reserve_slot", "enrol_client"],
)
def test_assert_tool_allowed_raises_for_action_tools(bad_tool):
    with pytest.raises(guardrails.GuardrailViolation):
        guardrails.assert_tool_allowed(bad_tool)


# ---------------------------------------------------------------------------
# 3. No action tool exists in the codebase
# ---------------------------------------------------------------------------

def test_no_send_or_book_tool_in_source():
    """
    Static scan of src/: there must be no top-level function whose name starts
    with an action verb (send_/book_/submit_/call_/...). This guards against a
    real-world-action tool being added later without an explicit design change.
    """
    hits = guardrails.scan_source_for_action_tools()
    assert hits == [], f"Action-capable tool definition(s) found in src/: {hits}"


def test_source_scan_would_catch_a_planted_sender(tmp_path):
    """Sanity-check the scanner itself: drop a fake sender in a temp dir and see it flagged."""
    (tmp_path / "rogue.py").write_text("def send_booking(service):\n    return 'sent'\n", encoding="utf-8")
    (tmp_path / "innocent.py").write_text("def search_services(profile):\n    return []\n", encoding="utf-8")

    hits = guardrails.scan_source_for_action_tools(tmp_path)
    assert len(hits) == 1
    assert "rogue.py" in hits[0] and "send_booking" in hits[0]
