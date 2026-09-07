"""
test_api.py — unit tests for the FastAPI surface (src/api.py).

No network, no FAISS index, no model. `graph.run` is monkeypatched to a canned
final state, and the index pre-flight is stubbed True. Under test:
  * the enrichment join: RankedMatch -> MatchOut with photo + coordinates,
    care_type / area / cost pulled from data/services.json by name
  * candidate_matches -> is_fallback False; fallback_matches -> is_fallback True
  * the draft is passed through only when present
  * the index pre-flight returns an actionable 503
  * an LLMError from the pipeline becomes a 502, not a stack trace

Run:  pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

import api
import llm
from state import CaregiverProfile, DraftedMessage, RankedMatch

# A real service name from data/services.json, so the enrichment join resolves
# to a real record (care_type / area / cost).
_REAL_NAME = "New Horizon Centre (Toa Payoh) - Dementia Singapore"


@pytest.fixture
def client(monkeypatch):
    """A TestClient with the index pre-flight satisfied."""
    monkeypatch.setattr(api, "_index_ready", lambda: True)
    return TestClient(api.app)


def _final_state(*, matches=None, fallback=None, draft=None, outcome="matches_ready"):
    """Build a canned graph.run() return value."""
    final_response = "Here are the respite options that best fit what you described:\n1. Some Centre — fits."
    if draft is not None:  # the Output node inlines the whole draft into final_response
        final_response += (
            f"\n\nI've drafted an enquiry to {draft.service_name} for you to review, "
            f"edit and send yourself:\n\nSubject: {draft.subject}\n\n{draft.body}"
        )
    return {
        "caregiver_profile": CaregiverProfile(
            needs_description="mother with Alzheimer's, afternoon agitation, weekday daytime cover",
            budget=85, area="Toa Payoh", relationship="mother",
        ),
        "candidate_matches": matches or [],
        "fallback_matches": fallback or [],
        "drafted_message": draft,
        "outcome": outcome,
        "final_response": final_response,
    }


def test_matches_are_enriched_with_photo_and_coordinates(client, monkeypatch):
    match = RankedMatch(
        name=_REAL_NAME,
        why="Dementia-specific weekday day care in Toa Payoh, built around memory loss.",
        grounding_passage="New Horizon Centre ... dementia-specific day care ...",
        similarity_score=0.21,
    )
    monkeypatch.setattr(api.graph, "run", lambda messages, **kw: _final_state(matches=[match]))

    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "weekday dementia care toa payoh under $85"}]})
    assert r.status_code == 200
    body = r.json()

    assert body["outcome"] == "matches_ready"
    assert len(body["matches"]) == 1
    m = body["matches"][0]
    assert m["rank"] == 1
    assert m["name"] == _REAL_NAME
    assert m["care_type"] == "day-care"            # joined from services.json
    assert m["area"] == "Toa Payoh"
    assert m["cost_per_session"] == 80
    assert m["photo_url"] == "/images/services/02.jpg"   # per-centre photo (record #2), not the care-type fallback
    assert m["is_fallback"] is False
    # Toa Payoh centroid ~ (1.3343, 103.8563); jitter keeps it within ~700 m.
    assert 1.30 < m["lat"] < 1.36 and 103.83 < m["lng"] < 103.88
    # profile echoed for the header pill
    assert body["profile"]["area"] == "Toa Payoh"
    assert body["profile"]["budget"] == 85


def test_fallback_matches_are_flagged_and_carry_no_draft_by_default(client, monkeypatch):
    fb = RankedMatch(
        name="Homage Home-Based Respite Care",
        why="Closest available option — about S$92/session, over your S$60 budget.",
        grounding_passage="A vetted care professional comes to the senior's own home ...",
        similarity_score=None,
    )
    monkeypatch.setattr(
        api.graph, "run",
        lambda messages, **kw: _final_state(fallback=[fb], outcome="fallback_matches"),
    )

    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "cheap overnight care in Sentosa"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["outcome"] == "fallback_matches"
    assert body["matches"][0]["is_fallback"] is True
    assert body["matches"][0]["photo_url"] == "/images/services/09.jpg"   # Homage is record #9
    assert body["draft"] is None


def test_fallback_can_carry_a_draft_when_the_caregiver_asked(client, monkeypatch):
    """When the pipeline drafts off the fallback list, /api/chat surfaces it with its email."""
    fb = RankedMatch(name="Homage Home-Based Respite Care", why="Closest available option — over your budget.",
                     grounding_passage="A vetted care professional comes to the senior's own home ...")
    dr = DraftedMessage(service_name="Homage Home-Based Respite Care",
                        subject="Enquiry about home-based respite",
                        body="Hello, I know this may be above my budget, but could you help with a few hours at home?")
    monkeypatch.setattr(api.graph, "run",
                        lambda messages, **kw: _final_state(fallback=[fb], draft=dr, outcome="fallback_matches"))

    body = client.post("/api/chat", json={"messages": [{"role": "user", "content": "draft an enquiry to Homage"}]}).json()
    assert body["outcome"] == "fallback_matches"
    assert body["matches"][0]["is_fallback"] is True
    assert body["draft"]["service_name"] == "Homage Home-Based Respite Care"
    assert body["draft"]["enquiry_form"] == "https://www.homage.sg/contact-us/"   # Homage is form-only
    assert "I've drafted an enquiry to" not in body["reply"]   # the draft dump is still stripped from the chat text


def test_draft_is_passed_through_when_present(client, monkeypatch):
    match = RankedMatch(name=_REAL_NAME, why="fits", grounding_passage="x")
    draft = DraftedMessage(service_name=_REAL_NAME, subject="Enquiry about weekday dementia day care",
                           body="Hello, I care for my mother ... Could you share availability and fees after subsidy?")
    monkeypatch.setattr(api.graph, "run", lambda messages, **kw: _final_state(matches=[match], draft=draft))

    body = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}).json()
    assert body["draft"]["service_name"] == _REAL_NAME
    assert body["draft"]["subject"].startswith("Enquiry")
    # the centre's enquiry email is attached for the frontend's mailto: button
    assert body["draft"]["enquiry_email"] == "info@dementia.org.sg"   # New Horizon Centres -> Dementia Singapore
    # the inlined draft dump is stripped from the chat reply (the web UI shows it in its own block)
    assert "I've drafted an enquiry to" not in body["reply"]
    assert body["draft"]["body"] not in body["reply"]
    assert "best fit what you described" in body["reply"]  # the shortlist prose is kept


def test_every_service_record_has_an_enquiry_channel():
    """Each record carries `enquiry_email` and `enquiry_form` (one may be None,
    and exactly one record — Tampines Care Home — has neither / AIC only)."""
    neither = 0
    for rec in api._SERVICES_BY_NAME.values():
        assert "enquiry_email" in rec and "enquiry_form" in rec, rec["name"]
        if not rec["enquiry_email"] and not rec["enquiry_form"]:
            neither += 1
    assert neither <= 1


def test_index_not_built_returns_actionable_503(monkeypatch):
    monkeypatch.setattr(api, "_index_ready", lambda: False)
    c = TestClient(api.app)
    r = c.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503
    assert "ingest.py" in r.json()["detail"]


def test_pipeline_llm_error_becomes_502(client, monkeypatch):
    def boom(messages, **kw):
        raise llm.LLMSchemaError("RankedMatches validation failed")

    monkeypatch.setattr(api.graph, "run", boom)
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert "validation" in r.json()["detail"].lower()


def test_empty_transcript_is_rejected(client):
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "   "}]})
    assert r.status_code == 422


def test_photo_url_prefers_per_centre_image_then_falls_back():
    assert api._photo_url({"image": "services/42.jpg", "care_type": "day-care"}) == "/images/services/42.jpg"
    assert api._photo_url({"care_type": "home-help"}) == "/images/home-help.jpg"          # no image -> care-type
    assert api._photo_url({"care_type": "something-new"}) == "/images/default.jpg"          # unknown -> default
    assert api._photo_url({}) == "/images/default.jpg"


def test_every_service_record_has_a_bundled_image_file():
    """The `image` field on each of the 76 records must point at a real file."""
    for rec in api._SERVICES_BY_NAME.values():
        assert "image" in rec, rec["name"]
        assert (api.IMAGES_DIR / rec["image"]).is_file(), rec["image"]


def test_featured_returns_a_random_enriched_sample(client):
    """The pre-chat view: N real services, enriched with photo + coords, no model call."""
    body = client.get("/api/featured?n=4").json()
    assert len(body) == 4
    for i, m in enumerate(body, start=1):
        assert m["rank"] == i
        assert m["name"] in api._SERVICES_BY_NAME          # a real record, not invented
        assert m["photo_url"].startswith("/images/")
        assert m["why"]                                     # first sentence of the description
        assert 1.0 < m["lat"] < 1.6 and 103.0 < m["lng"] < 104.5   # somewhere in Singapore
        assert m["is_fallback"] is False
    # n is clamped to the dataset size, never raises
    big = client.get("/api/featured?n=9999").json()
    assert len(big) == len(api._SERVICES_BY_NAME)
