"""
evaluate.py — the slice-8 evaluation harness.

Runs a fixed set of scripted caregiver scenarios end to end through
`graph.run` and reports the five numbers that go on the evaluation slide:

  * schema-validation pass rate — of every `llm.complete` call the run made,
    how many produced output that validated against its Pydantic schema
    (i.e. did NOT raise `LLMSchemaError`). This is the "every model output is
    validated before use" guardrail, measured.
  * tool-call success rate     — of every `search_services` call, how many
    returned without raising.
  * task-completion rate        — how many scenarios ended on the `outcome`
    the scenario expected (clean match / no-match / clarify / cap).
  * answer fidelity (proxy)     — a cheap automated check that every shown
    match carries a real grounding passage and a non-empty "why", and that
    no draft implies a booking was made. `--judge` additionally runs an
    LLM-judge pass for a real supported-claims number. The full
    why / passage / draft text is dumped to the report for manual review.
  * retrieval recall@3          — reuses the labeled query set from
    `tests/test_retrieval.py` so the RAG number lives in one place.

This hits the real LLM backend (~1-3 calls per scenario) and needs the FAISS
index built (`python src/ingest.py`). It is therefore a script you run by
hand — `python src/evaluate.py` — and a skip-gated test
(`tests/test_eval.py`, only runs when `RUN_EVAL=1`). `pytest` stays offline
by default.

Run:  python src/evaluate.py            # writes docs/EVALUATION.md
      python src/evaluate.py --judge    # + LLM-judged fidelity (more tokens)
      python src/evaluate.py --no-recall  # skip the recall@3 pass
"""

from __future__ import annotations

import argparse  # command-line flags for the manual run
import datetime as _dt  # timestamp on the generated report
import importlib.util  # to load the labeled queries from tests/ without putting tests/ on sys.path
import re  # booking-language check in the fidelity proxy
import sys  # stdout reconfigure + exit code
from dataclasses import dataclass, field  # lightweight typed result containers
from pathlib import Path  # repo-root-anchored paths for the report + labeled queries

# Some Groq/Bedrock replies contain characters the legacy Windows console codepage
# cannot encode; replace rather than crash the run on print (mirrors check_env.py).
try:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass  # non-reconfigurable stream (piped) — nothing to do

import graph as graph_mod  # the pipeline under evaluation
import llm  # model seam — wrapped below to count calls / schema failures
import match_rank  # holds the `search_services` binding the Match & Rank node calls
from config import PROJECT_ROOT, settings  # report metadata + the live iteration cap

REPORT_PATH = PROJECT_ROOT / "docs" / "EVALUATION.md"  # where the slide-ready table is written


# ==========================================================================
# Scenario definitions — 8 scripted caregiver conversations
# ==========================================================================

@dataclass
class Scenario:
    """
    One scripted evaluation case.

    id                — short slug, used in the report table.
    messages          — the transcript fed to `graph.run`, chat-completions shape.
    expected_outcome  — the terminal `outcome` a correct run should reach.
    good_result       — prose describing what a good answer looks like; shown in
                        the report as the manual-review yardstick.
    max_iterations    — optional override of `settings.max_iterations` for this
                        scenario only (used to exercise the iteration-cap path).
    """

    id: str
    messages: list[dict[str, str]]
    expected_outcome: str
    good_result: str
    max_iterations: int | None = None


SCENARIOS: list[Scenario] = [
    # --- 1. clean match: dementia day care, in-area, within budget ----------
    Scenario(
        id="clean_match_dementia_toa_payoh",
        messages=[
            {
                "role": "user",
                "content": (
                    "My mother has Alzheimer's and gets agitated and repetitive in the "
                    "afternoons. I lecture part-time and need weekday daytime cover near "
                    "Toa Payoh. I can't really go above $85 a session."
                ),
            }
        ],
        expected_outcome="matches_ready",
        good_result=(
            "New Horizon Centre (Toa Payoh) surfaces first (dementia-trained, Toa Payoh, "
            "$80 <= $85). The 'why' cites de-escalation / structured routine from the "
            "passage. A draft enquiry is produced that asks about availability, fees "
            "after subsidy and transport, and claims no booking."
        ),
    ),
    # --- 2. budget clears nothing -> next-best fallback list --------------
    Scenario(
        id="budget_forces_fallback",
        messages=[
            {
                "role": "user",
                "content": (
                    "My father has dementia and wanders, so he needs a secure place. I "
                    "want weekday daytime care near Pasir Ris but I really cannot pay more "
                    "than $15 a session."
                ),
            }
        ],
        expected_outcome="fallback_matches",
        good_result=(
            "Every candidate is above the $15 ceiling, so the hard budget filter empties "
            "the shortlist. Output must NOT invent a cheaper option — instead it returns "
            "the 4 next-best services ranked location-first then budget (Pasir Ris options "
            "lead; within an area the least-over-budget wins), each labelled 'closest "
            "available option ... over your $15 budget', with no draft."
        ),
    ),
    # --- 3. area clears nothing -> next-best fallback list ---------------
    Scenario(
        id="area_forces_fallback",
        messages=[
            {
                "role": "user",
                "content": (
                    "My mother has early dementia. I need a weekday day programme close to "
                    "home in Sengkang — I can't travel far with her every morning. Budget "
                    "is not really an issue."
                ),
            }
        ],
        expected_outcome="fallback_matches",
        good_result=(
            "No service in the dataset is in Sengkang, so the area hard filter empties "
            "the shortlist. Nothing is in-area and no budget was stated, so the "
            "location-first / budget / schedule fallback falls through to schedule: it "
            "should surface weekday day programmes, each 'why' noting the area is not "
            "Sengkang. No out-of-area centre is presented as if it were nearby."
        ),
    ),
    # --- 4. vague input -> clarifying question ---------------------------
    Scenario(
        id="vague_needs_clarification",
        messages=[
            {
                "role": "user",
                "content": (
                    "I'm exhausted and I don't know how much longer I can keep doing "
                    "this. I need a break."
                ),
            }
        ],
        expected_outcome="needs_clarification",
        good_result=(
            "Intake has no care need, no budget, no area and no timing. It should ask a "
            "single, warm, specific question (who is cared for / what support they need) "
            "rather than guessing and matching."
        ),
    ),
    # --- 5. follow-up turn resolves the clarification --------------------
    Scenario(
        id="followup_resolves_clarification",
        messages=[
            {"role": "user", "content": "I'm exhausted and need a break."},
            {
                "role": "assistant",
                "content": (
                    "I'm sorry you're stretched so thin. Could you tell me who you care "
                    "for and what kind of support they need?"
                ),
            },
            {
                "role": "user",
                "content": (
                    "My husband had a stroke last year. He's mostly mobile now but needs "
                    "physiotherapy and supervision during the day. We're in Choa Chu Kang "
                    "and can manage up to about $70 a session."
                ),
            },
        ],
        expected_outcome="matches_ready",
        good_result=(
            "With the second turn the profile is complete, so the run proceeds past the "
            "clarify branch. St Luke's ElderCare Senior Care Centre (Teck Whye) — day "
            "care + rehab, Choa Chu Kang, $60 — should be the top match, grounded on the "
            "rehab line of its passage."
        ),
    ),
    # --- 6. in-home: refuses a centre, mobility-limited -----------------
    Scenario(
        id="in_home_refuses_centre",
        messages=[
            {
                "role": "user",
                "content": (
                    "My husband is bedbound after a bad fall and flatly refuses to go to "
                    "any centre. I just need someone to cover a few hours at home so I can "
                    "run errands. Anywhere on the island is fine and budget is flexible."
                ),
            }
        ],
        expected_outcome="matches_ready",
        good_result=(
            "Homage Home-Based Respite Care (islandwide, care worker travels to the home, "
            "flexible hourly booking) should be the top match, grounded on the "
            "'refuses point-blank to leave the house' line. The draft asks about hourly "
            "rate after the Home Caregiving Grant and about nurse availability."
        ),
    ),
    # --- 7. night respite: sundowning, caregiver can't sleep -----------
    Scenario(
        id="night_respite_sundowning",
        messages=[
            {
                "role": "user",
                "content": (
                    "My mother has dementia and is up and disoriented most of the night. "
                    "I haven't slept properly in weeks and I still have to work in the "
                    "day. I can travel for the right place; budget is around $120 a night."
                ),
            }
        ],
        expected_outcome="matches_ready",
        good_result=(
            "NTUC Health Night Respite / Staycay@Henderson (overnight-only, nursing aide "
            "on call, ~$80-120/night) should be the top match, grounded on the "
            "'up, disoriented and restless through the night' line. Day-care options "
            "should be dropped as the wrong shape."
        ),
    ),
    # --- 8. iteration cap: same as #1 but MAX_ITERATIONS=1 -------------
    Scenario(
        id="iteration_cap_stops_run",
        messages=[
            {
                "role": "user",
                "content": (
                    "My mother has Alzheimer's and gets agitated in the afternoons. I "
                    "need weekday daytime cover near Toa Payoh, up to $85 a session."
                ),
            }
        ],
        expected_outcome="stopped_iteration_cap",
        good_result=(
            "With the cap at 1, Intake alone trips it and routing goes straight to "
            "Output. The caregiver is told the run stopped early, that nothing was "
            "sent, and to retry with more detail — no partial recommendation leaks out."
        ),
        max_iterations=1,
    ),
]


# ==========================================================================
# Instrumentation — count llm.complete / search_services calls + failures
# ==========================================================================

# Per-scenario tallies. `_install_instrumentation()` wraps the two seams once
# for the process; `run_scenario` zeroes this before each scenario and reads it
# after, so the numbers are attributable per case and also summable.
_COUNTER: dict[str, int] = {
    "llm_calls": 0,        # every llm.complete invocation (any node)
    "schema_failures": 0,  # llm.complete calls that raised LLMSchemaError
    "tool_calls": 0,       # every search_services invocation
    "tool_failures": 0,    # search_services calls that raised
}

_REAL_GENERATE = llm.generate  # kept aside so the optional LLM-judge doesn't inflate _COUNTER


def _reset_counter() -> None:
    """Zero every tally before a scenario runs."""
    for key in _COUNTER:
        _COUNTER[key] = 0


def _install_instrumentation() -> None:
    """
    Monkeypatch `llm.complete` and `match_rank.search_services` with thin
    counting wrappers. Idempotent-ish: only call this once per process (the
    harness does, in `run_suite`).
    """
    real_complete = llm.complete            # the genuine schema-validating call
    real_search = match_rank.search_services  # the genuine hybrid-retrieval call

    def counting_complete(*args, **kwargs):
        """Count the call, then delegate; a schema failure is tallied and re-raised."""
        _COUNTER["llm_calls"] += 1
        try:
            return real_complete(*args, **kwargs)
        except llm.LLMSchemaError:
            _COUNTER["schema_failures"] += 1  # the guardrail fired — record it, don't hide it
            raise

    def counting_search(*args, **kwargs):
        """Count the call, then delegate; any exception is tallied and re-raised."""
        _COUNTER["tool_calls"] += 1
        try:
            return real_search(*args, **kwargs)
        except Exception:
            _COUNTER["tool_failures"] += 1
            raise

    llm.complete = counting_complete            # every node calls `llm.complete(...)` by attribute
    match_rank.search_services = counting_search  # the Match & Rank node's bound name


# ==========================================================================
# Answer fidelity — cheap automated proxy + optional LLM-judge
# ==========================================================================

# Language that would imply the agent booked / reserved / confirmed a place —
# which V1 must never do (there is no sender). A draft containing this is a
# fidelity failure, not just a style nit.
_BOOKING_LANGUAGE = re.compile(
    r"\b(booked|reserved|confirmed your (?:place|spot|slot|booking)|"
    r"you(?:'re| are) enrolled|i have (?:arranged|scheduled|set up) (?:a|your))\b",
    re.IGNORECASE,
)


def check_fidelity_proxy(final: dict) -> list[str]:
    """
    Cheap, no-token fidelity check on a finished run. Returns a list of issue
    strings (empty == clean):
      * a shown match with an empty 'why' or no grounding passage
      * a draft body that implies a booking was made, or is empty
    """
    issues: list[str] = []

    # Both the true-fit shortlist and the no-match fallback list must carry a
    # grounded passage + a non-empty 'why'.
    shown = (final.get("candidate_matches", []) or []) + (final.get("fallback_matches", []) or [])
    for match in shown:  # each service shown to the caregiver
        if not match.why.strip():
            issues.append(f"{match.name}: empty 'why' line")
        if not match.grounding_passage.strip():
            issues.append(f"{match.name}: no grounding passage attached")

    draft = final.get("drafted_message")  # DraftedMessage | None
    if draft is not None:
        if not draft.body.strip():
            issues.append("draft body is empty")
        if _BOOKING_LANGUAGE.search(draft.body):
            issues.append("draft implies a booking / reservation was made")

    return issues


def judge_fidelity(final: dict) -> tuple[int, int]:
    """
    Optional LLM-judge pass (`--judge`). For every shown match, ask the model
    whether the 'why' line is supported by that match's grounding passage.
    Returns (supported_count, total_count). Uses the real `generate` directly
    so it does not disturb the `_COUNTER` tallies.
    """
    matches = final.get("candidate_matches", []) or []
    supported = 0
    for match in matches:
        prompt = (
            "You are checking whether a claim is supported by a source passage.\n\n"
            f"SOURCE PASSAGE:\n{match.grounding_passage}\n\n"
            f"CLAIM (shown to a caregiver as 'why this service fits'):\n{match.why}\n\n"
            "Is every factual assertion in the CLAIM supported by, or a reasonable "
            "paraphrase of, the SOURCE PASSAGE? Reply with exactly one word: "
            "SUPPORTED or UNSUPPORTED."
        )
        verdict = _REAL_GENERATE(
            [{"role": "user", "content": prompt}],
            model_role="extract",  # a cheap yes/no classification
            max_tokens=256,        # reasoning-style models spend hidden tokens first
        ).text.strip().upper()
        if "UNSUPPORTED" not in verdict and "SUPPORTED" in verdict:
            supported += 1
    return supported, len(matches)


# ==========================================================================
# recall@3 — reuse the labeled query set from tests/test_retrieval.py
# ==========================================================================

def _load_labeled_queries() -> list[dict]:
    """
    Import `LABELED_QUERIES` from `tests/test_retrieval.py` by file path, so the
    labeled set stays defined in exactly one place (the retrieval test) without
    putting `tests/` on `sys.path`.
    """
    path = PROJECT_ROOT / "tests" / "test_retrieval.py"
    spec = importlib.util.spec_from_file_location("_eval_test_retrieval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return list(module.LABELED_QUERIES)


def compute_recall_at_3() -> tuple[float, int, int]:
    """
    Run each labeled query through the retrieval tool (semantic-only, no budget
    or area constraint) and count how many surface their expected service in the
    top 3. Returns (recall, hits, total).
    """
    import retrieval_tool  # imported here so `--no-recall` runs need not touch FAISS at all

    queries = _load_labeled_queries()
    hits = 0
    for case in queries:
        profile = {"needs_description": case["query"], "budget": None, "area": None}
        results = retrieval_tool.search_services(profile, k=8)[:3]  # NOT the wrapped name — recall != tool-call metric; tool returns top-4, this stays recall@3
        if any(r["name"] == case["expected"] for r in results):
            hits += 1
    total = len(queries)
    return (hits / total if total else 0.0), hits, total


# ==========================================================================
# Running one scenario / the whole suite
# ==========================================================================

@dataclass
class ScenarioResult:
    """Everything the report needs about one scenario after it ran."""

    scenario: Scenario
    outcome: str | None                 # terminal outcome, or None if the run raised
    completed: bool                     # outcome == scenario.expected_outcome
    llm_calls: int                      # llm.complete invocations this scenario made
    schema_failures: int                # of those, how many tripped the schema guardrail
    tool_calls: int                     # search_services invocations
    tool_failures: int                  # of those, how many raised
    fidelity_issues: list[str]          # from check_fidelity_proxy (matches_ready runs only)
    judge_supported: int | None = None  # LLM-judge: supported 'why' lines (--judge only)
    judge_total: int | None = None      # LLM-judge: total 'why' lines checked
    error: str | None = None            # exception text if the run blew up
    final: dict = field(default_factory=dict)  # the finished state (for the fidelity appendix)


def run_scenario(scenario: Scenario, *, judge: bool = False) -> ScenarioResult:
    """
    Execute one scenario through `graph.run` and package the result. Restores
    `settings.max_iterations` afterwards even if the scenario overrode it.
    """
    _reset_counter()  # per-scenario tallies start clean

    original_cap = settings.max_iterations  # remember so we can put it back
    if scenario.max_iterations is not None:
        settings.max_iterations = scenario.max_iterations  # shared Settings object; guardrails reads it live

    outcome: str | None = None
    error: str | None = None
    final: dict = {}
    try:
        final = graph_mod.run(scenario.messages)  # Intake -> ... -> Output
        outcome = final.get("outcome")
    except llm.LLMSchemaError as exc:  # the schema guardrail aborted the run — a real, countable failure
        error = f"LLMSchemaError: {exc}"
    except Exception as exc:  # backend down, retrieval broke, etc. — record, don't crash the suite
        error = f"{type(exc).__name__}: {exc}"
    finally:
        settings.max_iterations = original_cap  # always restore the cap

    # Fidelity proxy makes sense on any run that showed services (true fit or fallback).
    fidelity_issues = check_fidelity_proxy(final) if outcome in ("matches_ready", "fallback_matches") else []

    judge_supported = judge_total = None
    if judge and outcome == "matches_ready":
        judge_supported, judge_total = judge_fidelity(final)

    return ScenarioResult(
        scenario=scenario,
        outcome=outcome,
        completed=(outcome == scenario.expected_outcome),
        llm_calls=_COUNTER["llm_calls"],
        schema_failures=_COUNTER["schema_failures"],
        tool_calls=_COUNTER["tool_calls"],
        tool_failures=_COUNTER["tool_failures"],
        fidelity_issues=fidelity_issues,
        judge_supported=judge_supported,
        judge_total=judge_total,
        error=error,
        final=final,
    )


@dataclass
class SuiteReport:
    """Aggregate metrics across the whole scenario set — the slide numbers."""

    results: list[ScenarioResult]
    schema_pass_rate: float       # (llm_calls - schema_failures) / llm_calls
    tool_success_rate: float      # (tool_calls - tool_failures) / tool_calls
    task_completion_rate: float   # scenarios where outcome == expected, / total
    fidelity_issue_count: int     # total proxy issues across all scenarios
    judged_support_rate: float | None  # LLM-judge supported/total, or None if --judge off
    recall_at_3: float | None     # None if --no-recall
    recall_hits: int
    recall_total: int
    total_llm_calls: int
    total_tool_calls: int
    backend: str                  # which LLM backend served the run (for the report header)


def run_suite(*, judge: bool = False, compute_recall: bool = True) -> SuiteReport:
    """
    Run every scenario, then fold the per-scenario tallies into the headline
    rates. This is what `tests/test_eval.py` calls to assert on.
    """
    _install_instrumentation()  # wrap the two seams once for this process

    results = [run_scenario(sc, judge=judge) for sc in SCENARIOS]

    # --- schema-validation pass rate ------------------------------------
    llm_calls = sum(r.llm_calls for r in results)
    schema_failures = sum(r.schema_failures for r in results)
    schema_pass_rate = (llm_calls - schema_failures) / llm_calls if llm_calls else 1.0

    # --- tool-call success rate ---------------------------------------
    tool_calls = sum(r.tool_calls for r in results)
    tool_failures = sum(r.tool_failures for r in results)
    tool_success_rate = (tool_calls - tool_failures) / tool_calls if tool_calls else 1.0

    # --- task-completion rate ---------------------------------------
    completed = sum(1 for r in results if r.completed)
    task_completion_rate = completed / len(results) if results else 0.0

    # --- answer fidelity ------------------------------------------
    fidelity_issue_count = sum(len(r.fidelity_issues) for r in results)
    judged_support_rate: float | None = None
    if judge:
        s = sum(r.judge_supported or 0 for r in results)
        t = sum(r.judge_total or 0 for r in results)
        judged_support_rate = s / t if t else None

    # --- recall@3 -----------------------------------------------
    recall_at_3 = recall_hits = recall_total = None
    if compute_recall:
        recall_at_3, recall_hits, recall_total = compute_recall_at_3()

    return SuiteReport(
        results=results,
        schema_pass_rate=schema_pass_rate,
        tool_success_rate=tool_success_rate,
        task_completion_rate=task_completion_rate,
        fidelity_issue_count=fidelity_issue_count,
        judged_support_rate=judged_support_rate,
        recall_at_3=recall_at_3,
        recall_hits=recall_hits or 0,
        recall_total=recall_total or 0,
        total_llm_calls=llm_calls,
        total_tool_calls=tool_calls,
        backend=settings.llm_backend,
    )


# ==========================================================================
# Report rendering
# ==========================================================================

def _pct(value: float) -> str:
    """Format a 0-1 rate as a whole-ish percentage."""
    return f"{value * 100:.0f}%"


def render_report(report: SuiteReport, *, judged: bool) -> str:
    """Build the Markdown report string that gets written to docs/EVALUATION.md."""
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []

    lines.append("# RespiteSG V1 - evaluation report")
    lines.append("")
    lines.append(
        f"_Generated by `python src/evaluate.py` on {now}. "
        f"Backend: `{report.backend}`. {len(report.results)} scripted scenarios._"
    )
    lines.append("")
    lines.append(
        "> This is a small, honest harness: 8 hand-written caregiver conversations "
        "run end-to-end through the LangGraph pipeline against the real model and "
        "the real FAISS index. Numbers move run-to-run because the model is "
        "non-deterministic; the point is the shape, not a leaderboard score."
    )
    lines.append("")

    # --- headline table -------------------------------------------
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| Metric | Result | What it measures |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Schema-validation pass rate | **{_pct(report.schema_pass_rate)}** "
        f"({report.total_llm_calls - sum(r.schema_failures for r in report.results)}"
        f"/{report.total_llm_calls} `llm.complete` calls) | "
        "Every model output validated against its Pydantic schema before use "
        "(the core guardrail). |"
    )
    lines.append(
        f"| Tool-call success rate | **{_pct(report.tool_success_rate)}** "
        f"({report.total_tool_calls - sum(r.tool_failures for r in report.results)}"
        f"/{report.total_tool_calls} `search_services` calls) | "
        "Hybrid retrieval returned without raising. |"
    )
    lines.append(
        f"| Task-completion rate | **{_pct(report.task_completion_rate)}** "
        f"({sum(1 for r in report.results if r.completed)}/{len(report.results)} scenarios) | "
        "Run ended on the outcome the scenario expected. |"
    )
    proxy = "no issues" if report.fidelity_issue_count == 0 else f"{report.fidelity_issue_count} issue(s)"
    lines.append(
        f"| Answer fidelity (auto proxy) | **{proxy}** | "
        "Every shown match carries a real grounding passage + non-empty 'why'; "
        "no draft implies a booking. |"
    )
    if judged and report.judged_support_rate is not None:
        lines.append(
            f"| Answer fidelity (LLM-judge) | **{_pct(report.judged_support_rate)}** | "
            "'why' lines the judge rated as supported by their grounding passage. |"
        )
    if report.recall_at_3 is not None:
        lines.append(
            f"| Retrieval recall@3 | **{report.recall_at_3:.2f}** "
            f"({report.recall_hits}/{report.recall_total} labeled queries) | "
            "Expected service appears in the top 3 retrieved (semantic-only). |"
        )
    lines.append("")

    # --- per-scenario table --------------------------------------
    lines.append("## Per-scenario")
    lines.append("")
    lines.append("| # | Scenario | Expected | Got | Pass | LLM calls | Notes |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(report.results, start=1):
        got = r.outcome or "(raised)"
        passed = "yes" if r.completed else "**NO**"
        note = ""
        if r.error:
            note = r.error
        elif r.fidelity_issues:
            note = "; ".join(r.fidelity_issues)
        elif r.judge_total:
            note = f"judge: {r.judge_supported}/{r.judge_total} 'why' lines supported"
        lines.append(
            f"| {i} | `{r.scenario.id}` | {r.scenario.expected_outcome} | {got} | "
            f"{passed} | {r.llm_calls} | {note} |"
        )
    lines.append("")

    # --- fidelity appendix (manual-review material) -------------
    lines.append("## Appendix - fidelity review material")
    lines.append("")
    lines.append(
        "For each `matches_ready` / `fallback_matches` scenario: the shown "
        "services, each grounded 'why', the passage it must be traceable to, and "
        "the drafted enquiry (matches_ready only). Read these side by side to "
        "confirm nothing is invented."
    )
    lines.append("")
    for i, r in enumerate(report.results, start=1):
        if r.outcome not in ("matches_ready", "fallback_matches"):
            continue
        lines.append(f"### {i}. `{r.scenario.id}`  ({r.outcome})")
        lines.append("")
        lines.append(f"*Yardstick:* {r.scenario.good_result}")
        lines.append("")
        shown = (r.final.get("candidate_matches", []) or []) + (r.final.get("fallback_matches", []) or [])
        for m in shown:
            lines.append(f"- **{m.name}**")
            lines.append(f"  - why: {m.why}")
            lines.append(f"  - grounding passage: {m.grounding_passage}")
        draft = r.final.get("drafted_message")
        if draft is not None:
            lines.append(f"- **draft to {draft.service_name}**")
            lines.append(f"  - subject: {draft.subject}")
            body_indented = draft.body.replace("\n", "\n    ")
            lines.append(f"  - body:\n\n    {body_indented}")
        lines.append("")

    return "\n".join(lines)


# ==========================================================================
# CLI entry point
# ==========================================================================

def main() -> int:
    """Parse flags, run the suite, print + write the report, return an exit code."""
    parser = argparse.ArgumentParser(description="RespiteSG V1 evaluation harness")
    parser.add_argument(
        "--judge", action="store_true",
        help="also run an LLM-judge pass over each 'why' line (costs extra tokens)",
    )
    parser.add_argument(
        "--no-recall", dest="recall", action="store_false",
        help="skip the recall@3 pass (does not touch the FAISS index for that metric)",
    )
    args = parser.parse_args()

    print(f"Running {len(SCENARIOS)} scenarios against backend '{settings.llm_backend}' ...\n")
    report = run_suite(judge=args.judge, compute_recall=args.recall)

    text = render_report(report, judged=args.judge)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)  # docs/ already exists, but be safe
    REPORT_PATH.write_text(text, encoding="utf-8")

    print(text)
    print(f"\nReport written to {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    # Exit non-zero if the two guardrail rates are not perfect, so this can gate
    # a demo setup step. Task-completion is expected to wobble with the model, so
    # it does not fail the process here (the test applies a threshold instead).
    ok = report.schema_pass_rate == 1.0 and report.tool_success_rate == 1.0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
