"""
test_eval.py — thin wrapper that runs the slice-8 evaluation harness and
asserts the guardrail rates hold.

This is the one test in the suite that hits the real LLM backend and needs the
FAISS index, so it is double-gated: it only runs when BOTH
  * `RUN_EVAL=1` is set in the environment, and
  * the vector index has been built (`python src/ingest.py`).
Otherwise it skips, and `pytest -q` stays fully offline (like
`tests/test_retrieval.py`'s recall@3 test).

Run it explicitly with:
    RUN_EVAL=1 pytest tests/test_eval.py -v -s        # bash
    $env:RUN_EVAL=1; pytest tests/test_eval.py -v -s  # PowerShell
"""

import os  # to read the RUN_EVAL opt-in flag

import pytest  # skip markers + assertion helpers

from ingest import DEFAULT_INDEX_PATH  # to check the index exists before running

# One gate for the whole module: opt-in flag AND a built index.
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_EVAL") != "1" or not (DEFAULT_INDEX_PATH / "index.faiss").exists(),
    reason="eval hits the real LLM + needs the FAISS index; set RUN_EVAL=1 after `python src/ingest.py`",
)


def test_eval_suite_guardrail_rates_hold():
    """
    Run all scenarios once and assert:
      * schema-validation pass rate is perfect — every model output validated
        (a single LLMSchemaError here is a real regression, not model wobble)
      * tool-call success rate is perfect — retrieval never raised
      * task-completion rate clears a loose floor (the model is non-deterministic,
        so this is a floor, not an equality)
      * recall@3 still meets the retrieval target
    """
    import evaluate  # imported here so collection stays cheap when the module is skipped

    report = evaluate.run_suite(judge=False, compute_recall=True)

    # Print the slide table even on success (`pytest -s`) so a manual run doubles
    # as report generation.
    print("\n" + evaluate.render_report(report, judged=False))

    assert report.schema_pass_rate == 1.0, (
        f"schema-validation pass rate {report.schema_pass_rate:.2f} < 1.0 — "
        "a model output failed its Pydantic schema"
    )
    assert report.tool_success_rate == 1.0, (
        f"tool-call success rate {report.tool_success_rate:.2f} < 1.0 — search_services raised"
    )
    assert report.task_completion_rate >= 0.75, (
        f"task-completion rate {report.task_completion_rate:.2f} < 0.75 "
        f"({sum(1 for r in report.results if r.completed)}/{len(report.results)} scenarios)"
    )
    assert report.fidelity_issue_count == 0, (
        f"{report.fidelity_issue_count} answer-fidelity proxy issue(s): "
        f"{[iss for r in report.results for iss in r.fidelity_issues]}"
    )
    assert report.recall_at_3 is not None and report.recall_at_3 >= 0.7, (
        f"recall@3 {report.recall_at_3} < 0.70"
    )
