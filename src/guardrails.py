"""
guardrails.py — the three hard guardrails, in one auditable place.

CLAUDE.md's "responsible AI" talking point is that the agent is fenced in by
mechanism, not by prompt wording. This module is that fence:

  1. Iteration cap    — `iteration_cap_reached(state)` is the routing check the
                        graph consults after every worker node; when it trips,
                        the graph is forced to the terminal Output node instead
                        of looping further. The cap is `settings.max_iterations`
                        (from `MAX_ITERATIONS` in .env, default 6).
  2. Tool allow-list  — `ALLOWED_TOOLS` is the complete set of tools the agent
                        may ever reach: hybrid retrieval and the draft step.
                        There is deliberately **no** send / book / call tool
                        anywhere in V1. `assert_tool_allowed()` enforces it at
                        call sites; `scan_source_for_action_tools()` lets the
                        test suite prove none has been added later.
  3. Schema validation — not implemented here (it lives inside `llm.complete`,
                        so every node inherits it), but `LLMSchemaError` is
                        re-exported so the graph layer can name it when it
                        asserts that a validation failure aborts the run.

Nothing here touches a network or a model; it is pure policy, cheap to import.
"""

from __future__ import annotations

import re  # for the source scan that proves no action tool was added
from pathlib import Path  # repo-root-anchored path to src/
from typing import TYPE_CHECKING  # keep the state import type-only to avoid a cycle

from config import settings  # typed view of .env — this is where max_iterations comes from
from llm import LLMSchemaError  # re-exported so graph.py can reference the guardrail exception by one name

if TYPE_CHECKING:  # RespiteState is only needed for hints; importing it at runtime would be circular-ish and needless
    from state import RespiteState

__all__ = [
    "ALLOWED_TOOLS",
    "GuardrailViolation",
    "LLMSchemaError",
    "iteration_cap_reached",
    "assert_tool_allowed",
    "scan_source_for_action_tools",
]


# ==========================================================================
# 1. Iteration cap
# ==========================================================================

def iteration_cap_reached(state: "RespiteState") -> bool:
    """
    True once the pipeline has taken at least `settings.max_iterations` steps.

    Every node bumps `state["iteration_count"]` (including on its early-return
    paths), so this is a reliable "have we done enough work" check. The graph
    calls it in its conditional edges: when it returns True, routing goes
    straight to the Output node and the run ends.
    """
    return state.get("iteration_count", 0) >= settings.max_iterations


# ==========================================================================
# 2. Tool allow-list
# ==========================================================================

# The complete, closed set of tool names the agent is permitted to invoke.
#   search_services — hybrid retrieval over the curated dataset (read-only).
#   draft_message   — the Explain & Draft node; produces text, sends nothing.
# Adding anything that contacts a service on the caregiver's behalf is a
# design change that must be argued for, not a quiet import — hence the scan below.
ALLOWED_TOOLS: frozenset[str] = frozenset({"search_services", "draft_message"})

# Verb stems that would betray a tool capable of taking real-world action for
# the caregiver. `scan_source_for_action_tools()` greps src/ for a top-level
# `def <stem>...` and the graph test fails if it finds one.
_ACTION_TOOL_PATTERNS: tuple[str, ...] = (
    "send", "book", "submit", "dispatch", "reserve",
    "email", "call", "phone", "message_service", "contact_service", "enrol", "enroll",
)


class GuardrailViolation(RuntimeError):
    """Raised when something tries to step outside a hard guardrail (e.g. an unlisted tool)."""


def assert_tool_allowed(tool_name: str) -> None:
    """
    Call-site enforcement of the allow-list. Any code path that dispatches a
    named tool routes through here first; an unlisted name raises rather than
    running.
    """
    if tool_name not in ALLOWED_TOOLS:  # closed-set check — unknown == forbidden
        raise GuardrailViolation(
            f"Tool {tool_name!r} is not in the allow-list {sorted(ALLOWED_TOOLS)}. "
            "V1 has no tool that sends, books, or contacts a service — the agent "
            "drafts and the caregiver sends."
        )


def _src_dir() -> Path:
    """Absolute path to the project's src/ directory (this file lives in it)."""
    return Path(__file__).resolve().parent


def scan_source_for_action_tools(src_dir: Path | None = None) -> list[str]:
    """
    Static check: walk every .py file in src/ and return a list of
    "<file>:<line>: <definition>" strings for any top-level function whose name
    starts with an action verb (send_/book_/submit_/...).

    An empty list is the healthy state and what the guardrail test asserts.
    This is intentionally a source scan, not an import-and-introspect: it
    catches a forbidden tool even if nothing wires it into the graph yet.
    """
    root = src_dir or _src_dir()  # default to this package's own directory
    # `def send_foo(` / `async def book_bar(` at column 0 — a module-level tool definition.
    pattern = re.compile(
        r"^(?:async\s+)?def\s+(" + "|".join(_ACTION_TOOL_PATTERNS) + r")\w*\s*\(",
        re.MULTILINE,
    )
    hits: list[str] = []  # collects human-readable locations of any offending def
    for path in sorted(root.glob("*.py")):  # deterministic order for stable test output
        if path.name == "guardrails.py":  # skip this file — it only *names* the patterns
            continue
        text = path.read_text(encoding="utf-8")  # read the whole module source
        for match in pattern.finditer(text):  # every module-level action-verb def
            line_no = text.count("\n", 0, match.start()) + 1  # 1-based line number of the def
            hits.append(f"{path.name}:{line_no}: {match.group(0).strip()}")
    return hits
