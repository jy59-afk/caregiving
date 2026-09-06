"""
conftest.py — pytest picks this up automatically from the repo root.

It puts src/ on sys.path so tests can `import ingest`, `import retrieval_tool`,
`import config` etc. without a package install, matching how the scripts are
run in dev (`python src/ingest.py`).
"""

import sys  # to modify the module search path
from pathlib import Path  # for a robust path to src/

SRC = Path(__file__).parent / "src"  # .../hackathon/src
sys.path.insert(0, str(SRC))         # make src/ modules importable from tests
