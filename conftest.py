"""
conftest.py — pytest picks this up automatically from the repo root.

It puts src/ on sys.path so tests can `import ingest`, `import retrieval_tool`,
`import config` etc. without a package install, matching how the scripts are
run in dev (`python src/ingest.py`).
"""

import os  # to set process env before heavy libraries read it
import sys  # to modify the module search path
from pathlib import Path  # for a robust path to src/

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # silence the Windows symlink-cache warning from huggingface_hub
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")       # avoid the tokenizers fork-after-parallelism warning

SRC = Path(__file__).parent / "src"  # .../hackathon/src
sys.path.insert(0, str(SRC))         # make src/ modules importable from tests
