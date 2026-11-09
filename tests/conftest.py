"""Test-wide settings: never load the heavy phase backends from /health during the tests."""

import os

os.environ.setdefault("ARCA_RAG_DISABLED", "1")
os.environ.setdefault("ARCA_TOOLS_DISABLED", "1")
