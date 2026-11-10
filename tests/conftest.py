"""Shared test setup."""

import os

# /health tries to load every phase. Phase 2 needs no adapter, so it would download and load
# Qwen3-1.7B inside the test suite; "off" makes it answer 501 instead. Phase 2's own tests
# build ToolLoop directly with a scripted model.
os.environ["ARCA_TOOLS_MODEL"] = "off"
