# -*- coding: utf-8 -*-
"""
config.py
==========

Centralized, environment-overridable paths/constants for the backend.
Nothing here is secret (API keys are read directly from the environment by
the modules that need them — see `gemini/service.py` and
`djo/extractor.py` — never funneled through this file) — this module only
holds file-system paths and default agreement selection, so there's a
single place to point the backend at different data files (e.g. in tests).
"""
from __future__ import annotations

import os

# Project root = two levels up from this file (src/config.py -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ACE 18 source PDF + JSON cache (see ace18/ncm_service.py). The cache is
# reused as-is once built — no Docling run happens as long as it exists.
ACE18_PDF_PATH = os.environ.get(
    "ACE18_PDF_PATH",
    os.path.join(PROJECT_ROOT, "dados", "ACE_018_221_pt (1).pdf"),
)
ACE18_CACHE_PATH = os.environ.get(
    "ACE18_CACHE_PATH",
    os.path.join(PROJECT_ROOT, "dados", "reos_ace18_cache.json"),
)

DEFAULT_AGREEMENT = os.environ.get("DEFAULT_AGREEMENT", "ACE_18")
