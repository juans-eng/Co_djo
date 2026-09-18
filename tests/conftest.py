# -*- coding: utf-8 -*-
"""Shared fixtures for the test suite. No test in this suite requires a
real GEMINI_API_KEY / LLMWHISPERER_API_KEY / network access unless marked
`integration` — see test_djo_service_integration.py."""
import copy
import json
import os

import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def sample_json_djo():
    """A realistic, already-normalized json_djo — equivalent to
    djo_validation_core.ipynb's EXAMPLE_JSON_DJO. NCM 9404.90.00 maps to
    the real ACE 18 rule "MP ou MaxMNO 45%"; its non-originating materials
    (headings 5208/5503) don't share a heading with the product (9404), so
    MP passes deterministically and the origin engine reaches Rule C."""
    with open(os.path.join(FIXTURES_DIR, "sample_json_djo.json"), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def fake_pass_validation_result():
    """A synthetic PASS validation_result, for tests that exercise the
    origin engine specifically without depending on Rule 5's Gemini call."""
    return {
        "overall_status": "PASS",
        "agreement": "ACE_18",
        "validations": [{"rule": "fixture", "status": "PASS", "message": "synthetic pass for isolated testing"}],
    }


def deepcopy_djo(json_djo):
    return copy.deepcopy(json_djo)
