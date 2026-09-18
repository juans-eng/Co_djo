# -*- coding: utf-8 -*-
"""ACE 18 NCM lookup — src/ace18/ncm_service.py. Uses the REAL cached
reos_ace18_cache.json on disk (no Docling/network call; the cache already
exists, so this only ever reads a local JSON file)."""
from src import config
from src.ace18.ncm_service import ACE18Service


def _service():
    return ACE18Service(config.ACE18_PDF_PATH, config.ACE18_CACHE_PATH)


def test_exact_ncm_match_returns_mercosul_rule():
    result = _service().query_ncm("9404.90.00")
    assert result["found"] is True
    assert "matched_ncm_expression" in result["mercosul_rule"]
    assert "raw_rule" in result["mercosul_rule"]
    assert result["mercosul_rule"]["raw_rule"]  # non-empty


def test_ncm_not_found_returns_found_false():
    result = _service().query_ncm("0000.00.00")
    assert result["found"] is False
    assert result["mercosul_rule"] is None


def test_malformed_ncm_raises_value_error():
    import pytest
    with pytest.raises(ValueError):
        _service().query_ncm("123")  # not 8 digits


def test_parsed_details_carries_max_mno_percentage_when_present():
    # Real cache row: NCM 9404.90.00 -> chapter rule "MP ou MaxMNO 45%"
    result = _service().query_ncm("9404.90.00")
    parsed = result["mercosul_rule"]["parsed_details"]
    assert parsed["max_mno_percentage"] == 45
