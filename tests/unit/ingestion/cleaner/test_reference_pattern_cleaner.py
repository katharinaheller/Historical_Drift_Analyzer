# tests/core/ingestion/cleaner/test_reference_pattern_cleaner.py
import pytest
from src.core.ingestion.cleaner.simple_cleaners import ReferencePatternCleaner

@pytest.fixture
def cleaner():
    # Instantiate cleaner once for all tests
    return ReferencePatternCleaner()

def test_remove_apa_and_harvard(cleaner):
    text = "Reinforcement learning (Sutton & Barto, 2018) is fundamental."
    cleaned = cleaner.clean(text)
    assert "(Sutton & Barto, 2018)" not in cleaned
    assert "Reinforcement learning" in cleaned

def test_remove_et_al(cleaner):
    text = "Deep Q-Learning (Mnih et al., 2015) achieved strong results."
    cleaned = cleaner.clean(text)
    assert "Mnih" not in cleaned
    assert "2015" not in cleaned

def test_remove_ieee_numeric(cleaner):
    text = "This was first described in [12] and refined in [3, 4, 5]."
    cleaned = cleaner.clean(text)
    assert "[" not in cleaned and "]" not in cleaned

def test_remove_year_only(cleaner):
    text = "AI has evolved rapidly (2023)."
    cleaned = cleaner.clean(text)
    assert "(2023)" not in cleaned

def test_remove_doi_and_arxiv(cleaner):
    text = "See doi:10.1000/xyz123 and arXiv:2307.0192 for more details."
    cleaned = cleaner.clean(text)
    assert "doi:" not in cleaned.lower()
    assert "arxiv:" not in cleaned.lower()

def test_remove_conference_refs(cleaner):
    text = "Presented at Proc. IEEE 2019 and In Proceedings of ICML 2020."
    cleaned = cleaner.clean(text)
    assert "IEEE" not in cleaned
    assert "ICML" not in cleaned

def test_cleanup_whitespace_and_punctuation(cleaner):
    text = "Some text   with  extra   spaces , punctuation ; and refs [1]."
    cleaned = cleaner.clean(text)
    # Should have normalized spacing and removed bracketed refs
    assert "  " not in cleaned
    assert "[1]" not in cleaned
    assert not cleaned.endswith(" ")

def test_combined_case(cleaner):
    text = "Reinforcement Learning (Sutton et al., 2018) [12] doi:10.1000/xyz."
    cleaned = cleaner.clean(text)
    assert "Sutton" not in cleaned
    assert "[" not in cleaned
    assert "doi:" not in cleaned
    # Still retains core text
    assert "Reinforcement Learning" in cleaned
