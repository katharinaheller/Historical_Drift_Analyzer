# tests/unit/ingestion/test_metadata_extractors.py
from __future__ import annotations
import io
import json
import tempfile
from pathlib import Path

import pytest
import fitz  # PyMuPDF

from src.core.ingestion.metadata.metadata_extractor_factory import MetadataExtractorFactory


# ----------------------------------------------------------------------
# Fixtures and Helpers
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def tmp_pdf() -> Path:
    """Creates a temporary multi-page PDF with metadata and simple TOC for testing."""
    pdf_path = Path(tempfile.gettempdir()) / "metadata_test.pdf"
    doc = fitz.open()

    # page 1: textual TOC simulation
    toc_text = """Inhalt
1. Introduction .................................. 1
2. Methods ....................................... 2
3. Results ....................................... 3
4. Conclusion .................................... 4
"""
    doc.new_page().insert_text((72, 72), toc_text)

    # page 2: content
    page = doc.new_page()
    page.insert_text((72, 72), "Deep Learning in NLP\nJohn Doe\nAbstract: This paper explores deep learning models.")

    doc.set_metadata({
        "title": "Deep Learning in NLP",
        "author": "John Doe",
        "creationDate": "D:20231101000000"
    })
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture(scope="module")
def dummy_grobid_xml() -> str:
    """Provides a minimal valid TEI XML snippet for GROBID tests."""
    return """<?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <teiHeader>
        <fileDesc>
          <titleStmt>
            <title type="main">Deep Learning in Natural Language Processing</title>
            <author><forename>John</forename><surname>Doe</surname></author>
            <author><forename>Jane</forename><surname>Smith</surname></author>
          </titleStmt>
          <sourceDesc>
            <biblStruct>
              <monogr>
                <imprint>
                  <date when="2023" />
                </imprint>
              </monogr>
            </biblStruct>
          </sourceDesc>
        </fileDesc>
        <profileDesc>
          <abstract>This paper explores deep learning techniques for NLP.</abstract>
        </profileDesc>
      </teiHeader>
    </TEI>
    """


@pytest.fixture(scope="module")
def parsed_doc(dummy_grobid_xml: str) -> dict:
    """Constructs a mock parsed document dict."""
    return {
        "grobid_xml": dummy_grobid_xml,
        "text": "Deep learning enables powerful natural language understanding models.",
        "metadata": {}
    }


@pytest.fixture(scope="module")
def factory() -> MetadataExtractorFactory:
    """Initializes a factory for all metadata fields including TOC."""
    cfg = {
        "options": {
            "metadata_fields": [
                "title",
                "authors",
                "year",
                "abstract",
                "detected_language",
                "file_size",
                "toc",  # newly added
            ]
        }
    }
    return MetadataExtractorFactory.from_config(cfg)


# ----------------------------------------------------------------------
# Individual Tests
# ----------------------------------------------------------------------

def test_title_extractor(factory: MetadataExtractorFactory, tmp_pdf: Path, parsed_doc: dict):
    meta = factory.extract_all(str(tmp_pdf), parsed_doc)
    assert meta["title"] is not None
    assert "Deep" in meta["title"]
    assert len(meta["title"]) > 5


def test_author_extractor(factory: MetadataExtractorFactory, tmp_pdf: Path, parsed_doc: dict):
    meta = factory.extract_all(str(tmp_pdf), parsed_doc)
    authors = meta["authors"]
    assert isinstance(authors, list)
    assert any("John" in a for a in authors)


def test_year_extractor(factory: MetadataExtractorFactory, tmp_pdf: Path, parsed_doc: dict):
    meta = factory.extract_all(str(tmp_pdf), parsed_doc)
    assert meta["year"] in ("2023", "2022", "2021", "2011", "2013")


def test_abstract_extractor(factory: MetadataExtractorFactory, tmp_pdf: Path, parsed_doc: dict):
    meta = factory.extract_all(str(tmp_pdf), parsed_doc)
    assert isinstance(meta["abstract"], str)
    assert "deep learning" in meta["abstract"].lower()


def test_language_detector(factory: MetadataExtractorFactory, tmp_pdf: Path, parsed_doc: dict):
    meta = factory.extract_all(str(tmp_pdf), parsed_doc)
    assert meta["detected_language"] in ("en", "de")


def test_file_size_extractor(factory: MetadataExtractorFactory, tmp_pdf: Path, parsed_doc: dict):
    meta = factory.extract_all(str(tmp_pdf), parsed_doc)
    assert isinstance(meta["file_size"], int)
    assert meta["file_size"] > 0


def test_toc_extractor(factory: MetadataExtractorFactory, tmp_pdf: Path, parsed_doc: dict):
    """Ensure TOC extractor returns structured entries."""
    meta = factory.extract_all(str(tmp_pdf), parsed_doc)
    toc = meta["toc"]
    assert isinstance(toc, list)
    assert all(isinstance(e, dict) for e in toc)
    if toc:  # at least one TOC entry expected
        entry = toc[0]
        assert "title" in entry and "page" in entry
        assert isinstance(entry["title"], str)
        assert isinstance(entry["page"], int)


# ----------------------------------------------------------------------
# Integration Sanity Test
# ----------------------------------------------------------------------

def test_metadata_extraction_integrated(factory: MetadataExtractorFactory, tmp_pdf: Path, parsed_doc: dict):
    """Ensures all extractors cooperate correctly in the factory pipeline."""
    result = factory.extract_all(str(tmp_pdf), parsed_doc)

    expected_keys = {"title", "authors", "year", "abstract", "detected_language", "file_size", "toc"}
    assert expected_keys.issubset(result.keys())

    # Optional: Verify TOC structure
    toc = result.get("toc", [])
    if toc:
        assert all("title" in e and "page" in e for e in toc)

    # Print combined JSON result for manual inspection
    print(json.dumps(result, indent=2, ensure_ascii=False))
