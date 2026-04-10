"""
Unit tests for Export_Module (ExportModule, format_apa, format_mla).

Tests:
1.  Markdown export returns a string containing the report text
2.  Markdown export with APA references contains "Retrieved from"
3.  Markdown export with MLA references contains the MLA format (quotes around title)
4.  JSON export returns valid JSON string containing report_markdown
5.  JSON export contains references array
6.  APA and MLA format_apa/format_mla produce different strings for the same reference
7.  format_apa with known reference produces expected format
8.  format_mla with known reference produces expected format
9.  PDF export raises ExportError when fpdf raises an exception
10. PPT export raises ExportError when pptx raises an exception
11. Unsupported format raises ValueError
12. Markdown export with empty references still returns the report text

Requirements: 10.1, 10.2, 10.4
"""
from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from swarmiq.core.models import Figure, Reference
from swarmiq.export.citations import format_apa, format_mla
from swarmiq.export.exporter import ExportError, ExportModule

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

REPORT_TEXT = "# Climate Change\n\nThis report covers the latest findings on climate change."

KNOWN_REF = Reference(
    ref_id=1,
    url="https://example.com/paper",
    title="Climate Dynamics",
    authors=["Smith, J.", "Doe, A."],
    year=2023,
)

KNOWN_REF_NO_YEAR = Reference(
    ref_id=2,
    url="https://example.com/other",
    title="Ocean Temperatures",
    authors=["Brown, K."],
    year=None,
)


def _exporter() -> ExportModule:
    return ExportModule()


# ---------------------------------------------------------------------------
# Test 1: Markdown export returns a string containing the report text
# ---------------------------------------------------------------------------


def test_markdown_export_contains_report_text():
    """Markdown export returns a string containing the report text."""
    result = _exporter().export(REPORT_TEXT, [], [], "markdown")
    assert isinstance(result, str)
    assert REPORT_TEXT in result


# ---------------------------------------------------------------------------
# Test 2: Markdown export with APA references contains "Retrieved from"
# ---------------------------------------------------------------------------


def test_markdown_export_apa_contains_retrieved_from():
    """Markdown export with APA references contains 'Retrieved from'."""
    result = _exporter().export(REPORT_TEXT, [KNOWN_REF], [], "markdown", citation_style="apa")
    assert isinstance(result, str)
    assert "Retrieved from" in result


# ---------------------------------------------------------------------------
# Test 3: Markdown export with MLA references contains quotes around title
# ---------------------------------------------------------------------------


def test_markdown_export_mla_contains_quoted_title():
    """Markdown export with MLA references contains the MLA format (quotes around title)."""
    result = _exporter().export(REPORT_TEXT, [KNOWN_REF], [], "markdown", citation_style="mla")
    assert isinstance(result, str)
    # MLA wraps the title in double quotes: "Title."
    assert f'"{KNOWN_REF.title}."' in result


# ---------------------------------------------------------------------------
# Test 4: JSON export returns valid JSON string containing report_markdown
# ---------------------------------------------------------------------------


def test_json_export_is_valid_json_with_report_markdown():
    """JSON export returns valid JSON string containing report_markdown."""
    result = _exporter().export(REPORT_TEXT, [KNOWN_REF], [], "json")
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["report_markdown"] == REPORT_TEXT


# ---------------------------------------------------------------------------
# Test 5: JSON export contains references array
# ---------------------------------------------------------------------------


def test_json_export_contains_references_array():
    """JSON export contains references array."""
    result = _exporter().export(REPORT_TEXT, [KNOWN_REF], [], "json")
    parsed = json.loads(result)
    assert "references" in parsed
    assert isinstance(parsed["references"], list)
    assert len(parsed["references"]) == 1
    assert parsed["references"][0]["title"] == KNOWN_REF.title


# ---------------------------------------------------------------------------
# Test 6: APA and MLA produce different strings for the same reference
# ---------------------------------------------------------------------------


def test_apa_and_mla_produce_different_strings():
    """APA and MLA format_apa/format_mla produce different strings for the same reference."""
    apa = format_apa([KNOWN_REF])
    mla = format_mla([KNOWN_REF])
    assert apa != mla


# ---------------------------------------------------------------------------
# Test 7: format_apa with known reference produces expected format
# ---------------------------------------------------------------------------


def test_format_apa_known_reference():
    """format_apa with known reference produces expected APA format."""
    result = format_apa([KNOWN_REF])
    assert "Smith, J., Doe, A." in result
    assert "(2023)" in result
    assert "Climate Dynamics" in result
    assert "Retrieved from" in result
    assert KNOWN_REF.url in result


# ---------------------------------------------------------------------------
# Test 8: format_mla with known reference produces expected format
# ---------------------------------------------------------------------------


def test_format_mla_known_reference():
    """format_mla with known reference produces expected MLA format."""
    result = format_mla([KNOWN_REF])
    assert "Smith, J., Doe, A." in result
    # MLA format: "Title." (title in double quotes followed by period)
    assert f'"{KNOWN_REF.title}."' in result
    assert KNOWN_REF.url in result
    assert "2023" in result


# ---------------------------------------------------------------------------
# Test 9: PDF export raises ExportError when fpdf raises an exception
# ---------------------------------------------------------------------------


def test_pdf_export_raises_export_error_on_fpdf_failure():
    """PDF export raises ExportError when fpdf raises an exception."""
    mock_fpdf_instance = MagicMock()
    mock_fpdf_instance.add_page.side_effect = RuntimeError("fpdf internal error")
    mock_fpdf_class = MagicMock(return_value=mock_fpdf_instance)

    fpdf_mod = types.ModuleType("fpdf")
    fpdf_mod.FPDF = mock_fpdf_class  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"fpdf": fpdf_mod}):
        with pytest.raises(ExportError, match="PDF rendering failed"):
            _exporter().export(REPORT_TEXT, [KNOWN_REF], [], "pdf")


# ---------------------------------------------------------------------------
# Test 10: PPT export raises ExportError when pptx raises an exception
# ---------------------------------------------------------------------------


def test_ppt_export_raises_export_error_on_pptx_failure():
    """PPT export raises ExportError when pptx raises an exception."""
    mock_prs_instance = MagicMock()
    mock_prs_instance.slide_layouts = [MagicMock(), MagicMock()]
    mock_prs_instance.slides.add_slide.side_effect = RuntimeError("pptx internal error")
    mock_prs_class = MagicMock(return_value=mock_prs_instance)

    mock_pptx_util = types.ModuleType("pptx.util")
    mock_pptx_util.Inches = MagicMock(return_value=1)  # type: ignore[attr-defined]
    mock_pptx_util.Pt = MagicMock(return_value=11)  # type: ignore[attr-defined]

    pptx_mod = types.ModuleType("pptx")
    pptx_mod.Presentation = mock_prs_class  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"pptx": pptx_mod, "pptx.util": mock_pptx_util}):
        with pytest.raises(ExportError, match="PPT rendering failed"):
            _exporter().export(REPORT_TEXT, [KNOWN_REF], [], "ppt")


# ---------------------------------------------------------------------------
# Test 11: Unsupported format raises ValueError
# ---------------------------------------------------------------------------


def test_unsupported_format_raises_value_error():
    """Unsupported format raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported format"):
        _exporter().export(REPORT_TEXT, [], [], "docx")


# ---------------------------------------------------------------------------
# Test 12: Markdown export with empty references still returns the report text
# ---------------------------------------------------------------------------


def test_markdown_export_empty_references_returns_report_text():
    """Markdown export with empty references still returns the report text."""
    result = _exporter().export(REPORT_TEXT, [], [], "markdown")
    assert isinstance(result, str)
    assert REPORT_TEXT in result
    assert "## References" not in result
