"""
Property-based tests for Export_Module (Properties 22, 23).

Feature: swarmiq-v2
Validates: Requirements 10.1, 10.2
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Stub out heavy dependencies before any swarmiq imports
# ---------------------------------------------------------------------------

def _stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


for _dep in [
    "autogen",
    "sentence_transformers",
    "pinecone",
    "tavily",
    "serpapi",
    "transformers",
    "plotly",
    "plotly.graph_objects",
    "matplotlib",
    "matplotlib.pyplot",
]:
    if _dep not in sys.modules:
        _stub(_dep)

# ---------------------------------------------------------------------------

from swarmiq.core.models import Reference  # noqa: E402
from swarmiq.export.citations import format_apa, format_mla  # noqa: E402
from swarmiq.export.exporter import ExportModule  # noqa: E402

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def reference_strategy(draw) -> Reference:
    return Reference(
        ref_id=draw(st.integers(min_value=1)),
        url="https://example.com",
        title=draw(st.text(min_size=1)),
        authors=draw(st.lists(st.text(min_size=1), min_size=0, max_size=3)),
        year=draw(st.one_of(st.none(), st.integers(min_value=1900, max_value=2100))),
    )


# ---------------------------------------------------------------------------
# Property 22: Export module produces output for all supported formats
# ---------------------------------------------------------------------------


class TestExportModuleProducesOutputForAllFormats:
    """
    # Feature: swarmiq-v2, Property 22: Export module produces output for all
    # supported formats — for any valid report, calling the Export_Module with
    # each of the formats {markdown, pdf, ppt, json} must produce a non-empty
    # output without raising an exception.
    """

    @given(
        report_markdown=st.text(min_size=1, max_size=200),
        references=st.lists(reference_strategy(), min_size=1, max_size=5),
    )
    @settings(max_examples=100)
    def test_export_produces_nonempty_output_for_all_formats(
        self, report_markdown: str, references: list[Reference]
    ):
        # Feature: swarmiq-v2, Property 22: Export module produces output for all supported formats
        exporter = ExportModule()
        figures: list = []

        # --- Markdown ---
        md_output = exporter.export(report_markdown, references, figures, "markdown")
        assert isinstance(md_output, str) and len(md_output) > 0, (
            "Markdown export must return a non-empty string"
        )

        # --- JSON ---
        json_output = exporter.export(report_markdown, references, figures, "json")
        assert isinstance(json_output, str) and len(json_output) > 0, (
            "JSON export must return a non-empty string"
        )

        # --- PDF (mock fpdf.FPDF to avoid needing fpdf2 installed) ---
        mock_pdf_bytes = b"%PDF-1.4 minimal"

        mock_fpdf_instance = MagicMock()
        mock_fpdf_instance.output.return_value = mock_pdf_bytes
        mock_fpdf_class = MagicMock(return_value=mock_fpdf_instance)

        fpdf_mod = types.ModuleType("fpdf")
        fpdf_mod.FPDF = mock_fpdf_class  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"fpdf": fpdf_mod}):
            pdf_output = exporter.export(report_markdown, references, figures, "pdf")
        assert isinstance(pdf_output, bytes) and len(pdf_output) > 0, (
            "PDF export must return non-empty bytes"
        )

        # --- PPT (mock pptx.Presentation to avoid needing python-pptx installed) ---
        mock_ppt_bytes = b"PK\x03\x04minimal-pptx"

        mock_slide = MagicMock()
        mock_slide.shapes.title.text = ""
        mock_slide.placeholders = {1: MagicMock()}
        mock_slide.placeholders[1].text_frame.word_wrap = True
        mock_slide.placeholders[1].text_frame.text = ""

        mock_prs_instance = MagicMock()
        mock_prs_instance.slide_layouts = [MagicMock(), MagicMock()]
        mock_prs_instance.slides.add_slide.return_value = mock_slide

        def _mock_save(buf):
            buf.write(mock_ppt_bytes)

        mock_prs_instance.save.side_effect = _mock_save
        mock_prs_class = MagicMock(return_value=mock_prs_instance)

        mock_pptx_util = types.ModuleType("pptx.util")
        mock_pptx_util.Inches = MagicMock(return_value=1)  # type: ignore[attr-defined]
        mock_pptx_util.Pt = MagicMock(return_value=11)  # type: ignore[attr-defined]

        pptx_mod = types.ModuleType("pptx")
        pptx_mod.Presentation = mock_prs_class  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"pptx": pptx_mod, "pptx.util": mock_pptx_util}):
            ppt_output = exporter.export(report_markdown, references, figures, "ppt")
        assert isinstance(ppt_output, bytes) and len(ppt_output) > 0, (
            "PPT export must return non-empty bytes"
        )


# ---------------------------------------------------------------------------
# Property 23: Citation formatting produces distinct APA and MLA outputs
# ---------------------------------------------------------------------------


class TestCitationFormattingProducesDistinctApaMlaOutputs:
    """
    # Feature: swarmiq-v2, Property 23: Citation formatting produces distinct
    # APA and MLA outputs — for any non-empty reference list, formatting it in
    # APA style and formatting it in MLA style must each produce a non-empty
    # string, and the two strings must not be identical.
    """

    @given(
        references=st.lists(reference_strategy(), min_size=1, max_size=5),
    )
    @settings(max_examples=100)
    def test_apa_and_mla_are_nonempty_and_distinct(self, references: list[Reference]):
        # Feature: swarmiq-v2, Property 23: Citation formatting produces distinct APA and MLA outputs
        apa = format_apa(references)
        mla = format_mla(references)

        assert isinstance(apa, str) and len(apa) > 0, (
            "APA output must be a non-empty string"
        )
        assert isinstance(mla, str) and len(mla) > 0, (
            "MLA output must be a non-empty string"
        )
        assert apa != mla, (
            "APA and MLA outputs must not be identical for the same reference list"
        )
