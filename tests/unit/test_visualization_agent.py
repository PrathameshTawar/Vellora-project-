"""
Unit tests for VisualizationAgent.

Tests:
1.  Claim with "increased by 42%" → generates at least 1 figure
2.  Claim with "3.5 million users" → generates at least 1 figure
3.  Claim with "in 2020" → generates at least 1 figure (temporal)
4.  Claim with "from 2010 to 2020" → generates at least 1 figure (temporal range)
5.  Claim with no numbers or years → returns empty figures list
6.  Empty claims list → returns empty figures list
7.  Figure type is "plotly" when plotly is available
8.  Exception during figure generation → returns empty figures list (no crash)
9.  Multiple claims with numerical data → generates figures (bar chart)
10. Multiple claims with temporal data → generates figures (line chart)

Requirements: 8.1, 8.4
"""
from __future__ import annotations

import sys
import types
import uuid
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out autogen before any swarmiq imports
# ---------------------------------------------------------------------------
_autogen_mod = types.ModuleType("autogen")
_autogen_mod.AssistantAgent = MagicMock  # type: ignore[attr-defined]
sys.modules.setdefault("autogen", _autogen_mod)

from swarmiq.agents.visualization import VisualizationAgent, VisualizationOutput
from swarmiq.core.models import Figure, ScoredClaim

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_PLOTLY_JSON = '{"data": [], "layout": {}}'


def _claim(text: str) -> ScoredClaim:
    return ScoredClaim(
        claim_id=str(uuid.uuid4()),
        claim_text=text,
        confidence=0.8,
        source_url="https://example.com",
        subtask_id=str(uuid.uuid4()),
        credibility_score=0.8,
    )


def _agent() -> VisualizationAgent:
    return VisualizationAgent()


# ---------------------------------------------------------------------------
# Test 1: "increased by 42%" → at least 1 figure
# ---------------------------------------------------------------------------


def test_percentage_claim_generates_figure():
    """Claim with 'increased by 42%' generates at least 1 figure."""
    with patch(
        "swarmiq.agents.visualization._build_plotly_bar", return_value=_FAKE_PLOTLY_JSON
    ):
        result = _agent().generate([_claim("Revenue increased by 42% last quarter.")])

    assert isinstance(result, VisualizationOutput)
    assert len(result.figures) >= 1


# ---------------------------------------------------------------------------
# Test 2: "3.5 million users" → at least 1 figure
# ---------------------------------------------------------------------------


def test_million_claim_generates_figure():
    """Claim with '3.5 million users' generates at least 1 figure."""
    with patch(
        "swarmiq.agents.visualization._build_plotly_bar", return_value=_FAKE_PLOTLY_JSON
    ):
        result = _agent().generate([_claim("The platform reached 3.5 million users.")])

    assert isinstance(result, VisualizationOutput)
    assert len(result.figures) >= 1


# ---------------------------------------------------------------------------
# Test 3: "in 2020" → at least 1 figure (temporal)
# ---------------------------------------------------------------------------


def test_temporal_single_year_generates_figure():
    """Claim with 'in 2020' generates at least 1 figure."""
    with patch(
        "swarmiq.agents.visualization._build_plotly_line", return_value=_FAKE_PLOTLY_JSON
    ):
        result = _agent().generate([_claim("The policy was enacted in 2020.")])

    assert isinstance(result, VisualizationOutput)
    assert len(result.figures) >= 1


# ---------------------------------------------------------------------------
# Test 4: "from 2010 to 2020" → at least 1 figure (temporal range)
# ---------------------------------------------------------------------------


def test_temporal_range_generates_figure():
    """Claim with 'from 2010 to 2020' generates at least 1 figure."""
    with patch(
        "swarmiq.agents.visualization._build_plotly_line", return_value=_FAKE_PLOTLY_JSON
    ):
        result = _agent().generate([_claim("Emissions rose from 2010 to 2020.")])

    assert isinstance(result, VisualizationOutput)
    assert len(result.figures) >= 1


# ---------------------------------------------------------------------------
# Test 5: No numbers or years → empty figures list
# ---------------------------------------------------------------------------


def test_no_data_claim_returns_empty_figures():
    """Claim with no numbers or years returns empty figures list."""
    result = _agent().generate([_claim("The sky is blue and the grass is green.")])

    assert isinstance(result, VisualizationOutput)
    assert result.figures == []


# ---------------------------------------------------------------------------
# Test 6: Empty claims list → empty figures list
# ---------------------------------------------------------------------------


def test_empty_claims_returns_empty_figures():
    """Empty claims list returns empty figures list."""
    result = _agent().generate([])

    assert isinstance(result, VisualizationOutput)
    assert result.figures == []


# ---------------------------------------------------------------------------
# Test 7: Figure type is "plotly" when plotly is available
# ---------------------------------------------------------------------------


def test_figure_type_is_plotly_when_available():
    """Figure type is 'plotly' when plotly is available."""
    with patch(
        "swarmiq.agents.visualization._build_plotly_bar", return_value=_FAKE_PLOTLY_JSON
    ):
        result = _agent().generate([_claim("Sales increased by 30% year over year.")])

    assert len(result.figures) >= 1
    for fig in result.figures:
        assert isinstance(fig, Figure)
        assert fig.figure_type == "plotly"


# ---------------------------------------------------------------------------
# Test 8: Exception during figure generation → empty figures list (no crash)
# ---------------------------------------------------------------------------


def test_exception_during_generation_returns_empty_figures():
    """Exception during figure generation returns empty figures list without crashing."""
    with patch(
        "swarmiq.agents.visualization._build_plotly_bar",
        side_effect=RuntimeError("unexpected failure"),
    ), patch(
        "swarmiq.agents.visualization._build_plotly_line",
        side_effect=RuntimeError("unexpected failure"),
    ):
        result = _agent().generate([_claim("Revenue increased by 50% in 2021.")])

    assert isinstance(result, VisualizationOutput)
    assert result.figures == []


# ---------------------------------------------------------------------------
# Test 9: Multiple claims with numerical data → bar chart figure
# ---------------------------------------------------------------------------


def test_multiple_numerical_claims_generate_bar_chart():
    """Multiple claims with numerical data generate a bar chart figure."""
    claims = [
        _claim("Revenue increased by 15% after the restructuring."),
        _claim("Productivity increased by 28% this quarter."),
        _claim("Customer satisfaction increased by 10%."),
    ]

    with patch(
        "swarmiq.agents.visualization._build_plotly_bar", return_value=_FAKE_PLOTLY_JSON
    ) as mock_bar:
        result = _agent().generate(claims)

    assert isinstance(result, VisualizationOutput)
    assert len(result.figures) >= 1
    mock_bar.assert_called_once()
    bar_fig = next(f for f in result.figures if f.figure_type == "plotly")
    assert bar_fig.data == _FAKE_PLOTLY_JSON


# ---------------------------------------------------------------------------
# Test 10: Multiple claims with temporal data → line chart figure
# ---------------------------------------------------------------------------


def test_multiple_temporal_claims_generate_line_chart():
    """Multiple claims with temporal data generate a line chart figure."""
    claims = [
        _claim("The initiative launched in 2015."),
        _claim("Adoption peaked in 2018."),
        _claim("The program concluded in 2022."),
    ]

    with patch(
        "swarmiq.agents.visualization._build_plotly_line", return_value=_FAKE_PLOTLY_JSON
    ) as mock_line:
        result = _agent().generate(claims)

    assert isinstance(result, VisualizationOutput)
    assert len(result.figures) >= 1
    mock_line.assert_called_once()
    line_fig = next(f for f in result.figures if f.figure_type == "plotly")
    assert line_fig.data == _FAKE_PLOTLY_JSON
