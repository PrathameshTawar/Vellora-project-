"""
Visualization_Agent for SwarmIQ v2.

Detects numerical and temporal data in claim text and generates
Plotly JSON specs (or matplotlib PNG bytes as fallback).

If no structured data is found, logs a notice and returns an empty
figures list without raising an exception.
"""
from __future__ import annotations

import base64
import io
import logging
import re
import uuid
from dataclasses import dataclass, field

from swarmiq.core.models import Figure, ScoredClaim

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class VisualizationOutput:
    figures: list[Figure] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Regex patterns for structured data detection
# ---------------------------------------------------------------------------

# Numerical patterns: "X% of", "increased by X", "X million", "X billion",
# numbers with units (e.g. "42 kg", "3.5 GHz")
_NUMERICAL_PATTERNS = [
    re.compile(r"\b(\d+(?:\.\d+)?)\s*%\s*of\b", re.IGNORECASE),
    re.compile(r"\bincreased?\s+by\s+(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*(million|billion|trillion)\b", re.IGNORECASE),
    re.compile(r"\b(\d+(?:\.\d+)?)\s+[a-zA-Z]{1,5}\b"),  # number + short unit
]

# Temporal patterns: "in 2020", "by 2025", "from 2010 to 2020", year ranges
_TEMPORAL_PATTERNS = [
    re.compile(r"\bin\s+((?:19|20)\d{2})\b", re.IGNORECASE),
    re.compile(r"\bby\s+((?:19|20)\d{2})\b", re.IGNORECASE),
    re.compile(r"\b((?:19|20)\d{2})\s*[-–]\s*((?:19|20)\d{2})\b"),
    re.compile(r"\bfrom\s+((?:19|20)\d{2})\b", re.IGNORECASE),
]


def _extract_numbers(text: str) -> list[float]:
    """Return all numeric values found by the numerical patterns."""
    values: list[float] = []
    for pattern in _NUMERICAL_PATTERNS:
        for match in pattern.finditer(text):
            try:
                values.append(float(match.group(1)))
            except (IndexError, ValueError):
                pass
    return values


def _extract_years(text: str) -> list[int]:
    """Return all year values found by the temporal patterns."""
    years: set[int] = set()
    for pattern in _TEMPORAL_PATTERNS:
        for match in pattern.finditer(text):
            for group in match.groups():
                if group and re.fullmatch(r"(?:19|20)\d{2}", group):
                    years.add(int(group))
    return sorted(years)


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _build_plotly_bar(labels: list[str], values: list[float]) -> str:
    """Return a Plotly bar chart as a JSON string."""
    import plotly.graph_objects as go  # type: ignore

    fig = go.Figure(
        data=[go.Bar(x=labels, y=values)],
        layout=go.Layout(title="Numerical Data from Claims"),
    )
    return fig.to_json()


def _build_plotly_line(years: list[int], values: list[float]) -> str:
    """Return a Plotly line chart as a JSON string."""
    import plotly.graph_objects as go  # type: ignore

    fig = go.Figure(
        data=[go.Scatter(x=years, y=values, mode="lines+markers")],
        layout=go.Layout(title="Temporal Data from Claims", xaxis_title="Year"),
    )
    return fig.to_json()


def _build_matplotlib_bar(labels: list[str], values: list[float]) -> bytes:
    """Return a matplotlib bar chart as base64-encoded PNG bytes."""
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title("Numerical Data from Claims")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read())


def _build_matplotlib_line(years: list[int], values: list[float]) -> bytes:
    """Return a matplotlib line chart as base64-encoded PNG bytes."""
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    fig, ax = plt.subplots()
    ax.plot(years, values, marker="o")
    ax.set_title("Temporal Data from Claims")
    ax.set_xlabel("Year")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read())


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class VisualizationAgent:
    """Generate charts from structured numerical/temporal data in claims.

    The ``generate`` method is synchronous and does not require an LLM.
    """

    def generate(self, claims: list[ScoredClaim]) -> VisualizationOutput:
        """Detect structured data in claims and produce figures.

        Args:
            claims: Scored claims from the pipeline.

        Returns:
            VisualizationOutput with zero or more Figure objects.
            Never raises an exception — returns empty figures list when no
            structured data is found.
        """
        try:
            return self._generate(claims)
        except Exception:  # noqa: BLE001
            logger.exception("VisualizationAgent: unexpected error during generation")
            return VisualizationOutput(figures=[])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate(self, claims: list[ScoredClaim]) -> VisualizationOutput:
        numerical_claims: list[tuple[str, list[float]]] = []
        temporal_claims: list[tuple[list[int], list[float]]] = []

        for claim in claims:
            text = claim.claim_text
            nums = _extract_numbers(text)
            years = _extract_years(text)

            if years:
                # Pair years with extracted numbers (or use index as proxy value)
                paired_values = nums[:len(years)] if nums else list(range(1, len(years) + 1))
                # Pad if needed
                while len(paired_values) < len(years):
                    paired_values.append(float(len(paired_values) + 1))
                temporal_claims.append((years, paired_values[:len(years)]))
            elif nums:
                numerical_claims.append((text[:40], nums))

        if not numerical_claims and not temporal_claims:
            logger.info(
                "VisualizationAgent: no structured numerical or temporal data found "
                "in %d claim(s); returning empty figures list.",
                len(claims),
            )
            return VisualizationOutput(figures=[])

        figures: list[Figure] = []

        # Build numerical bar chart
        if numerical_claims:
            labels = [label for label, _ in numerical_claims]
            values = [nums[0] for _, nums in numerical_claims]
            figures.append(self._make_figure("plotly", labels=labels, values=values, chart="bar"))

        # Build temporal line chart
        if temporal_claims:
            all_years: list[int] = []
            all_values: list[float] = []
            for years, vals in temporal_claims:
                all_years.extend(years)
                all_values.extend(vals)
            # Sort by year
            paired = sorted(zip(all_years, all_values), key=lambda t: t[0])
            sorted_years = [p[0] for p in paired]
            sorted_values = [p[1] for p in paired]
            figures.append(
                self._make_figure("plotly", years=sorted_years, values=sorted_values, chart="line")
            )

        return VisualizationOutput(figures=figures)

    def _make_figure(
        self,
        preferred: str,
        chart: str,
        values: list[float] | None = None,
        labels: list[str] | None = None,
        years: list[int] | None = None,
    ) -> Figure:
        """Try to build a Plotly figure; fall back to matplotlib on ImportError."""
        figure_id = str(uuid.uuid4())

        try:
            if chart == "bar":
                data = _build_plotly_bar(labels or [], values or [])
            else:
                data = _build_plotly_line(years or [], values or [])
            return Figure(figure_id=figure_id, figure_type="plotly", data=data)
        except ImportError:
            logger.warning(
                "VisualizationAgent: plotly not available, falling back to matplotlib."
            )
            if chart == "bar":
                data = _build_matplotlib_bar(labels or [], values or [])
            else:
                data = _build_matplotlib_line(years or [], values or [])
            return Figure(figure_id=figure_id, figure_type="matplotlib", data=data)
