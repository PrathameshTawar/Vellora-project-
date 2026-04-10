"""
Citation formatting utilities for SwarmIQ v2 Export_Module.
Supports APA and MLA citation styles.
"""
from __future__ import annotations

from swarmiq.core.models import Reference


def format_apa(references: list[Reference]) -> str:
    """Format a list of references in APA style.

    Format: Author(s). (Year). Title. Retrieved from URL
    """
    lines: list[str] = []
    for ref in references:
        authors = ", ".join(ref.authors) if ref.authors else "Unknown"
        year = str(ref.year) if ref.year is not None else "n.d."
        title = ref.title or "Unknown"
        url = ref.url or ""
        lines.append(f"{authors}. ({year}). {title}. Retrieved from {url}")
    return "\n".join(lines)


def format_mla(references: list[Reference]) -> str:
    """Format a list of references in MLA style.

    Format: Author(s). "Title." URL, Year.
    """
    lines: list[str] = []
    for ref in references:
        authors = ", ".join(ref.authors) if ref.authors else "Unknown"
        year = str(ref.year) if ref.year is not None else "n.d."
        title = ref.title or "Unknown"
        url = ref.url or ""
        lines.append(f'{authors}. "{title}." {url}, {year}.')
    return "\n".join(lines)
