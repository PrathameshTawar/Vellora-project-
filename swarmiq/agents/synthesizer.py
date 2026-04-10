"""
Synthesizer_Agent for SwarmIQ v2.

Composes the final research report in Markdown with:
  - Collapsible <details>/<summary> sections per subtask
  - Inline citations [1], [2], … for every accepted claim
  - Reference list formatted per domain_mode:
      - APA for "research" and "policy"
      - Numbered for "business"
  - ⚠️ Weak Evidence marker for sections where all claims are uncertain
  - "Regulatory Implications" section when domain_mode == "policy"
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Union

from autogen import AssistantAgent

from swarmiq.core.models import (
    AgentError,
    Reference,
    Resolution,
    ScoredClaim,
    SubTask,
    SynthesizerOutput,
)
from swarmiq.utils.rate_limiter import groq_limiter

logger = logging.getLogger(__name__)

_SECTION_SYSTEM_MESSAGE = """\
You are a research report writer. You will be given a subtask description and a list \
of accepted claims with inline citation markers already assigned (e.g. [1], [2]).

Write a concise, well-structured paragraph (3–6 sentences) that synthesises the claims \
into coherent prose. Preserve every citation marker exactly as given — do not add, \
remove, or renumber them. Return only the prose paragraph, no headings, no markdown \
fences, no extra commentary.
"""

_REGULATORY_SYSTEM_MESSAGE = """\
You are a policy analyst. You will be given a research report summary and a list of \
accepted claims. Write a concise "Regulatory Implications" section (3–6 sentences) \
that identifies relevant regulations, compliance considerations, or policy \
recommendations implied by the findings. Return only the prose paragraph.
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_apa_reference(ref: Reference) -> str:
    """Format a Reference as an APA-style entry."""
    authors_str = ", ".join(ref.authors) if ref.authors else "Unknown Author"
    year_str = f"({ref.year})" if ref.year else "(n.d.)"
    title_str = ref.title or "Untitled"
    url_str = ref.url
    return f"{authors_str} {year_str}. {title_str}. Retrieved from {url_str}"


def _format_numbered_reference(ref: Reference) -> str:
    """Format a Reference as a simple numbered entry."""
    title_str = ref.title or "Untitled"
    url_str = ref.url
    return f"[{ref.ref_id}] {title_str}. {url_str}"


class SynthesizerAgent:
    """Compose the final research report from conflict-resolved claims.

    Args:
        llm_config: AutoGen LLM config dict.
    """

    def __init__(self, llm_config: dict) -> None:
        self._llm_config = llm_config
        self._section_agent = AssistantAgent(
            name="synthesizer_section_writer",
            system_message=_SECTION_SYSTEM_MESSAGE,
            llm_config=llm_config,
        )
        self._regulatory_agent = AssistantAgent(
            name="synthesizer_regulatory_writer",
            system_message=_REGULATORY_SYSTEM_MESSAGE,
            llm_config=llm_config,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        resolutions: list[Resolution],
        domain_mode: str,
        subtasks: list[SubTask],
        scored_claims: list[ScoredClaim],
    ) -> Union[SynthesizerOutput, AgentError]:
        """Compose the final Markdown report.

        Args:
            resolutions:   Resolution objects from ConflictResolverAgent.
            domain_mode:   "research", "business", or "policy".
            subtasks:      Original subtasks for section organisation.
            scored_claims: ScoredClaim objects used to look up source URLs.

        Returns:
            SynthesizerOutput with report_markdown and references list.
            AgentError on unexpected failure.
        """
        try:
            return await self._build_report(
                resolutions, domain_mode, subtasks, scored_claims
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("SynthesizerAgent: unexpected error: %s", exc)
            return AgentError(
                agent_type="synthesizer_agent",
                subtask_id=None,
                error_code="UNHANDLED_EXCEPTION",
                message=str(exc),
                timestamp=_now_iso(),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _build_report(
        self,
        resolutions: list[Resolution],
        domain_mode: str,
        subtasks: list[SubTask],
        scored_claims: list[ScoredClaim],
    ) -> SynthesizerOutput:
        """Core report-building logic."""
        # Build claim_id → ScoredClaim lookup for source URL retrieval
        claim_map: dict[str, ScoredClaim] = {sc.claim_id: sc for sc in scored_claims}

        # Build claim_id → Resolution lookup
        resolution_map: dict[str, Resolution] = {r.claim_id: r for r in resolutions}

        # Group resolutions by subtask_id using the scored_claims mapping
        subtask_resolutions: dict[str, list[Resolution]] = {
            st.subtask_id: [] for st in subtasks
        }
        # Resolutions without a matching subtask go into a catch-all bucket
        unmatched: list[Resolution] = []
        for res in resolutions:
            sc = claim_map.get(res.claim_id)
            if sc and sc.subtask_id in subtask_resolutions:
                subtask_resolutions[sc.subtask_id].append(res)
            else:
                unmatched.append(res)

        # Build the global reference list from accepted claims (deduplicated by URL)
        references: list[Reference] = []
        url_to_ref_id: dict[str, int] = {}
        ref_counter = 1

        # Pre-pass: assign ref IDs to all accepted claims in subtask order
        for subtask in subtasks:
            for res in subtask_resolutions.get(subtask.subtask_id, []):
                if res.status == "accepted":
                    sc = claim_map.get(res.claim_id)
                    if sc and sc.source_url not in url_to_ref_id:
                        url_to_ref_id[sc.source_url] = ref_counter
                        references.append(
                            Reference(
                                ref_id=ref_counter,
                                url=sc.source_url,
                                title=sc.source_url,  # title not available; use URL
                            )
                        )
                        ref_counter += 1

        # Build each subtask section
        sections: list[str] = []
        for subtask in subtasks:
            section_md = await self._build_section(
                subtask=subtask,
                resolutions=subtask_resolutions.get(subtask.subtask_id, []),
                claim_map=claim_map,
                url_to_ref_id=url_to_ref_id,
                domain_mode=domain_mode,
            )
            sections.append(section_md)

        # Assemble report body
        report_parts: list[str] = sections

        # Policy mode: append Regulatory Implications section
        if domain_mode == "policy":
            reg_section = await self._build_regulatory_section(
                resolutions=resolutions,
                claim_map=claim_map,
                sections_summary="\n".join(sections),
            )
            report_parts.append(reg_section)

        # Append reference list
        ref_section = self._build_reference_list(references, domain_mode)
        report_parts.append(ref_section)

        report_markdown = "\n\n".join(report_parts)

        return SynthesizerOutput(
            report_markdown=report_markdown,
            references=references,
        )

    async def _build_section(
        self,
        subtask: SubTask,
        resolutions: list[Resolution],
        claim_map: dict[str, ScoredClaim],
        url_to_ref_id: dict[str, int],
        domain_mode: str,
    ) -> str:
        """Build a collapsible Markdown section for one subtask."""
        accepted_claims: list[tuple[Resolution, ScoredClaim, int]] = []
        all_uncertain = True

        for res in resolutions:
            sc = claim_map.get(res.claim_id)
            if res.status != "uncertain":
                all_uncertain = False
            if res.status == "accepted" and sc:
                ref_id = url_to_ref_id.get(sc.source_url, 0)
                accepted_claims.append((res, sc, ref_id))

        # If no resolutions at all, treat as all uncertain
        if not resolutions:
            all_uncertain = True

        # Build the prose content
        if accepted_claims:
            prose = await self._write_section_prose(subtask, accepted_claims)
        else:
            prose = "_No accepted claims available for this subtask._"

        # Weak evidence marker
        weak_evidence_marker = ""
        if all_uncertain:
            weak_evidence_marker = "\n\n> ⚠️ Weak Evidence — all claims for this section are uncertain."

        content = prose + weak_evidence_marker

        # Wrap in collapsible <details> block
        section_md = (
            f"<details>\n"
            f"<summary><strong>{subtask.description}</strong></summary>\n\n"
            f"{content}\n\n"
            f"</details>"
        )
        return section_md

    async def _write_section_prose(
        self,
        subtask: SubTask,
        accepted_claims: list[tuple[Resolution, ScoredClaim, int]],
    ) -> str:
        """Use the LLM to write prose for a section, preserving citation markers."""
        claims_text = "\n".join(
            f"- [{ref_id}] {sc.claim_text}"
            for _, sc, ref_id in accepted_claims
        )
        prompt = (
            f"Subtask: {subtask.description}\n\n"
            f"Accepted claims with citation markers:\n{claims_text}\n\n"
            "Write a concise synthesis paragraph that incorporates all citation markers."
        )

        try:
            groq_limiter.wait_if_needed()
            reply = await self._section_agent.a_generate_reply(
                messages=[{"role": "user", "content": prompt}]
            )
            if isinstance(reply, dict):
                text = reply.get("content", "")
            else:
                text = str(reply) if reply is not None else ""
            return text.strip() or self._fallback_prose(accepted_claims)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SynthesizerAgent: LLM section prose failed: %s. Using fallback.", exc
            )
            return self._fallback_prose(accepted_claims)

    def _fallback_prose(
        self,
        accepted_claims: list[tuple[Resolution, ScoredClaim, int]],
    ) -> str:
        """Generate simple prose without LLM when the call fails."""
        sentences = [
            f"{sc.claim_text} [{ref_id}]."
            for _, sc, ref_id in accepted_claims
        ]
        return " ".join(sentences)

    async def _build_regulatory_section(
        self,
        resolutions: list[Resolution],
        claim_map: dict[str, ScoredClaim],
        sections_summary: str,
    ) -> str:
        """Generate the Regulatory Implications section for policy mode."""
        accepted_texts = [
            claim_map[r.claim_id].claim_text
            for r in resolutions
            if r.status == "accepted" and r.claim_id in claim_map
        ]
        claims_summary = (
            "\n".join(f"- {t}" for t in accepted_texts)
            if accepted_texts
            else "No accepted claims available."
        )
        prompt = (
            f"Research summary:\n{sections_summary[:2000]}\n\n"
            f"Accepted claims:\n{claims_summary}\n\n"
            "Write the Regulatory Implications section."
        )

        try:
            groq_limiter.wait_if_needed()
            reply = await self._regulatory_agent.a_generate_reply(
                messages=[{"role": "user", "content": prompt}]
            )
            if isinstance(reply, dict):
                prose = reply.get("content", "")
            else:
                prose = str(reply) if reply is not None else ""
            prose = prose.strip() or "Regulatory implications require further analysis."
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SynthesizerAgent: LLM regulatory section failed: %s. Using fallback.", exc
            )
            prose = "Regulatory implications require further analysis."

        return f"## Regulatory Implications\n\n{prose}"

    def _build_reference_list(
        self,
        references: list[Reference],
        domain_mode: str,
    ) -> str:
        """Build the formatted reference list section."""
        if not references:
            return "## References\n\n_No references._"

        use_apa = domain_mode in ("research", "policy")

        lines: list[str] = []
        for ref in references:
            if use_apa:
                lines.append(_format_apa_reference(ref))
            else:
                lines.append(_format_numbered_reference(ref))

        formatted = "\n".join(lines)
        return f"## References\n\n{formatted}"
