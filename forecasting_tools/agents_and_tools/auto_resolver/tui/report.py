"""Shared report / confusion-matrix logic for the TUI.

Builds a ``ResolutionAssessmentReport`` from a dict of ``QuestionItem``
objects so both the home panel and the markdown export can reuse the same
data and formatting that ``assess.py`` already provides.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from forecasting_tools.agents_and_tools.auto_resolver.assess import (
    QuestionAssessmentResult,
    ResolutionAssessmentReport,
)
from forecasting_tools.agents_and_tools.auto_resolver.tui.models import QuestionItem
from forecasting_tools.data_models.questions import (
    BinaryResolution,
    CanceledResolution,
    ResolutionType,
)


# ------------------------------------------------------------------
# Mapping resolver status strings to typed resolutions
# ------------------------------------------------------------------

def _status_str_to_resolution(status: str | None) -> Optional[ResolutionType]:
    """Convert a resolution_status_str (e.g. 'TRUE') to a typed resolution."""
    if status is None:
        return 
    s = status.strip().upper()
    if s == "TRUE":
        return True
    elif s == "FALSE":
        return False
    elif s == "AMBIGUOUS":
        return CanceledResolution.AMBIGUOUS
    elif s == "ANNULLED":
        return CanceledResolution.ANNULLED
    elif s == "NOT_YET_RESOLVABLE":
        return None


# ------------------------------------------------------------------
# Build a ResolutionAssessmentReport from TUI items
# ------------------------------------------------------------------

def build_report_from_items(
    items: dict[int, QuestionItem],
) -> ResolutionAssessmentReport:
    """Populate a ``ResolutionAssessmentReport`` from the TUI question items.

    Only completed items whose questions have a ground-truth resolution are
    included in the confusion matrix.  Items that are still pending / running /
    errored are silently skipped.
    """
    report = ResolutionAssessmentReport()

    for post_id, item in items.items():
        if item.status != "completed":
            continue

        true_resolution = item.question.typed_resolution
        predicted_resolution = _status_str_to_resolution(item.resolution_status_str)

        key_evidence: list[str] | None = None
        if item.resolution_metadata:
            key_evidence = item.resolution_metadata.get("key_evidence")

        outcome_category: str | None = None

        # Classify into the confusion matrix — mirrors assess.py logic
        if isinstance(true_resolution, bool):
            match true_resolution, predicted_resolution:
                # Positive actual
                case True, True:
                    report.pp.append(post_id)
                    outcome_category = "True Positive"
                case True, False:
                    report.np.append(post_id)
                    outcome_category = "False Negative"
                case True, None:
                    report.xp.append(post_id)
                    outcome_category = "Missed Positive"
                case True, CanceledResolution():
                    report.cp.append(post_id)
                    outcome_category = "Positive Incorrectly Predicted as Cancelled"
                # Negative actual
                case False, True:
                    report.pn.append(post_id)
                    outcome_category = "False Positive"
                case False, False:
                    report.nn.append(post_id)
                    outcome_category = "True Negative"
                case False, None:
                    report.xn.append(post_id)
                    outcome_category = "Missed Negative"
                case False, CanceledResolution():
                    report.cn.append(post_id)
                    outcome_category = "Negative Incorrectly Predicted as Cancelled"
                case _:
                    if true_resolution is True:
                        report.xp.append(post_id)
                        outcome_category = "Unmatched - Positive"
                    else:
                        report.xn.append(post_id)
                        outcome_category = "Unmatched - Negative"

        elif isinstance(true_resolution, CanceledResolution):
            match predicted_resolution:
                case True:
                    report.pc.append(post_id)
                    outcome_category = "Cancelled Incorrectly Predicted as Positive"
                case False:
                    report.nc.append(post_id)
                    outcome_category = "Cancelled Incorrectly Predicted as Negative"
                case CanceledResolution():
                    report.cc.append(post_id)
                    outcome_category = "Correct Cancel"
                case None:
                    report.xc.append(post_id)
                    outcome_category = "Cancelled Not Answered"
                case _:
                    report.xc.append(post_id)
                    outcome_category = "Unmatched - Cancelled"

        elif true_resolution is None:
            match predicted_resolution:
                case True:
                    report.nyr_p.append(post_id)
                    outcome_category = "Not Yet Resolvable Predicted as Positive"
                case False:
                    report.nyr_n.append(post_id)
                    outcome_category = "Not Yet Resolvable Predicted as Negative"
                case CanceledResolution():
                    report.nyr_c.append(post_id)
                    outcome_category = "Not Yet Resolvable Predicted as Cancelled"
                case None:
                    report.nyr_nyr.append(post_id)
                    outcome_category = "Correct Not Yet Resolvable"
                case _:
                    report.nyr_x.append(post_id)
                    outcome_category = "Unmatched - Not Yet Resolvable"
        else:
            # Unknown resolution type — skip from matrix
            continue

        question_result = QuestionAssessmentResult(
            question_id=post_id,
            question_title=(item.question.question_text or "No title")[:100],
            question_text=item.question.question_text or "No text available",
            question_url=item.question.page_url or f"https://metaculus.com/{post_id}",
            actual_resolution=true_resolution,
            predicted_resolution=predicted_resolution,
            key_evidence=key_evidence,
            outcome_category=outcome_category,
        )
        report.question_results[post_id] = question_result

    return report


# ------------------------------------------------------------------
# Full markdown report (mirrors assess.py detailed_report + cost)
# ------------------------------------------------------------------

def generate_markdown_report(items: dict[int, QuestionItem]) -> str:
    """Generate a markdown report matching the ``assess.py`` format, with cost info."""
    report = build_report_from_items(items)

    total_cost = sum(item.cost for item in items.values())

    lines: list[str] = []
    lines.append("# Auto Resolver Assessment Report\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Summary section
    lines.append("## Summary\n")
    lines.append(report.binary_results_table())
    lines.append("")

    # Totals
    total = (
        report.n_pp + report.n_pn + report.n_pc
        + report.n_np + report.n_nn + report.n_nc
        + report.n_cp + report.n_cn + report.n_cc
        + report.n_xp + report.n_xn + report.n_xc
        + report.n_nyr_p + report.n_nyr_n + report.n_nyr_c
        + report.n_nyr_nyr
    )
    correct = report.n_pp + report.n_nn + report.n_cc + report.n_nyr_nyr
    accuracy = (correct / total * 100) if total > 0 else 0

    lines.append(f"**Total Questions:** {total}")
    lines.append(f"**Correct Predictions:** {correct} ({accuracy:.1f}%)")
    lines.append(f"**Total Cost:** ${total_cost:.4f}")
    lines.append("")

    # Detailed results section
    lines.append("## Detailed Results\n")
    lines.append("")

    # Sort by category then question ID (same as assess.py)
    sorted_results = sorted(
        report.question_results.values(),
        key=lambda x: (x.outcome_category or "", x.question_id),
    )

    for result in sorted_results:
        lines.append(f"### Question {result.question_id}\n")
        lines.append(f"**Title:** {result.question_title}")
        lines.append(f"**URL:** {result.question_url}")
        lines.append("")

        # Question contents
        lines.append("**Question Contents:**\n")
        lines.append("> " + result.question_text.replace("\n", "\n> "))
        lines.append("")

        # Resolution comparison table
        lines.append("| Output Resolution | Correct Resolution |")
        lines.append("|-------------------|--------------------|")
        actual_str = ResolutionAssessmentReport._resolution_to_str(
            result.actual_resolution
        )
        predicted_str = ResolutionAssessmentReport._resolution_to_str(
            result.predicted_resolution
        )
        lines.append(f"| {predicted_str} | {actual_str} |")
        lines.append("")

        # Cost per question
        item = items.get(result.question_id)
        if item and item.cost > 0:
            lines.append(f"**Cost:** ${item.cost:.4f}")
            lines.append("")

        # Key evidence
        if result.key_evidence:
            lines.append("**Key Evidence:**")
            for evidence in result.key_evidence:
                lines.append(f"- {evidence}")
            lines.append("")

        lines.append("---\n")

    return "\n".join(lines)
