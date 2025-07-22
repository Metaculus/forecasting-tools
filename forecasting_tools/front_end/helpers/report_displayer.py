from __future__ import annotations

import logging
import re

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.markdown_tree import MarkdownTree
from forecasting_tools.data_models.questions import BinaryQuestion

logger = logging.getLogger(__name__)


class ReportDisplayer:
    DOWNLOAD_REPORT_BUTTON_KEY = "download_report_button"
    REPORT_SELECTBOX_KEY = "select_report_selectbox"

    @classmethod
    def display_report_list(cls, reports_to_display: list[BinaryReport]) -> None:
        if len(reports_to_display) == 0:
            return

        sorted_reports = ReportDisplayer._make_new_list_of_sorted_reports(
            reports_to_display
        )
        if "selected_report_index" not in st.session_state:
            st.session_state.selected_report_index = None

        report_options: list[str] = []
        for report in sorted_reports:
            option = ""
            option += f"🤖: {report.prediction:.2%}"
            if report.community_prediction is not None:
                option += f" | 👥: {report.community_prediction:.2%}"
            option += (
                f" | {report.question.question_text[:60]}..."
                if len(report.question.question_text) > 60
                else f"| {report.question.question_text}"
            )
            report_options.append(option)

        selected_option_index = st.selectbox(
            "Select a report to view:",
            range(len(sorted_reports)),
            format_func=lambda i: report_options[i],
            key=ReportDisplayer.REPORT_SELECTBOX_KEY,
        )

        st.session_state.selected_report_index = selected_option_index

        selected_index = st.session_state.selected_report_index
        if selected_index is not None and selected_index < len(sorted_reports):
            selected_report = sorted_reports[selected_index]
            ReportDisplayer.display_report(selected_report)
        else:
            st.write("Select a report to view.")

    @staticmethod
    def _make_new_list_of_sorted_reports(
        reports: list[BinaryReport],
    ) -> list[BinaryReport]:
        copy_of_reports = reports.copy()
        sorted_reports = sorted(
            copy_of_reports,
            key=lambda r: (
                r.expected_baseline_score
                if r.expected_baseline_score is not None
                else float("inf")
            ),
            reverse=True,
        )
        return sorted_reports

    @classmethod
    def display_report(cls, report: BinaryReport) -> None:
        sections = report.report_sections
        if not sections:
            logger.warning("No sections found in report")
            return
        assert all(section.section_content is not None for section in sections)

        tab_names = [section.title or "Untitled" for section in sections]
        show_question_details = (
            report.question.id_of_post is not None and report.question.id_of_post > 0
        )
        if show_question_details:
            tab_names.append("Question Details")
        tabs = st.tabs(tab_names)
        normal_tabs = tabs[:-1] if show_question_details else tabs
        for tab, section in zip(normal_tabs, sections):
            cls.__display_normal_tab(tab, section)
        if show_question_details:
            cls.__display_question_detail_tab(tabs[-1], report.question)

        st.divider()
        st.download_button(
            label="Download Report",
            data=report.explanation,
            file_name="forecast_report.md",
            mime="text/markdown",
            key=cls.DOWNLOAD_REPORT_BUTTON_KEY,
        )

    @classmethod
    def __display_normal_tab(cls, tab: DeltaGenerator, section: MarkdownTree) -> None:
        with tab:
            st.markdown(cls.clean_markdown(section.section_content))
            for sub_section in section.sub_sections:
                with st.expander(sub_section.title or "Untitled"):
                    st.markdown(cls.clean_markdown(sub_section.section_content))
                    cls.__display_nested_sections(sub_section.sub_sections)

    @classmethod
    def __display_question_detail_tab(
        cls, tab: DeltaGenerator, question: BinaryQuestion
    ) -> None:
        with tab:
            st.write(f"**Question Text:** {cls.clean_markdown(question.question_text)}")
            st.write(
                f"**Resolution Criteria:** {cls.clean_markdown(question.resolution_criteria or 'None')}"
            )
            st.write(
                f"**Fine Print:** {cls.clean_markdown(question.fine_print or 'None')}"
            )
            st.write(
                f"**Background Info:** {cls.clean_markdown(question.background_info or 'None')}"
            )
            st.write(f"**Question Type:** {type(question)}")
            st.write(f"**Page URL:** {question.page_url}")
            st.write(f"**Number of Forecasters:** {question.num_forecasters}")
            st.write(f"**Number of Predictions:** {question.num_predictions}")
            community_prediction_formatted = (
                f"{question.community_prediction_at_access_time:.2%}"
                if question.community_prediction_at_access_time is not None
                else "N/A"
            )
            st.write(f"**Community Prediction:** {community_prediction_formatted}")
            st.write(f"**Date Accessed:** {question.date_accessed}")
            st.write(f"**State at Access:** {question.state.value}")
            if question.close_time:
                st.write(f"**Close Time:** {question.close_time}")
            if question.actual_resolution_time:
                st.write(
                    f"**Actual Resolution Time:** {question.actual_resolution_time}"
                )
            if question.scheduled_resolution_time:
                st.write(
                    f"**Scheduled Resolution Time:** {question.scheduled_resolution_time}"
                )
            if question.id_of_post and question.id_of_post > 0:
                st.write(f"**Question ID:** {question.id_of_post}")

    @classmethod
    def __display_nested_sections(
        cls, sections: list[MarkdownTree], level: int = 3
    ) -> None:
        for section in sections:
            st.markdown(cls.clean_markdown(section.section_content))
            ReportDisplayer.__display_nested_sections(section.sub_sections, level + 1)

    @staticmethod
    def clean_markdown(text: str) -> str:
        def replace_dollar(match: re.Match) -> str:
            backslashes = match.group(1) or ""
            if len(backslashes) % 2 == 1:  # Odd number of backslashes means escaped
                return match.group(0)
            else:  # Even number of backslashes means not escaped
                return backslashes + r"\$"

        # Regex to match any number of backslashes followed by $
        pattern = r"(\\*)\$"
        cleaned_text = re.sub(pattern, replace_dollar, text)
        return cleaned_text

    @staticmethod
    def markdown_is_clean(text: str) -> bool:
        return text == ReportDisplayer.clean_markdown(text)
