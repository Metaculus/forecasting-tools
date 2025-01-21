import os

import pandas as pd
import plotly.express as px
import streamlit as st
import typeguard

from forecasting_tools.forecasting.questions_and_reports.benchmark_for_bot import (
    BenchmarkForBot,
)
from forecasting_tools.forecasting.questions_and_reports.binary_report import (
    BinaryReport,
)
from forecasting_tools.util import file_manipulation
from front_end.helpers.report_displayer import ReportDisplayer


def get_json_files(directory: str) -> list[str]:
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json") and "bench" in file:
                full_path = os.path.join(root, file)
                json_files.append(full_path)
    return sorted(json_files)


def display_deviation_scores(reports: list[BinaryReport]) -> None:
    with st.expander("Deviation Scores", expanded=False):
        certain_reports = [
            r
            for r in reports
            if r.community_prediction is not None
            and (r.community_prediction > 0.9 or r.community_prediction < 0.1)
        ]
        uncertain_reports = [
            r
            for r in reports
            if r.community_prediction is not None
            and 0.1 <= r.community_prediction <= 0.9
        ]
        display_stats_for_report_type(reports, "All Questions")
        display_stats_for_report_type(
            certain_reports,
            "Certain Questions: Community Prediction >90% or <10%",
        )
        display_stats_for_report_type(
            uncertain_reports,
            "Uncertain Questions: Community Prediction 10%-90%",
        )


def display_stats_for_report_type(
    reports: list[BinaryReport], title: str
) -> None:
    average_expected_log_score = (
        BinaryReport.calculate_average_inverse_expected_log_score(reports)
    )
    average_deviation = BinaryReport.calculate_average_deviation_points(
        reports
    )
    st.markdown(
        f"""
        #### {title}
        - Number of Questions: {len(reports)}
        - Expected Log Score (lower is better): {average_expected_log_score:.4f}
        - Average Deviation: On average, there is a {average_deviation:.2%} percentage point difference between community and bot
        """
    )


def display_questions_and_forecasts(reports: list[BinaryReport]) -> None:
    with st.expander("Questions and Forecasts", expanded=False):
        st.subheader("Question List")
        certain_reports = [
            r
            for r in reports
            if r.community_prediction is not None
            and (r.community_prediction > 0.9 or r.community_prediction < 0.1)
        ]
        uncertain_reports = [
            r
            for r in reports
            if r.community_prediction is not None
            and 0.1 <= r.community_prediction <= 0.9
        ]

        display_question_stats_in_list(
            certain_reports,
            "Certain Questions (Community Prediction >90% or <10%)",
        )
        display_question_stats_in_list(
            uncertain_reports,
            "Uncertain Questions (Community Prediction 10%-90%)",
        )


def display_question_stats_in_list(
    report_list: list[BinaryReport], title: str
) -> None:
    st.subheader(title)
    sorted_reports = sorted(
        report_list,
        key=lambda r: (
            r.inversed_expected_log_score
            if r.inversed_expected_log_score is not None
            else -1
        ),
        reverse=True,
    )
    for report in sorted_reports:
        deviation = (
            report.inversed_expected_log_score
            if report.inversed_expected_log_score is not None
            else -1
        )
        st.write(
            f"- **Δ:** {deviation:.4f} | **🤖:** {report.prediction:.2%} | **👥:** {report.community_prediction:.2%} | **Question:** {report.question.question_text}"
        )


def display_benchmark_list(benchmarks: list[BenchmarkForBot]) -> None:
    if not benchmarks:
        return

    st.markdown("# Select Benchmark")
    benchmark_options = [
        f"{b.name} (Score: {b.average_inverse_expected_log_score:.4f})"
        for b in benchmarks
    ]
    selected_benchmark_name = st.selectbox(
        "Select a benchmark to view details:", benchmark_options
    )

    st.markdown("---")

    selected_idx = benchmark_options.index(selected_benchmark_name)
    benchmark = benchmarks[selected_idx]

    with st.expander(benchmark.name, expanded=False):
        st.markdown(f"**Description:** {benchmark.description}")
        st.markdown(
            f"**Time Taken (minutes):** {benchmark.time_taken_in_minutes}"
        )
        st.markdown(f"**Total Cost:** {benchmark.total_cost}")
        st.markdown(f"**Git Commit Hash:** {benchmark.git_commit_hash}")
        st.markdown(
            f"**Average Inverse Expected Log Score:** {benchmark.average_inverse_expected_log_score:.4f}"
        )
        # Add average deviation score if reports are binary
        if isinstance(benchmark.forecast_reports[0], BinaryReport):
            reports = typeguard.check_type(
                benchmark.forecast_reports, list[BinaryReport]
            )
            average_deviation = (
                BinaryReport.calculate_average_deviation_points(reports)
            )
            st.markdown(
                f"**Average Deviation Score:** {average_deviation:.2%} percentage points"
            )

    with st.expander("Bot Configuration", expanded=False):
        st.markdown("### Bot Configuration")
        for key, value in benchmark.forecast_bot_config.items():
            st.markdown(f"**{key}:** {value}")

    if benchmark.code:
        with st.expander("Forecast Bot Code", expanded=False):
            st.code(benchmark.code, language="python")

    # Display deviation scores and questions for this benchmark
    reports = benchmark.forecast_reports
    if isinstance(reports[0], BinaryReport):
        reports = typeguard.check_type(reports, list[BinaryReport])
        display_deviation_scores(reports)
        display_questions_and_forecasts(reports)
        ReportDisplayer.display_report_list(reports)


def display_benchmark_comparison_graphs(
    benchmarks: list[BenchmarkForBot],
) -> None:
    st.subheader("Benchmark Score Comparisons")
    st.markdown("Lower score is better for both metrics.")

    # Prepare data for all categories
    data_by_benchmark = []

    for benchmark in benchmarks:
        reports = benchmark.forecast_reports
        reports = typeguard.check_type(reports, list[BinaryReport])
        certain_reports = [
            r
            for r in reports
            if r.community_prediction is not None
            and (r.community_prediction > 0.9 or r.community_prediction < 0.1)
        ]
        uncertain_reports = [
            r
            for r in reports
            if r.community_prediction is not None
            and 0.1 <= r.community_prediction <= 0.9
        ]

        # Calculate all scores
        data_by_benchmark.extend(
            [
                {
                    "Benchmark": benchmark.name,
                    "Category": "All Questions",
                    "Expected Log Score": benchmark.average_inverse_expected_log_score,
                    "Deviation Score": BinaryReport.calculate_average_deviation_points(
                        reports
                    )
                    * 100,
                },
                {
                    "Benchmark": benchmark.name,
                    "Category": "Certain Questions",
                    "Expected Log Score": BinaryReport.calculate_average_inverse_expected_log_score(
                        certain_reports
                    ),
                    "Deviation Score": BinaryReport.calculate_average_deviation_points(
                        certain_reports
                    )
                    * 100,
                },
                {
                    "Benchmark": benchmark.name,
                    "Category": "Uncertain Questions",
                    "Expected Log Score": BinaryReport.calculate_average_inverse_expected_log_score(
                        uncertain_reports
                    ),
                    "Deviation Score": BinaryReport.calculate_average_deviation_points(
                        uncertain_reports
                    )
                    * 100,
                },
            ]
        )

    if not data_by_benchmark:
        return

    try:

        df = pd.DataFrame(data_by_benchmark)

        st.markdown("### Expected Log Scores")
        st.markdown("Lower score indicates better performance.")
        fig = px.bar(
            df,
            x="Benchmark",
            y="Expected Log Score",
            color="Category",
            barmode="group",
            title="Expected Log Scores by Benchmark and Category",
        )
        fig.update_layout(yaxis_title="Expected Log Score")
        st.plotly_chart(fig)

        st.markdown("### Deviation Scores")
        st.markdown(
            "Lower score indicates predictions closer to community consensus. Shown as difference in percentage points between bot and community."
        )
        fig = px.bar(
            df,
            x="Benchmark",
            y="Deviation Score",
            color="Category",
            barmode="group",
            title="Deviation Scores by Benchmark and Category",
        )
        fig.update_layout(yaxis_title="Deviation Score (percentage points)")
        st.plotly_chart(fig)

    except ImportError:
        st.error(
            "Please install plotly and pandas to view the graphs: `pip install plotly pandas`"
        )


def main() -> None:
    st.title("Benchmark Viewer")
    st.write("Select a JSON file containing BenchmarkForBot objects.")

    project_directory = file_manipulation.get_absolute_path("")
    json_files = get_json_files(project_directory)

    if not json_files:
        st.warning(f"No JSON files found in {project_directory}")
        return

    selected_file = st.selectbox(
        "Select a benchmark file:",
        json_files,
        format_func=lambda x: os.path.basename(x),
    )

    if selected_file:
        try:
            benchmarks: list[BenchmarkForBot] = (
                BenchmarkForBot.load_json_from_file_path(selected_file)
            )
            display_benchmark_comparison_graphs(benchmarks)
            display_benchmark_list(benchmarks)
        except Exception as e:
            st.error(f"Could not load file. Error: {str(e)}")


if __name__ == "__main__":
    main()
