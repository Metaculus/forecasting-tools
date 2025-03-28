import logging
import os
from typing import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import typeguard

from forecasting_tools.data_models.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.front_end.helpers.report_displayer import (
    ReportDisplayer,
)
from forecasting_tools.util import file_manipulation
from forecasting_tools.util.stats import ConfidenceIntervalCalculator

logger = logging.getLogger(__name__)


def get_json_files(directory: str) -> list[str]:
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json") and "bench" in file.lower():
                full_path = os.path.join(root, file)
                json_files.append(full_path)
    return sorted(json_files)


def display_score_overview(reports: list[BinaryReport]) -> None:
    with st.expander("Scores by category", expanded=False):
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
    assert len(reports) > 0, "Must have at least one report to display stats"
    average_expected_baseline_score = (
        BinaryReport.calculate_average_expected_baseline_score(reports)
    )
    average_deviation = BinaryReport.calculate_average_deviation_points(
        reports
    )
    st.markdown(
        f"""
        #### {title}
        - Number of Questions: {len(reports)}
        - Expected Baseline Score (lower is better): {average_expected_baseline_score:.4f}
        - Average Deviation: On average, there is a difference of {average_deviation:.2%} percentage points between community and bot predictions
        """
    )


def display_questions_and_forecasts(reports: list[BinaryReport]) -> None:
    with st.expander("Scores and forecasts by question", expanded=False):
        st.subheader("Score list")
        st.write(
            "- **🎯:** Expected Baseline Score\n"
            "- **🤖:** Bot Prediction\n"
            "- **👥:** Community Prediction\n"
        )
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
            r.expected_baseline_score
            if r.expected_baseline_score is not None
            else -1
        ),
        reverse=True,
    )
    for report in sorted_reports:
        score = (
            report.expected_baseline_score
            if report.expected_baseline_score is not None
            else -1
        )
        if report.question.page_url:
            question_text = f"[{report.question.question_text}]({report.question.page_url})"
        else:
            question_text = report.question.question_text
        st.write(
            f"- **🎯:** {score:.4f} | **🤖:** {report.prediction:.2%} | "
            f"**👥:** {report.community_prediction:.2%} | **Question:** {question_text}"
        )


def display_benchmark_list(benchmarks: list[BenchmarkForBot]) -> None:
    if not benchmarks:
        return

    st.markdown("# Select Benchmark")
    benchmark_options = [
        f"{b.name} (Score: {b.average_expected_baseline_score:.4f})"
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
            f"**Average Expected Baseline Score:** {benchmark.average_expected_baseline_score:.4f}"
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
        display_score_overview(reports)
        display_questions_and_forecasts(reports)
        ReportDisplayer.display_report_list(reports)


def add_star_annotations(
    fig: go.Figure,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    is_starred_col: str,
) -> None:
    """Add star annotations to bars meeting the starred condition."""
    for idx, row in df[df[is_starred_col]].iterrows():
        fig.add_annotation(
            x=row[x_col],
            y=row[y_col],
            text="★",
            showarrow=False,
            yshift=10,
            font=dict(size=20),
        )


def calculate_expected_baseline_margin_of_error(
    reports: Sequence[ForecastReport | BinaryReport], confidence_level: float
) -> float | None:
    scores = [r.expected_baseline_score for r in reports]
    scores = typeguard.check_type(scores, list[float])
    try:
        margin_of_error = (
            ConfidenceIntervalCalculator.confidence_interval_from_observations(
                scores, confidence_level
            ).margin_of_error
        )
    except Exception as e:
        logger.error(f"Error calculating margin of error: {e}")
        return None
    return margin_of_error


def display_benchmark_comparison_graphs(
    benchmarks: list[BenchmarkForBot],
) -> None:
    st.markdown("# Benchmark Score Comparisons")
    st.markdown(
        """
        Note that there are seasonal changes with the certainty of questions (e.g. there are more certain questions near the end of the year).
        'Certain' questions score better, so be careful of comparing benchmarks from different time periods.

        - Uncertain Question: Questions with community prediction between 10% and 90%
        - Certain Question: Questions with community prediction greater than 90% or less than 10%
        - Perfect predictor: automatically created and shows what a perfect score (predicting community prediction) would be.
        """
    )
    data_by_benchmark = []

    for index, benchmark in enumerate(benchmarks):
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

        reports = typeguard.check_type(reports, Sequence[BinaryReport])
        certain_reports = typeguard.check_type(
            certain_reports, Sequence[BinaryReport]
        )
        uncertain_reports = typeguard.check_type(
            uncertain_reports, Sequence[BinaryReport]
        )

        if len(certain_reports) > 0:
            average_certain_expected_baseline_score = (
                BinaryReport.calculate_average_expected_baseline_score(
                    certain_reports
                )
            )
            average_certain_deviation_score = (
                BinaryReport.calculate_average_deviation_points(
                    certain_reports
                )
                * 100
            )
        else:
            average_certain_expected_baseline_score = None
            average_certain_deviation_score = None

        if len(uncertain_reports) > 0:
            average_uncertain_expected_baseline_score = (
                BinaryReport.calculate_average_expected_baseline_score(
                    uncertain_reports
                )
            )
            average_uncertain_deviation_score = (
                BinaryReport.calculate_average_deviation_points(
                    uncertain_reports
                )
                * 100
            )
        else:
            average_uncertain_expected_baseline_score = None
            average_uncertain_deviation_score = None

        benchmark_name = f"{index}: {benchmark.name}"
        confidence_level = 0.90

        data_by_benchmark.extend(
            [
                {
                    "Benchmark": benchmark_name,
                    "Category": "All Questions",
                    "Expected Baseline Score": benchmark.average_expected_baseline_score,
                    "Deviation Score": BinaryReport.calculate_average_deviation_points(
                        reports
                    )
                    * 100,
                    "Baseline Error": calculate_expected_baseline_margin_of_error(
                        reports, confidence_level
                    ),
                },
                {
                    "Benchmark": benchmark_name,
                    "Category": "Certain Questions",
                    "Expected Baseline Score": average_certain_expected_baseline_score,
                    "Deviation Score": average_certain_deviation_score,
                    "Baseline Error": calculate_expected_baseline_margin_of_error(
                        certain_reports, confidence_level
                    ),
                },
                {
                    "Benchmark": benchmark_name,
                    "Category": "Uncertain Questions",
                    "Expected Baseline Score": average_uncertain_expected_baseline_score,
                    "Deviation Score": average_uncertain_deviation_score,
                    "Baseline Error": calculate_expected_baseline_margin_of_error(
                        uncertain_reports, confidence_level
                    ),
                },
            ]
        )

    if not data_by_benchmark:
        return

    df = pd.DataFrame(data_by_benchmark)

    st.markdown("### Expected Baseline Scores")
    st.markdown(
        "Higher score indicates better performance. Read more [here](https://www.metaculus.com/help/scores-faq/#:~:text=The%20Baseline%20score%20compares,probability%20to%20all%20outcomes.). "
        "This is a proper score assuming the community prediction is the true probability.\n\n"
        f"Error bars are for a {confidence_level*100}% confidence interval. "
        "If an error bar is 0, then either:\n"
        "- You have < 4 forecasts\n"
        "- The data violated the normality assumption for a T-based confidence interval when num_forecasts < 30.\n\n"
    )

    # Mark highest scores with stars (higher is better for baseline score)
    max_scores = df.groupby("Category")["Expected Baseline Score"].transform(
        "max"
    )
    df["Is Best Expected"] = df["Expected Baseline Score"] == max_scores

    fig = px.bar(
        df,
        x="Benchmark",
        y="Expected Baseline Score",
        color="Category",
        barmode="group",
        title="Expected Baseline Scores by Benchmark and Category",
        error_y="Baseline Error",
    )
    fig.update_layout(yaxis_title="Expected Baseline Score")

    add_star_annotations(
        fig, df, "Benchmark", "Expected Baseline Score", "Is Best Expected"
    )

    st.plotly_chart(fig)

    st.markdown("### Deviation Scores")
    st.markdown(
        "Lower score indicates predictions closer to community consensus. "
        "Shown as difference in percentage points between bot and community. "
        "This is not a proper score and is less meaningful than the expected baseline score. However it is more intuitive."
    )

    # Mark lowest deviations with stars (lower is better for deviation score)
    min_deviations = df.groupby("Category")["Deviation Score"].transform("min")
    df["Is Best Deviation"] = df["Deviation Score"] == min_deviations

    fig = px.bar(
        df,
        x="Benchmark",
        y="Deviation Score",
        color="Category",
        barmode="group",
        title="Deviation Scores by Benchmark and Category",
    )
    fig.update_layout(yaxis_title="Deviation Score (percentage points)")

    add_star_annotations(
        fig, df, "Benchmark", "Deviation Score", "Is Best Deviation"
    )

    st.plotly_chart(fig)


def make_perfect_benchmark(
    model_benchmark: BenchmarkForBot,
) -> BenchmarkForBot:
    perfect_benchmark = model_benchmark.model_copy()
    reports_of_perfect_benchmark = [
        report.model_copy() for report in perfect_benchmark.forecast_reports
    ]
    reports_of_perfect_benchmark = typeguard.check_type(
        reports_of_perfect_benchmark, list[BinaryReport]
    )
    for report in reports_of_perfect_benchmark:
        assert report.community_prediction is not None
        report.prediction = report.community_prediction
    perfect_benchmark.forecast_reports = reports_of_perfect_benchmark
    perfect_benchmark.explicit_name = (
        "Perfect Predictor (uses same questions as benchmark 1)"
    )
    return perfect_benchmark


def run_benchmark_streamlit_page(
    input: list[BenchmarkForBot] | str | None = None,
) -> None:
    """
    Run this function as a streamlit app. `streamlit run file_running_this_function.py`

    This function runs the benchmark streamlit page.
    If input_benchmarks is provided, it will display the benchmarks passed in.
    If input is a string, it will be treated as a folder path that contains benchmark JSON file.
    Otherwise, it will display the benchmarks in the project directory.

    Files containing "bench" in the name and ending in '.json' will be collected as benchmark files
    from the project directory tree.
    """

    st.title("Benchmark Viewer")

    if isinstance(input, list):
        display_benchmark_comparison_graphs(input)
        display_benchmark_list(input)
        return
    elif isinstance(input, str):
        project_directory = input
    else:
        project_directory = file_manipulation.get_absolute_path("")

    st.write("Select JSON files containing BenchmarkForBot objects.")

    json_files = get_json_files(project_directory)

    if not json_files:
        st.warning(f"No JSON files found in {project_directory}")
        return

    selected_files = st.multiselect(
        "Select benchmark files:",
        json_files,
        format_func=lambda x: os.path.basename(x),
    )

    if selected_files:
        try:
            all_benchmarks: list[BenchmarkForBot] = []
            for file in selected_files:
                benchmarks = BenchmarkForBot.load_json_from_file_path(file)
                all_benchmarks.extend(benchmarks)

            perfect_benchmark = make_perfect_benchmark(all_benchmarks[0])
            all_benchmarks.insert(0, perfect_benchmark)

            benchmark_options = [
                f"{i}: {b.name} (Score: {b.average_expected_baseline_score:.4f})"
                for i, b in enumerate(all_benchmarks)
            ]
            selected_benchmarks = st.multiselect(
                "Select benchmarks to display:",
                range(len(all_benchmarks)),
                default=range(len(all_benchmarks)),
                format_func=lambda i: benchmark_options[i],
            )

            if selected_benchmarks:
                filtered_benchmarks = [
                    all_benchmarks[i] for i in selected_benchmarks
                ]
                display_benchmark_comparison_graphs(filtered_benchmarks)
                display_benchmark_list(filtered_benchmarks)
        except Exception as e:
            st.error(f"Could not load files. Error: {str(e)}")


if __name__ == "__main__":
    run_benchmark_streamlit_page()
