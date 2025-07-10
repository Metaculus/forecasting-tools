import logging
import os
from typing import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import typeguard

from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.data_organizer import ReportTypes
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.front_end.helpers.report_displayer import ReportDisplayer
from forecasting_tools.util import file_manipulation
from forecasting_tools.util.stats import ConfidenceIntervalCalculator

logger = logging.getLogger(__name__)

PERFECT_PREDICTOR_NAME = "Perfect Predictor"


def get_json_files(directory: str) -> list[str]:
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if (
                file.endswith(".json") or file.endswith(".jsonl")
            ) and "bench" in file.lower():
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


def display_stats_for_report_type(reports: list[BinaryReport], title: str) -> None:
    if len(reports) == 0:
        logger.warning(f"No reports found for {title}")
        return
    average_expected_baseline_score = (
        BinaryReport.calculate_average_expected_baseline_score(reports)
    )
    average_deviation = BinaryReport.calculate_average_deviation_points(reports)
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
            "- **💻 :** Bot Prediction\n"
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


def display_question_stats_in_list(report_list: list[BinaryReport], title: str) -> None:
    st.subheader(title)
    sorted_reports = sorted(
        report_list,
        key=lambda r: (
            r.expected_baseline_score if r.expected_baseline_score is not None else -1
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
            question_text = (
                f"[{report.question.question_text}]({report.question.page_url})"
            )
        else:
            question_text = report.question.question_text
        st.write(
            f"- **🎯:** {score:.4f} | **💻:** {report.prediction:.2%} | "
            f"**👥:** {report.community_prediction:.2%} | **Question:** {question_text}"
        )


def display_benchmark_list(benchmarks: list[BenchmarkForBot]) -> None:
    if not benchmarks:
        return

    st.markdown("# Select Benchmark")
    benchmark_options = [
        f"{b.name} (Score: {b.average_expected_baseline_score:.4f})" for b in benchmarks
    ]
    selected_benchmark_name = st.selectbox(
        "Select a benchmark to view details:", benchmark_options
    )

    st.markdown("---")

    selected_idx = benchmark_options.index(selected_benchmark_name)
    benchmark = benchmarks[selected_idx]

    with st.expander(benchmark.name, expanded=False):
        st.markdown(f"**Description:** {benchmark.description}")
        st.markdown(f"**Number of Input Questions:** {benchmark.num_input_questions}")
        st.markdown(
            f"**Number of Successful forecasts:** {len(benchmark.forecast_reports)}"
        )
        failed_forecast_reports = (
            benchmark.num_failed_forecasts
            if benchmark.num_failed_forecasts is not None
            else "N/A"
        )
        st.markdown(f"**Number of Failed forecasts:** {failed_forecast_reports}")
        st.markdown(f"**Time Taken (minutes):** {benchmark.time_taken_in_minutes}")
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
            average_deviation = BinaryReport.calculate_average_deviation_points(reports)
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

    if benchmark.failed_report_errors:
        with st.expander("Failed Forecast Errors", expanded=False):
            st.markdown("### Failed Forecast Errors")
            for error in benchmark.failed_report_errors:
                st.markdown(f"**Error:** {error}")

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


def display_overview_stats(input_benchmarks: list[BenchmarkForBot]) -> None:
    benchmarks = [
        benchmark
        for benchmark in input_benchmarks
        if benchmark.explicit_name != PERFECT_PREDICTOR_NAME
    ]
    total_cost = sum(
        benchmark.total_cost
        for benchmark in benchmarks
        if benchmark.total_cost is not None
    )
    st.markdown("### Overview")
    st.markdown(f"**Total Cost:** ${total_cost:.2f}")
    st.markdown(
        f"**Number of input questions (set of all observed inputs values for all benchmarks):** {set(benchmark.num_input_questions for benchmark in benchmarks if benchmark.num_input_questions is not None)}"
    )
    st.markdown(
        f"**Number of forecast reports for benchmarks (set of all observed number of forecast reports for all benchmarks):** {set(len(benchmark.forecast_reports) for benchmark in benchmarks if benchmark.forecast_reports is not None)}"
    )
    st.markdown(
        f"**Total number of failed forecasts:** {sum(benchmark.num_failed_forecasts for benchmark in benchmarks if benchmark.num_failed_forecasts is not None)}"
    )
    st.markdown(
        f"**Average time per benchmark:** {sum(benchmark.time_taken_in_minutes for benchmark in benchmarks if benchmark.time_taken_in_minutes is not None) / len(benchmarks):.2f} minutes"
    )
    st.markdown(
        f"**Total time taken:** {sum(benchmark.time_taken_in_minutes for benchmark in benchmarks if benchmark.time_taken_in_minutes is not None):.2f} minutes"
    )
    st.markdown(
        f"**First benchmark's date:** {benchmarks[0].timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
    )


def display_benchmark_comparison_graphs(
    benchmarks: list[BenchmarkForBot],
) -> None:
    st.markdown("### To Note")
    st.markdown(
        """
        Note that there are seasonal changes with the certainty of questions (e.g. there are more certain questions near the end of the year).
        'Certain' questions score better, so be careful of comparing benchmarks from different time periods.

        - Uncertain Question: Questions with community prediction between 10% and 90%
        - Certain Question: Questions with community prediction greater than 90% or less than 10%
        - Perfect predictor: automatically created and shows what a perfect score (predicting community prediction) would be. To calculate this, it uses same questions as the first benchmark in the list.
        """
    )

    confidence_level = st.sidebar.slider(
        "Error Bar Confidence Level",
        min_value=0.0,
        max_value=1.0,
        value=0.90,
        step=0.05,
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
        certain_reports = typeguard.check_type(certain_reports, Sequence[BinaryReport])
        uncertain_reports = typeguard.check_type(
            uncertain_reports, Sequence[BinaryReport]
        )

        if len(certain_reports) > 0:
            average_certain_expected_baseline_score = (
                BinaryReport.calculate_average_expected_baseline_score(certain_reports)
            )
            average_certain_deviation_score = (
                BinaryReport.calculate_average_deviation_points(certain_reports) * 100
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
                BinaryReport.calculate_average_deviation_points(uncertain_reports) * 100
            )
        else:
            average_uncertain_expected_baseline_score = None
            average_uncertain_deviation_score = None

        benchmark_name = f"{index}: {benchmark.name}"

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
        "Higher score indicates better performance. Read more about baseline score "
        "[here](https://www.metaculus.com/help/scores-faq/#:~:text=The%20Baseline%20score%20compares,probability%20to%20all%20outcomes.). "
        "Expected baseline score is equal to `100 * (c * (np.log2(p) + 1.0) + (1.0 - c) * (np.log2(1.0 - p) + 1.0))` "
        "where c is the community prediction and p is your prediction. "
        "This is the expected value of the baseline score and is a proper score assuming the community prediction is the true probability. "
        f"Error bars are for a {confidence_level*100}% confidence interval. If an error bar is 0, then either:\n"
        "- You have < 4 forecasts\n"
        "- The data violated the normality assumption for a T-based confidence interval when num_forecasts < 30.\n\n"
    )

    # Mark highest scores with stars (higher is better for baseline score)
    max_scores = df.groupby("Category")["Expected Baseline Score"].transform("max")
    df["Is Best Expected"] = df["Expected Baseline Score"] == max_scores
    font_size = 11

    fig = px.bar(
        df,
        x="Benchmark",
        y="Expected Baseline Score",
        color="Category",
        barmode="group",
        title="Expected Baseline Scores by Benchmark and Category",
        error_y="Baseline Error",
    )
    fig.update_layout(
        yaxis_title="Expected Baseline Score",
        xaxis=dict(tickfont=dict(size=font_size)),
        yaxis=dict(tickfont=dict(size=font_size), dtick=10),
    )

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
    fig.update_layout(
        yaxis_title="Deviation Score",
        xaxis=dict(tickfont=dict(size=font_size)),
        yaxis=dict(tickfont=dict(size=font_size)),
    )

    add_star_annotations(fig, df, "Benchmark", "Deviation Score", "Is Best Deviation")

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
    reports_without_community_predictions: list[BinaryReport] = []
    for report in reports_of_perfect_benchmark:
        if report.community_prediction is None:
            reports_without_community_predictions.append(report)
        else:
            report.prediction = report.community_prediction
    if reports_without_community_predictions:
        urls = [
            report.question.page_url for report in reports_without_community_predictions
        ]
        raise ValueError(
            f"Found {len(reports_without_community_predictions)} reports without community predictions: {urls}"
        )
    perfect_benchmark.forecast_reports = reports_of_perfect_benchmark  # type: ignore
    perfect_benchmark.explicit_name = PERFECT_PREDICTOR_NAME
    return perfect_benchmark


def display_errors_of_benchmarks(benchmarks: list[BenchmarkForBot]) -> None:
    benchmarks_with_zero_forecasts = [
        benchmark for benchmark in benchmarks if len(benchmark.forecast_reports) == 0
    ]
    benchmarks_with_errors = [
        benchmark for benchmark in benchmarks if benchmark.failed_report_errors
    ]

    if not benchmarks_with_zero_forecasts and not benchmarks_with_errors:
        return

    with st.expander("Empty/Errored Benchmarks", expanded=False):
        if benchmarks_with_zero_forecasts:
            st.write("These are benchmark files that have 0 successful forecasts")
        for benchmark in benchmarks_with_zero_forecasts:
            st.write(f"- {benchmark.name}")

        if benchmarks_with_errors:
            st.write("These are benchmark files that have errors")
            st.write(
                f"Total errors: {sum(len(benchmark.failed_report_errors) for benchmark in benchmarks_with_errors)}"
            )
        for benchmark in benchmarks_with_errors:
            st.write(
                f"#### {len(benchmark.failed_report_errors)} Error for {benchmark.name}"
            )
            for error in benchmark.failed_report_errors:
                st.write(f"- **Error:** {error}")


def display_metaculus_research_reports(benchmarks: list[BenchmarkForBot]) -> None:
    metaculus_reports: list[tuple[str, ReportTypes]] = []

    for benchmark in benchmarks:
        for report in benchmark.forecast_reports:
            try:
                research_content = report.research.lower()
                if "metaculus" in research_content:
                    metaculus_reports.append((benchmark.name, report))
            except Exception as e:
                logger.warning(f"Error accessing research content for report: {e}")

    if not metaculus_reports:
        return

    with st.expander("Reports with Metaculus in Research", expanded=False):
        st.write(
            f"Found {len(metaculus_reports)} reports containing 'Metaculus' in their research section:"
        )
        for benchmark_name, report in metaculus_reports:
            st.write(f"# {benchmark_name}")
            question_text = (
                f"[{report.question.question_text}]({report.question.page_url})"
            )
            st.write(f"**Question:** {question_text}")
            st.write("**Research Excerpt:**")
            st.write(report.research.strip("#"))
            st.write("---")


def display_main_page(benchmarks: list[BenchmarkForBot]) -> None:
    st.markdown("# Benchmark Score Comparisons")
    display_overview_stats(benchmarks)
    display_benchmark_comparison_graphs(benchmarks)
    display_benchmark_list(benchmarks)


def load_benchmarks_from_files(
    file_paths: list[str],
) -> tuple[list[BenchmarkForBot], list[BenchmarkForBot]]:
    all_successful_benchmarks: list[BenchmarkForBot] = []
    all_benchmarks: list[BenchmarkForBot] = []
    for file in file_paths:
        benchmarks = BenchmarkForBot.load_json_from_file_path(file)
        for benchmark in benchmarks:
            all_benchmarks.append(benchmark)
            if len(benchmark.forecast_reports) > 0:
                all_successful_benchmarks.append(benchmark)
    return all_successful_benchmarks, all_benchmarks


load_benchmarks_from_files = st.cache_data(load_benchmarks_from_files)


def run_benchmark_streamlit_page(
    input: list[BenchmarkForBot] | str | None = None,
) -> None:
    """
    Run this function as a streamlit app. `streamlit run file_running_this_function.py`

    This function runs the benchmark streamlit page.
    If input_benchmarks is provided, it will display the benchmarks passed in.
    If input is a string, it will be treated as a folder path that contains benchmark JSON files.
    Otherwise, it will display the benchmarks in the project directory.

    Files containing "bench" in the name and ending in '.json' will be collected as benchmark files
    from the project directory tree.
    """
    # TODO: Refactor this file. It doesn't match the cleanliness standards of the rest of the code base.

    st.title("Benchmark Viewer")

    if isinstance(input, list):
        display_main_page(input)
        return
    elif isinstance(input, str):
        project_directory = input
    else:
        project_directory = file_manipulation.normalize_package_path("")

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
            all_successful_benchmarks, all_benchmarks = load_benchmarks_from_files(
                selected_files
            )

            logger.info(f"Loaded {len(all_successful_benchmarks)} benchmarks")

            perfect_benchmark = make_perfect_benchmark(all_successful_benchmarks[0])
            all_successful_benchmarks.insert(0, perfect_benchmark)

            benchmark_options = []
            for i, b in enumerate(all_successful_benchmarks):
                benchmark_options.append(
                    f"{i}: {b.name} (Score: {b.average_expected_baseline_score:.4f})"
                )

            selected_benchmarks = st.multiselect(
                "Select benchmarks to display:",
                range(len(all_successful_benchmarks)),
                default=range(len(all_successful_benchmarks)),
                format_func=lambda i: benchmark_options[i],
            )

            display_errors_of_benchmarks(all_benchmarks)
            display_metaculus_research_reports(all_benchmarks)

            if selected_benchmarks:
                filtered_benchmarks = [
                    all_successful_benchmarks[i] for i in selected_benchmarks
                ]
                display_main_page(filtered_benchmarks)
        except Exception as e:
            st.error(
                f"Error when loading/displaying benchmarks: {e.__class__.__name__}: {str(e)}"
            )


if __name__ == "__main__":
    run_benchmark_streamlit_page()
