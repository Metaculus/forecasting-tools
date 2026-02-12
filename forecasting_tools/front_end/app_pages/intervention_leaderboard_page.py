import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.graph_objs as go
import streamlit as st

from forecasting_tools.agents_and_tools.situation_simulator.intervention_testing.data_models import (
    ForecastCategory,
    InterventionForecast,
    InterventionRun,
)
from forecasting_tools.agents_and_tools.situation_simulator.intervention_testing.intervention_storage import (
    list_run_folders,
    load_all_intervention_runs,
)
from forecasting_tools.front_end.helpers.app_page import AppPage

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_BASE = "temp/intervention_benchmarks/"


class InterventionLeaderboardPage(AppPage):
    PAGE_DISPLAY_NAME: str = "🎯 Intervention Leaderboard"
    URL_PATH: str = "/intervention-leaderboard"
    ENABLE_HEADER: bool = False
    ENABLE_FOOTER: bool = False

    @classmethod
    async def _async_main(cls) -> None:
        run_intervention_leaderboard_page(DEFAULT_RESULTS_BASE)


def run_intervention_leaderboard_page(results_base: str) -> None:
    st.title("Intervention Forecast Leaderboard")
    st.markdown(
        "Compare model performance on conditional intervention forecasts "
        "from simulation scenarios."
    )

    results_dir = _display_folder_picker(results_base)

    all_runs = _load_runs_cached(results_dir)
    if not all_runs:
        st.warning(
            "No intervention benchmark results found in the selected folder. "
            "Run the benchmark script to generate results."
        )
        return

    category_filter = st.radio(
        "Forecast Category",
        ["All", "Hard Metric", "Qualitative"],
        horizontal=True,
    )
    selected_category = _map_category_filter(category_filter)

    _display_leaderboard_table(all_runs, selected_category)
    st.markdown("---")
    _display_calibration_chart(all_runs, selected_category)
    st.markdown("---")
    _display_hard_metric_details(all_runs)
    st.markdown("---")
    _display_run_details(all_runs)


def _display_folder_picker(results_base: str) -> str:
    with st.sidebar:
        st.subheader("Select Run Folder")

        available_folders = list_run_folders(results_base)
        folder_display_names = [Path(f).name for f in available_folders]

        load_all_label = f"All runs in {results_base}"
        options = [load_all_label] + folder_display_names

        selected = st.selectbox("Run folder", options)

        if selected == load_all_label:
            active_dir = results_base
        else:
            selected_idx = folder_display_names.index(selected)
            active_dir = available_folders[selected_idx]

        custom_path = st.text_input(
            "Or enter a custom folder path",
            placeholder="e.g. temp/intervention_benchmarks/run_2026-...",
        )
        if custom_path.strip():
            active_dir = custom_path.strip()

        st.caption(f"Loading from: `{active_dir}`")
    return active_dir


@st.cache_data(ttl=60)
def _load_runs_cached(results_dir: str) -> list[dict]:
    runs = load_all_intervention_runs(results_dir)
    return [r.to_json() for r in runs]


def _deserialize_runs(run_dicts: list[dict]) -> list[InterventionRun]:
    return [InterventionRun.from_json(d) for d in run_dicts]


def _map_category_filter(label: str) -> ForecastCategory | None:
    if label == "Hard Metric":
        return ForecastCategory.HARD_METRIC
    elif label == "Qualitative":
        return ForecastCategory.QUALITATIVE
    return None


def _filter_forecasts(
    forecasts: list[InterventionForecast],
    category: ForecastCategory | None,
) -> list[InterventionForecast]:
    if category is None:
        return forecasts
    return [f for f in forecasts if f.category == category]


def _display_leaderboard_table(
    run_dicts: list[dict],
    category: ForecastCategory | None,
) -> None:
    st.subheader("Model Leaderboard")
    runs = _deserialize_runs(run_dicts)

    model_stats = _aggregate_model_stats(runs, category)
    rows = _build_leaderboard_rows(model_stats)

    rows.sort(
        key=lambda r: (
            float(r["Avg Brier Score"]) if r["Avg Brier Score"] != "N/A" else 999
        )
    )

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No data available for the selected category.")

    st.caption("Lower Brier Score is better (0 = perfect, 1 = worst possible).")


def _aggregate_model_stats(
    runs: list[InterventionRun],
    category: ForecastCategory | None,
) -> dict[str, dict]:
    model_stats: dict[str, dict] = {}
    for run in runs:
        if run.model_name not in model_stats:
            model_stats[run.model_name] = {
                "runs": 0,
                "total_forecasts": 0,
                "resolved_forecasts": 0,
                "total_brier": 0.0,
                "total_cost": 0.0,
            }
        stats = model_stats[run.model_name]
        stats["runs"] += 1
        stats["total_cost"] += run.total_cost

        for f in _filter_forecasts(run.forecasts, category):
            stats["total_forecasts"] += 1
            if f.brier_score is not None:
                stats["resolved_forecasts"] += 1
                stats["total_brier"] += f.brier_score
    return model_stats


def _build_leaderboard_rows(model_stats: dict[str, dict]) -> list[dict]:
    rows = []
    for model_name, stats in model_stats.items():
        avg_brier = (
            stats["total_brier"] / stats["resolved_forecasts"]
            if stats["resolved_forecasts"] > 0
            else None
        )
        rows.append(
            {
                "Model": model_name,
                "Runs": stats["runs"],
                "Forecasts": stats["total_forecasts"],
                "Resolved": stats["resolved_forecasts"],
                "Avg Brier Score": (
                    f"{avg_brier:.4f}" if avg_brier is not None else "N/A"
                ),
                "Total Cost ($)": f"{stats['total_cost']:.2f}",
            }
        )
    return rows


def _display_calibration_chart(
    run_dicts: list[dict],
    category: ForecastCategory | None,
) -> None:
    st.subheader("Calibration Chart")
    runs = _deserialize_runs(run_dicts)

    model_forecasts = _collect_resolved_forecasts_by_model(runs, category)

    if not any(model_forecasts.values()):
        st.info("No resolved forecasts available for calibration chart.")
        return

    fig = _build_calibration_figure(model_forecasts)
    st.plotly_chart(fig, use_container_width=True)


def _collect_resolved_forecasts_by_model(
    runs: list[InterventionRun],
    category: ForecastCategory | None,
) -> dict[str, list[InterventionForecast]]:
    model_forecasts: dict[str, list[InterventionForecast]] = defaultdict(list)
    for run in runs:
        filtered = _filter_forecasts(run.forecasts, category)
        resolved = [f for f in filtered if f.resolved and f.resolution is not None]
        model_forecasts[run.model_name].extend(resolved)
    return model_forecasts


def _compute_calibration_bins(
    forecasts: list[InterventionForecast],
) -> tuple[list[float], list[float]]:
    bin_edges = [i / 10 for i in range(11)]
    bin_predictions: list[float] = []
    bin_resolutions: list[float] = []
    num_bins = len(bin_edges) - 1

    for i in range(num_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        is_last_bin = i == num_bins - 1
        in_bin = [
            f
            for f in forecasts
            if low <= f.prediction < high or (is_last_bin and f.prediction == high)
        ]
        if in_bin:
            avg_pred = sum(f.prediction for f in in_bin) / len(in_bin)
            avg_res = sum(1.0 if f.resolution else 0.0 for f in in_bin) / len(in_bin)
            bin_predictions.append(avg_pred)
            bin_resolutions.append(avg_res)

    return bin_predictions, bin_resolutions


def _build_calibration_figure(
    model_forecasts: dict[str, list[InterventionForecast]],
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line={"dash": "dash", "color": "gray"},
        )
    )
    for model_name, forecasts in model_forecasts.items():
        if not forecasts:
            continue
        bin_preds, bin_res = _compute_calibration_bins(forecasts)
        if bin_preds:
            fig.add_trace(
                go.Scatter(
                    x=bin_preds,
                    y=bin_res,
                    mode="lines+markers",
                    name=model_name,
                )
            )
    fig.update_layout(
        title="Calibration: Predicted Probability vs Actual Resolution Rate",
        xaxis_title="Predicted Probability",
        yaxis_title="Actual Resolution Rate",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    )
    return fig


def _display_hard_metric_details(run_dicts: list[dict]) -> None:
    st.subheader("Hard Metric Forecast Details")
    st.markdown(
        "Inventory-based forecasts with auto-resolved outcomes. "
        "These track specific item quantities at simulation end."
    )
    runs = _deserialize_runs(run_dicts)

    rows = []
    for run in runs:
        for f in run.hard_metric_forecasts:
            rows.append(_build_hard_metric_row(run, f))

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No hard metric forecasts available.")


def _build_hard_metric_row(
    run: InterventionRun,
    forecast: InterventionForecast,
) -> dict:
    resolved_label = _format_resolution_label(forecast.resolution)
    criteria_label = _format_criteria_label(forecast)
    brier_label = (
        f"{forecast.brier_score:.4f}" if forecast.brier_score is not None else "N/A"
    )

    return {
        "Model": run.model_name,
        "Scenario": run.situation_name,
        "Type": "Conditional" if forecast.is_conditional else "Baseline",
        "Question": forecast.question_title,
        "Prediction": f"{forecast.prediction:.1%}",
        "Resolved": resolved_label,
        "Brier": brier_label,
        "Criteria": criteria_label,
    }


def _format_resolution_label(resolution: bool | None) -> str:
    if resolution is None:
        return "N/A"
    return "Yes" if resolution else "No"


def _format_criteria_label(forecast: InterventionForecast) -> str:
    if not forecast.hard_metric_criteria:
        return "N/A"
    c = forecast.hard_metric_criteria
    return f"{c.agent_name}.{c.item_name} {c.operator} {c.threshold}"


def _display_run_details(run_dicts: list[dict]) -> None:
    st.subheader("Individual Run Details")
    runs = _deserialize_runs(run_dicts)

    if not runs:
        st.info("No runs to display.")
        return

    run_labels = [
        f"{run.model_name} | {run.situation_name} | {run.target_agent_name} "
        f"({run.run_id})"
        for run in runs
    ]
    selected_label = st.selectbox("Select a run", run_labels)
    if selected_label is None:
        return

    selected_idx = run_labels.index(selected_label)
    run = runs[selected_idx]

    with st.expander("Run Metadata", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model", run.model_name.split("/")[-1])
        col2.metric("Scenario", run.situation_name)
        col3.metric("Target Agent", run.target_agent_name)
        col4.metric("Cost", f"${run.total_cost:.2f}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Warmup Steps", run.warmup_steps)
        col6.metric("Total Steps", run.total_steps)
        avg_brier = run.average_brier_score
        col7.metric(
            "Avg Brier",
            f"{avg_brier:.4f}" if avg_brier is not None else "N/A",
        )
        col8.metric("Forecasts", len(run.forecasts))

    with st.expander("Evaluation Criteria"):
        for i, criterion in enumerate(run.evaluation_criteria, 1):
            st.markdown(f"{i}. {criterion}")

    with st.expander("Intervention Description"):
        st.markdown(run.intervention_description)

    with st.expander("Policy Proposal"):
        st.markdown(run.policy_proposal_markdown)

    with st.expander("All Forecasts"):
        baseline_forecasts = run.baseline_forecasts
        conditional_forecasts = run.conditional_forecasts

        st.markdown("### Baseline (Status Quo) Forecasts")
        _display_forecast_table(baseline_forecasts)

        st.markdown("### Conditional (With Intervention) Forecasts")
        _display_forecast_table(conditional_forecasts)


def _display_forecast_table(forecasts: list[InterventionForecast]) -> None:
    if not forecasts:
        st.info("No forecasts in this category.")
        return

    rows = []
    for f in forecasts:
        resolution_label = _format_resolution_label(f.resolution)
        if not f.resolved:
            resolution_label = "Pending"
        brier_label = f"{f.brier_score:.4f}" if f.brier_score is not None else "N/A"
        reasoning_text = (
            f.reasoning[:150] + "..." if len(f.reasoning) > 150 else f.reasoning
        )
        rows.append(
            {
                "Category": f.category.value,
                "Question": f.question_title,
                "Prediction": f"{f.prediction:.1%}",
                "Resolution": resolution_label,
                "Brier": brier_label,
                "Reasoning": reasoning_text,
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    InterventionLeaderboardPage.main()
