import asyncio

import streamlit as st

from forecasting_tools.agents_and_tools.decision_hub.data_models import (
    Forecast,
    ScenarioSet,
)
from forecasting_tools.agents_and_tools.decision_hub.forecast_agent import (
    run_forecasts_streamed,
)
from forecasting_tools.agents_and_tools.decision_hub.pages._shared import (
    get_db,
    get_session_info,
    load_forecast_questions,
    load_forecasts,
    load_research_reports,
    load_scenario_set,
    require_active_session,
    stream_agent_response,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.misc import clean_indents


async def main() -> None:
    st.title("5. Forecasts")
    if not require_active_session():
        return

    username, session_id, policy_question = get_session_info()
    db = get_db()

    st.info(f"**Policy Question:** {policy_question}")

    questions = load_forecast_questions(db, session_id)
    existing_forecasts = load_forecasts(db, session_id)

    if not questions:
        st.warning(
            "No forecast questions found. Go to **4. Forecast Questions** first."
        )
        return

    forecast_by_qid = {f.question_id: f for f in existing_forecasts}

    st.subheader(f"Questions & Forecasts ({len(questions)} questions)")
    for i, q in enumerate(questions):
        f = forecast_by_qid.get(q.question_id)
        status_icon = "predicted" if f else "pending"
        with st.expander(
            f"Q{i+1} [{status_icon}]: {q.question_text[:70]}...",
            expanded=not f,
        ):
            st.write(f"**Type:** {q.question_type}")
            if q.conditional_on_scenario:
                st.write(f"**Scenario:** {q.conditional_on_scenario}")
            if f:
                st.write(f"**Prediction:** {f.prediction}")
                st.write(f"**Reasoning:** {f.reasoning}")
            else:
                st.caption("No forecast yet.")

    st.markdown("---")
    st.subheader("Generate AI Forecasts")

    scenarios = load_scenario_set(db, session_id)
    research_reports = load_research_reports(db, session_id)
    research_context = "\n\n".join(r.report_markdown for r in research_reports)

    if st.button("Generate Forecasts", key="gen_forecasts_btn", type="primary"):
        with MonetaryCostManager():
            result = run_forecasts_streamed(
                questions,
                policy_question,
                research_context or "No prior research.",
                scenarios or ScenarioSet(session_id=session_id, username=username),
            )
            streamed_text = await stream_agent_response(result)

        if streamed_text:
            question_mapping = "\n".join(
                f'question_id="{q.question_id}" -> "{q.question_text}"'
                for q in questions
            )
            forecasts = await structure_output(
                streamed_text,
                list[Forecast],
                additional_instructions=clean_indents(
                    f"""
                    Extract forecasts from the text. Match each to a question_id:
                    {question_mapping}
                    Set session_id to "{session_id}" and username to "{username}".
                    Extract prediction, reasoning, and key_sources for each.
                    """
                ),
            )
            for f in forecasts:
                f.session_id = session_id
                f.username = username
                db.save_artifact(session_id, "forecasts", f.to_json(), username)
            st.success(f"Saved {len(forecasts)} forecasts!")
            st.rerun()

    st.markdown("---")
    st.subheader("Add Your Own Forecast")

    with st.form("manual_forecast_form"):
        selected_q = st.selectbox(
            "Select Question",
            options=questions,
            format_func=lambda q: q.question_text[:80],
            key="manual_forecast_question",
        )
        prediction = st.text_input(
            "Your Prediction (e.g., 65% or 0.65)",
            key="manual_forecast_prediction",
        )
        reasoning = st.text_area(
            "Reasoning", key="manual_forecast_reasoning", height=100
        )
        if st.form_submit_button("Save Forecast"):
            if selected_q and prediction.strip():
                forecast = Forecast(
                    session_id=session_id,
                    username=username,
                    question_id=selected_q.question_id,
                    prediction=prediction.strip(),
                    reasoning=reasoning.strip(),
                )
                db.save_artifact(session_id, "forecasts", forecast.to_json(), username)
                st.success("Forecast saved!")
                st.rerun()
            else:
                st.error("Please select a question and enter a prediction.")


asyncio.run(main())
