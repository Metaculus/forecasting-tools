import asyncio

import streamlit as st

from forecasting_tools.agents_and_tools.decision_hub.autopilot_runner import (
    STEP_NAMES,
    run_full_pipeline,
)
from forecasting_tools.agents_and_tools.decision_hub.database import get_database
from forecasting_tools.agents_and_tools.decision_hub.pages._shared import (
    get_session_info,
    require_active_session,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)

STEP_DISPLAY_NAMES = {
    "research": "Background Research",
    "scenarios": "Scenario Generation",
    "forecast_questions": "Forecast Questions",
    "forecasts": "Forecasts",
    "proposal": "Policy Proposal",
    "robustness": "Robustness Analysis",
    "synthesis": "Final Synthesis",
}


async def main() -> None:
    st.title("Autopilot")
    if not require_active_session():
        return

    username, session_id, policy_question = get_session_info()

    st.info(f"**Policy Question:** {policy_question}")

    st.write(
        "Run the entire analysis pipeline automatically. The AI will execute "
        "all steps sequentially (Research -> Scenarios -> Forecast Questions -> "
        "Forecasts -> Proposal -> Robustness -> Synthesis)."
    )

    st.warning(
        "This will take approximately 3-5 minutes and will overwrite any "
        "existing AI-generated artifacts for this analysis. Your manually "
        "added content will be preserved."
    )

    if st.button(
        "Run Full Pipeline",
        key="run_autopilot_btn",
        type="primary",
    ):
        db = get_database()
        status_container = st.status("Running autopilot...", expanded=True)
        completed_steps: list[str] = []

        def on_step_start(step_name: str) -> None:
            display_name = STEP_DISPLAY_NAMES.get(step_name, step_name)
            status_container.write(f"Starting: {display_name}...")

        def on_step_complete(step_name: str) -> None:
            display_name = STEP_DISPLAY_NAMES.get(step_name, step_name)
            completed_steps.append(step_name)
            progress = len(completed_steps) / len(STEP_NAMES)
            status_container.write(f"Completed: {display_name}")
            status_container.progress(progress)

        try:
            with MonetaryCostManager():
                session = await run_full_pipeline(
                    session_id,
                    db,
                    on_step_start=on_step_start,
                    on_step_complete=on_step_complete,
                )

            status_container.update(label="Autopilot complete!", state="complete")
            st.session_state["active_session"] = session
            st.success(
                "All steps completed! Navigate through the sidebar to review "
                "and edit each step's output."
            )
            st.balloons()

        except Exception as e:
            status_container.update(label="Autopilot failed", state="error")
            st.error(f"Pipeline failed: {e}")
            if completed_steps:
                st.info(
                    f"Completed steps before failure: {', '.join(STEP_DISPLAY_NAMES.get(s, s) for s in completed_steps)}. "
                    "You can review these in the sidebar."
                )


asyncio.run(main())
