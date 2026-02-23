import asyncio

import streamlit as st

from forecasting_tools.agents_and_tools.decision_hub.data_models import (
    Scenario,
    ScenarioSet,
)
from forecasting_tools.agents_and_tools.decision_hub.pages._shared import (
    get_db,
    get_session_info,
    load_research_reports,
    load_scenario_set,
    require_active_session,
    stream_agent_response,
)
from forecasting_tools.agents_and_tools.decision_hub.scenario_agent import (
    generate_scenarios_streamed,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.misc import clean_indents


async def main() -> None:
    st.title("3. Scenario Generation")
    if not require_active_session():
        return

    username, session_id, policy_question = get_session_info()
    db = get_db()

    st.info(f"**Policy Question:** {policy_question}")

    existing_scenarios = load_scenario_set(db, session_id)
    if existing_scenarios:
        st.subheader("Current Scenarios")
        if existing_scenarios.rationale:
            st.write(f"**Rationale:** {existing_scenarios.rationale}")

        if existing_scenarios.drivers:
            st.markdown("### Drivers")
            for driver in existing_scenarios.drivers:
                with st.expander(f"Driver: {driver.name}"):
                    st.write(f"**Description:** {driver.description}")
                    st.write(f"**High state:** {driver.high_state}")
                    st.write(f"**Low state:** {driver.low_state}")

        if existing_scenarios.scenarios:
            st.markdown("### Scenarios")
            for scenario in existing_scenarios.scenarios:
                with st.expander(f"Scenario: {scenario.name}"):
                    st.write(f"**Narrative:** {scenario.narrative}")
                    if scenario.driver_states:
                        for d_name, state in scenario.driver_states.items():
                            st.write(f"- {d_name}: {state}")
                    if scenario.probability is not None:
                        st.write(f"**Probability:** {scenario.probability:.0%}")

    st.markdown("---")
    st.subheader("Generate AI Scenarios")

    research_reports = load_research_reports(db, session_id)
    research_context = "\n\n".join(r.report_markdown for r in research_reports)

    if not research_context.strip():
        st.warning(
            "No research found. Go to **2. Research** first, or generate scenarios without research context."
        )

    if st.button("Generate Scenarios", key="gen_scenarios_btn", type="primary"):
        with MonetaryCostManager():
            result = generate_scenarios_streamed(
                policy_question, research_context or "No prior research available."
            )
            streamed_text = await stream_agent_response(result)

        if streamed_text:
            scenario_set = await structure_output(
                streamed_text,
                ScenarioSet,
                additional_instructions=clean_indents(
                    f"""
                    Extract scenario drivers and scenarios from the text.
                    Set session_id to "{session_id}" and username to "{username}".
                    For each driver, extract name, description, high_state, low_state.
                    For each scenario, extract name, narrative, driver_states, probability.
                    Also extract the rationale text.
                    """
                ),
            )
            scenario_set.session_id = session_id
            scenario_set.username = username
            db.save_artifact(session_id, "scenarios", scenario_set.to_json(), username)
            st.success("Scenarios saved!")
            st.rerun()

    st.markdown("---")
    st.subheader("Add Your Own Scenario")

    with st.form("manual_scenario_form"):
        scenario_name = st.text_input("Scenario Name", key="manual_scenario_name")
        scenario_narrative = st.text_area(
            "Narrative (2-3 sentences)", key="manual_scenario_narrative"
        )
        scenario_prob = st.slider(
            "Estimated Probability",
            0.0,
            1.0,
            0.25,
            0.05,
            key="manual_scenario_prob",
        )
        if st.form_submit_button("Add Scenario"):
            if scenario_name.strip() and scenario_narrative.strip():
                new_scenario = Scenario(
                    name=scenario_name.strip(),
                    narrative=scenario_narrative.strip(),
                    probability=scenario_prob,
                )
                current_set = load_scenario_set(db, session_id)
                if current_set:
                    current_set.scenarios.append(new_scenario)
                else:
                    current_set = ScenarioSet(
                        session_id=session_id,
                        username=username,
                        scenarios=[new_scenario],
                    )
                db.save_artifact(
                    session_id, "scenarios", current_set.to_json(), username
                )
                st.success(f"Scenario '{scenario_name}' added!")
                st.rerun()
            else:
                st.error("Please fill in both name and narrative.")


asyncio.run(main())
