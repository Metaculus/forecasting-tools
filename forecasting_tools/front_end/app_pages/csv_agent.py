from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import pandas as pd
import streamlit as st
from pydantic import BaseModel, create_model

from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AgentSdkLlm,
    AgentTool,
    AiAgent,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.front_end.app_pages.chat_page import ChatPage
from forecasting_tools.front_end.helpers.app_page import AppPage
from forecasting_tools.front_end.helpers.custom_auth import CustomAuth
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "openrouter/google/gemini-2.5-pro"
MODEL_CHOICES: list[str] = [
    DEFAULT_MODEL,
    "openai/gpt-5",
    "openrouter/x-ai/grok-4",
    "openrouter/anthropic/claude-opus-4.1",
    "openrouter/anthropic/claude-sonnet-4",
    "openai/o3",
    "openai/o4-mini",
    "openai/gpt-4.1",
    "gpt-4o",
    "openrouter/google/gemini-2.5-pro-preview",
]

DEFAULT_BATCH_SIZE: int = 20


class CsvAgentPage(AppPage):
    PAGE_DISPLAY_NAME: str = "📊 CSV Agent"
    URL_PATH: str = "/csv-agent"

    @classmethod
    @CustomAuth.add_access_control()
    async def _async_main(cls) -> None:
        st.title(cls.PAGE_DISPLAY_NAME)

        st.markdown(
            """
            Upload a CSV file and run an AI agent on each row with customizable prompts and structured outputs.

            **How it works:**
            1. Upload your CSV file
            2. Write a prompt template using `{column_name}` to reference CSV columns
            3. Define the JSON output schema for structured responses
            4. Select which tools the agent should have access to
            5. Process all rows and download the results
            """
        )

        csv_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help="Upload a CSV file to process each row with an AI agent",
        )

        if csv_file is not None:
            try:
                df = pd.read_csv(csv_file)
                st.success(
                    f"✅ CSV loaded successfully: {len(df)} rows, {len(df.columns)} columns"
                )

                with st.expander("📋 CSV Preview (first 5 rows)", expanded=True):
                    st.dataframe(df.head())

                await cls._display_configuration_and_process(df)

            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")

        if "processed_df" in st.session_state and "processed_cost" in st.session_state:
            st.divider()
            cls._display_results(
                st.session_state["processed_df"], st.session_state["processed_cost"]
            )

    @classmethod
    async def _display_configuration_and_process(cls, df: pd.DataFrame) -> None:
        with st.form("csv_agent_form"):
            prompt_template = st.text_area(
                "Prompt Template",
                placeholder='Example: "Analyze the sentiment of this review: {review_text}"\n\nUse {column_name} to reference CSV columns.',
                height=450,
                help="Use {column_name} syntax to insert values from CSV columns",
            )

            json_schema_str = st.text_area(
                "JSON Output Schema (flat dictionary)",
                placeholder='{"summary": "str", "score": "int", "category": "str"}',
                height=100,
                help="Define the structure of the output. Each field should map to a Python type as a string.",
            )

            model_choice = st.selectbox(
                "Select Model",
                MODEL_CHOICES,
                index=0,
                help="Choose which AI model to use for processing",
            )

            col1, col2 = st.columns(2)

            with col1:
                batch_size = st.number_input(
                    "Batch Size (concurrent processing)",
                    min_value=1,
                    max_value=100,
                    value=DEFAULT_BATCH_SIZE,
                    help="Number of rows to process concurrently. Higher values are faster but use more resources.",
                )

            with col2:
                max_rows = st.number_input(
                    "Number of rows to process",
                    min_value=1,
                    max_value=len(df),
                    value=len(df),
                    help="Process only the first N rows. Useful for testing on a subset before running on all data.",
                )

            active_tools = cls._display_tool_selection()

            submitted = st.form_submit_button("🚀 Process CSV", type="primary")

            if submitted:
                validation_errors = cls._validate_inputs(
                    df, prompt_template, json_schema_str
                )

                if validation_errors:
                    for error in validation_errors:
                        st.error(error)
                else:
                    if max_rows > 1000:
                        st.warning(
                            f"Please contact Ben if you really want to process more than 1000 rows. Currently selected: {max_rows} rows."
                        )
                        return
                    elif max_rows > 25:
                        st.warning(
                            f"⚠️ You are about to process {max_rows} rows. "
                            "Consider testing on a smaller set first. Some tools "
                            "can be quite expensive and sometimes runs won't stop once started."
                        )
                        confirm = st.checkbox(
                            "I understand and want to proceed", key="confirm_large_csv"
                        )
                        if not confirm:
                            st.info("Please confirm to proceed with processing.")
                            return

                    try:
                        json_schema = json.loads(json_schema_str)
                        pydantic_model = cls._create_pydantic_model_from_schema(
                            json_schema
                        )

                        await cls._process_csv(
                            df,
                            prompt_template,
                            pydantic_model,
                            json_schema,
                            model_choice,
                            active_tools,
                            batch_size,
                            max_rows,
                        )

                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
                        logger.exception("Error processing CSV")

    @classmethod
    def _display_tool_selection(cls) -> list[AgentTool]:
        default_tools: list[AgentTool] = ChatPage.get_chat_tools()
        active_tools: list[AgentTool] = []

        with st.expander("🛠️ Select Tools (optional)", expanded=False):
            st.markdown(
                "*Select which tools the agent can use. Leave all unchecked for no tools.*"
            )

            if st.checkbox(
                "Select/Deselect All Tools", value=False, key="toggle_all_tools"
            ):
                default_checked = True
            else:
                default_checked = False

            for tool in default_tools:
                key = f"csv_tool_{tool.name}"
                tool_active = st.checkbox(
                    tool.name,
                    key=key,
                    value=default_checked,
                    help=(
                        tool.description[:100] + "..."
                        if len(tool.description) > 100
                        else tool.description
                    ),
                )

                if tool_active:
                    active_tools.append(tool)

        return active_tools

    @classmethod
    def _validate_inputs(
        cls, df: pd.DataFrame, prompt_template: str, json_schema_str: str
    ) -> list[str]:
        errors = []

        if not prompt_template.strip():
            errors.append("Prompt template is required")

        if not json_schema_str.strip():
            errors.append("JSON output schema is required")

        try:
            json_schema = json.loads(json_schema_str)
            if not isinstance(json_schema, dict):
                errors.append("JSON schema must be a dictionary")
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON schema: {str(e)}")
            return errors

        template_vars = re.findall(r"\{(\w+)\}", prompt_template)
        missing_columns = [var for var in template_vars if var not in df.columns]
        if missing_columns:
            errors.append(
                f"Template variables not found in CSV columns: {', '.join(missing_columns)}"
            )

        return errors

    @classmethod
    def _create_pydantic_model_from_schema(
        cls, json_schema: dict[str, str]
    ) -> type[BaseModel]:
        type_mapping = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
        }

        field_definitions = {}
        for field_name, type_str in json_schema.items():
            python_type = type_mapping.get(type_str.lower(), str)
            field_definitions[field_name] = (python_type, ...)

        return create_model("DynamicOutputModel", **field_definitions)

    @classmethod
    async def _process_csv(
        cls,
        df: pd.DataFrame,
        prompt_template: str,
        pydantic_model: type[BaseModel],
        json_schema: dict[str, str],
        model_choice: str,
        active_tools: list[AgentTool],
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_rows: int | None = None,
    ) -> None:
        original_row_count = len(df)
        if max_rows is not None and max_rows < original_row_count:
            df = df.head(max_rows)
            st.info(
                f"ℹ️ Processing only the first {max_rows} rows (out of {original_row_count} total rows in CSV)"
            )

        st.subheader("Processing Rows...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        results: list[dict[str, Any]] = [{}] * len(df)
        total_cost = 0.0
        processed_count = 0

        rows_with_indices = [(i, row) for i, (_, row) in enumerate(df.iterrows())]

        for batch_start in range(0, len(rows_with_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(rows_with_indices))
            batch = rows_with_indices[batch_start:batch_end]

            status_text.text(
                f"Processing rows {batch_start + 1}-{batch_end} of {len(df)} "
                f"(batch of {len(batch)})..."
            )

            batch_tasks = [
                cls._process_single_row(
                    idx,
                    row,
                    prompt_template,
                    pydantic_model,
                    json_schema,
                    model_choice,
                    active_tools,
                )
                for idx, row in batch
            ]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for i, (original_idx, _) in enumerate(batch):
                result = batch_results[i]
                if isinstance(result, Exception):
                    logger.error(f"Error processing row {original_idx}: {str(result)}")
                    row_dict = rows_with_indices[original_idx][1].to_dict()
                    for field_name in json_schema.keys():
                        row_dict[field_name] = None
                    row_dict["error"] = str(result)
                    results[original_idx] = row_dict
                elif isinstance(result, dict):
                    results[original_idx] = result["result"]
                    total_cost += result["cost"]

                processed_count += 1
                progress_bar.progress(processed_count / len(df))

        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")

        result_df = pd.DataFrame(results)

        st.session_state["processed_df"] = result_df
        st.session_state["processed_cost"] = total_cost

        st.success(
            f"Processing complete! Processed {len(df)} rows in batches of {batch_size}."
        )

    @classmethod
    async def _process_single_row(
        cls,
        idx: int,
        row: pd.Series,
        prompt_template: str,
        pydantic_model: type[BaseModel],
        json_schema: dict[str, str],
        model_choice: str,
        active_tools: list[AgentTool],
    ) -> dict[str, Any]:
        try:
            filled_prompt = prompt_template.format(**row.to_dict())

            with MonetaryCostManager() as cost_manager:
                agent_output = await cls._run_agent(
                    filled_prompt, model_choice, active_tools
                )

                structured_output = await structure_output(
                    agent_output, pydantic_model, model=model_choice
                )

                row_cost = cost_manager.current_usage

            result_dict = row.to_dict()
            for field_name in json_schema.keys():
                result_dict[field_name] = getattr(structured_output, field_name)
            result_dict["error"] = None

            return {"result": result_dict, "cost": row_cost}

        except Exception as e:
            logger.error(f"Error processing row {idx}: {str(e)}")
            result_dict = row.to_dict()
            for field_name in json_schema.keys():
                result_dict[field_name] = None
            result_dict["error"] = str(e)

            return {"result": result_dict, "cost": 0.0}

    @classmethod
    async def _run_agent(
        cls, prompt: str, model_choice: str, active_tools: list[AgentTool]
    ) -> str:
        instructions = clean_indents(
            """
            You are a helpful AI assistant processing data from a CSV file.

            Follow these guidelines:
            - Follow the user's instructions carefully
            - Be concise and accurate in your responses
            - If you use tools, incorporate their outputs into your final answer
            - Return your answer in a clear format that can be structured
            """
        )

        agent = AiAgent(
            name="CSV Processing Agent",
            instructions=instructions,
            model=AgentSdkLlm(model=model_choice),
            tools=active_tools if active_tools else [],  # type: ignore
            handoffs=[],
        )

        result = await AgentRunner.run(agent, prompt, max_turns=15)

        final_output = result.final_output
        if not isinstance(final_output, str):
            final_output = str(final_output)

        return final_output

    @classmethod
    def _display_results(cls, result_df: pd.DataFrame, total_cost: float) -> None:
        st.subheader("Results")

        st.warning(
            f"⚠️ Estimated cost: ${total_cost:.2f} "
            "(some tool cost estimates are wildly off, talk to Ben if unsure about cost)"
        )

        error_count = result_df["error"].notna().sum()
        if error_count > 0:
            st.warning(
                f"⚠️ {error_count} rows encountered errors. Check the 'error' column for details."
            )

        st.dataframe(result_df, use_container_width=True)


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()
    CsvAgentPage.main()
