from __future__ import annotations

import logging
import os
import time
from datetime import datetime

import streamlit as st
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel, Field

from forecasting_tools.agents_and_tools.minor_tools import (
    create_tool_for_forecasting_bot,
    grab_open_questions_from_tournament,
    grab_question_details_from_metaculus,
    perplexity_quick_search_high_context,
    perplexity_reasoning_pro_search,
    query_asknews,
    smart_searcher_search,
)
from forecasting_tools.agents_and_tools.other.data_analyzer import DataAnalyzer
from forecasting_tools.agents_and_tools.other.hosted_file import (
    FileToUpload,
    HostedFile,
)
from forecasting_tools.agents_and_tools.question_generators.info_hazard_identifier import (
    InfoHazardIdentifier,
)
from forecasting_tools.agents_and_tools.question_generators.question_decomposer import (
    QuestionDecomposer,
)
from forecasting_tools.agents_and_tools.question_generators.question_operationalizer import (
    QuestionOperationalizer,
)
from forecasting_tools.agents_and_tools.question_generators.topic_generator import (
    TopicGenerator,
)
from forecasting_tools.agents_and_tools.research.computer_use import ComputerUse
from forecasting_tools.agents_and_tools.research.find_a_dataset import DatasetFinder
from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AgentSdkLlm,
    AgentTool,
    AiAgent,
    event_to_tool_message,
    general_trace_or_span,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecast_bots.bot_lists import get_all_important_bot_classes
from forecasting_tools.front_end.helpers.app_page import AppPage
from forecasting_tools.front_end.helpers.report_displayer import ReportDisplayer
from forecasting_tools.helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


DEFAULT_MODEL: str = (
    "openrouter/google/gemini-2.5-pro"  # "openrouter/anthropic/claude-sonnet-4"
)
MODEL_CHOICES: list[str] = [
    DEFAULT_MODEL,
    "openrouter/anthropic/claude-sonnet-4",
    "openai/o3",
    "openai/o4-mini",
    "openai/gpt-4.1",
    "gpt-4o",
    "openrouter/google/gemini-2.5-pro-preview",
]


class ChatSession(BaseModel, Jsonable):
    name: str
    messages: list[dict]
    model_choice: str = DEFAULT_MODEL
    trace_id: str | None = None
    last_chat_cost: float | None = None
    last_chat_duration: float | None = None
    time_stamp: datetime = Field(default_factory=datetime.now)


class ChatPage(AppPage):
    PAGE_DISPLAY_NAME: str = "💬 Chatbot"
    URL_PATH: str = "/chat"
    ENABLE_HEADER: bool = False
    ENABLE_FOOTER: bool = False
    DEFAULT_MESSAGE: dict = {
        "role": "assistant",
        "content": "How may I assist you today?",
    }

    @classmethod
    async def _async_main(cls) -> None:
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [cls.DEFAULT_MESSAGE]
        cls.display_debug_mode()
        st.sidebar.button("Clear Chat History", on_click=cls.clear_chat_history)
        cls.display_model_selector()
        active_tools = cls.display_tools()
        cls.display_chat_metadata()
        cls.display_premade_examples()
        cls.display_chat_files()
        st.sidebar.write("---")
        cls.display_messages(st.session_state.messages)
        prompt = cls.display_chat_bar_and_gather_files()
        await cls.process_prompt(prompt, active_tools)

    @classmethod
    def display_debug_mode(cls) -> None:
        local_streamlit_mode = (
            os.getenv("LOCAL_STREAMLIT_MODE", "false").lower() == "true"
        )
        if local_streamlit_mode:
            if st.sidebar.checkbox("Debug Mode", value=False):
                st.session_state["debug_mode"] = True
            else:
                st.session_state["debug_mode"] = False

    @classmethod
    def display_model_selector(cls) -> None:
        if "model_choice" not in st.session_state.keys():
            st.session_state["model_choice"] = DEFAULT_MODEL
        model_name: str = st.session_state["model_choice"]
        model_choice = st.sidebar.selectbox(
            "Choose your model (used for chat, not tools)",
            MODEL_CHOICES,
            index=MODEL_CHOICES.index(model_name),
        )
        st.session_state["model_choice"] = model_choice

    @classmethod
    def get_chat_tools(cls) -> list[AgentTool]:
        return [
            TopicGenerator.find_random_headlines_tool,
            QuestionDecomposer.decompose_into_questions_tool,
            QuestionOperationalizer.question_operationalizer_tool,
            perplexity_reasoning_pro_search,
            query_asknews,
            smart_searcher_search,
            grab_question_details_from_metaculus,
            grab_open_questions_from_tournament,
            TopicGenerator.get_headlines_on_random_company_tool,
            perplexity_quick_search_high_context,
            InfoHazardIdentifier.info_hazard_identifier_tool,
            DataAnalyzer.data_analysis_tool,
            ComputerUse.computer_use_tool,
            DatasetFinder.find_a_dataset_tool,
        ]

    @classmethod
    def display_tools(cls) -> list[AgentTool]:
        default_tools: list[AgentTool] = cls.get_chat_tools()
        bot_options = get_all_important_bot_classes()

        active_tools: list[AgentTool] = []
        with st.sidebar.expander("🛠️ Select Tools"):
            bot_choice = st.selectbox(
                "Select a bot for forecast_question_tool (Main Bot is best)",
                [bot.__name__ for bot in bot_options],
            )
            bot = next(bot for bot in bot_options if bot.__name__ == bot_choice)
            default_tools = [create_tool_for_forecasting_bot(bot)] + default_tools

            tool_names = [tool.name for tool in default_tools]
            all_checked = all(
                st.session_state.get(f"tool_{name}", True) for name in tool_names
            )
            toggle_label = "Toggle all Tools"
            if st.button(toggle_label):
                for name in tool_names:
                    st.session_state[f"tool_{name}"] = not all_checked
            for tool in default_tools:
                key = f"tool_{tool.name}"
                if key not in st.session_state:
                    st.session_state[key] = True

                tool_active = st.checkbox(tool.name, key=key)

                if tool_active:
                    active_tools.append(tool)

        with st.sidebar.expander("ℹ️ Tool Explanations"):
            for tool in active_tools:
                assert isinstance(
                    tool, AgentTool
                ), f"Tool {tool.name} is not an AgentTool"
                property_description = ""
                for property_name, metadata in tool.params_json_schema[
                    "properties"
                ].items():
                    description = metadata.get("description", "No description provided")
                    field_type = metadata.get("type", "No type provided")
                    property_description += (
                        f"- {property_name}: {description} (type: {field_type})\n"
                    )
                st.write(
                    clean_indents(
                        f"""
                        **{tool.name}**

                        {clean_indents(tool.description)}

                        Input Arguments:
                        {clean_indents(property_description)}

                        ---

                        """
                    )
                )
        return active_tools

    @classmethod
    def display_chat_metadata(cls) -> None:
        with st.sidebar.expander("📈 Chat Metadata"):
            debug_mode = st.session_state.get("debug_mode", False)
            if "last_chat_cost" not in st.session_state.keys():
                st.session_state.last_chat_cost = 0
            if st.session_state.last_chat_cost > 0 and debug_mode:
                st.markdown(
                    f"**Last Chat Cost:** ${st.session_state.last_chat_cost:.7f}"
                )
            if "last_chat_duration" in st.session_state.keys():
                st.markdown(
                    f"**Last Chat Duration:** {st.session_state.last_chat_duration:.2f} seconds"
                )
            if "trace_id" in st.session_state.keys():
                trace_id = st.session_state.trace_id
                st.markdown(
                    f"**Conversation in Foresight Project:** [link](https://platform.openai.com/traces/trace?trace_id={trace_id})"
                )

    @classmethod
    def display_premade_examples(cls) -> None:
        save_path = "front_end/saved_chats.json"
        debug_mode = st.session_state.get("debug_mode", False)
        try:
            saved_sessions = ChatSession.load_json_from_file_path(save_path)
        except Exception:
            saved_sessions = []
            st.sidebar.warning("No saved chat sessions found")
        with st.expander("📚 Getting Started", expanded=True):
            st.write(
                """
                Welcome to the [forecasting-tools](https://github.com/Metaculus/forecasting-tools) chatbot!
                This is a chatbot to help with forecasting tasks that has access to a number of custom tools useful for forecasting!
                1. **See examples**: Explore examples of some of the tools being used via the buttons below
                2. **Choose tools**: Choose which tools you want to use in the sidebar (or leave all of them active and let the AI decide)
                3. **Ask a question**: Click 'Clear Chat History' to start a new conversation and ask a question! See the full detailed output of tools populate in the sidebar.
                """
            )
            if saved_sessions:
                for session in saved_sessions:
                    if st.button(session.name, key=session.name):
                        st.session_state.messages = session.messages
                        st.session_state.model_choice = session.model_choice
                        if session.trace_id:
                            st.session_state.trace_id = session.trace_id
                        if session.last_chat_cost:
                            st.session_state.last_chat_cost = session.last_chat_cost
                        if session.last_chat_duration:
                            st.session_state.last_chat_duration = (
                                session.last_chat_duration
                            )
                        logger.info(
                            f"Chat session '{session.name}' loaded. Rerunning app"
                        )
                        st.rerun()
                    if debug_mode:
                        if st.button("🗑️", key=f"delete_{session.name}"):
                            saved_sessions.remove(session)
                            ChatSession.save_object_list_to_file_path(
                                saved_sessions, save_path
                            )
                            st.sidebar.success(
                                f"Chat session '{session.name}' deleted."
                            )
                            logger.info(
                                f"Chat session '{session.name}' deleted. Rerunning app"
                            )
                            st.rerun()
            if debug_mode:
                if "chat_save_name" not in st.session_state:
                    st.session_state["chat_save_name"] = ""
                st.text_input("Chat Session Name", key="chat_save_name")
                if st.button("Save Chat Session"):
                    chat_session = ChatSession(
                        name=st.session_state["chat_save_name"],
                        model_choice=st.session_state["model_choice"],
                        messages=st.session_state.messages,
                        trace_id=st.session_state.trace_id,
                        last_chat_cost=st.session_state.last_chat_cost,
                        last_chat_duration=st.session_state.last_chat_duration,
                    )
                    saved_sessions.append(chat_session)
                    ChatSession.save_object_list_to_file_path(saved_sessions, save_path)
                    st.sidebar.success(f"Chat session '{chat_session.name}' saved.")
                    logger.info(
                        f"Chat session '{chat_session.name}' saved. Rerunning app"
                    )
                    st.rerun()

    @classmethod
    def display_chat_files(cls) -> None:
        if "chat_files" not in st.session_state.keys():
            st.session_state.chat_files = []
        session_files: list[HostedFile] = st.session_state.chat_files
        with st.sidebar.expander("📁 Uploaded Files", expanded=True):
            debug_mode = st.session_state.get("debug_mode", False)
            for i, file in enumerate(session_files):
                st.write(f"File {i+1}: `{file.file_name}`")
                if debug_mode:
                    st.write(f"File ID: `{file.file_id}`")

    @classmethod
    def display_messages(cls, messages: list[dict]) -> None:
        assistant_message_num = 0
        st.sidebar.write("**Tool Calls and Outputs:**")
        for message in messages:
            output_emoji = "📝"
            call_emoji = "📞"
            if "type" in message and message["type"] == "function_call":
                call_id = message["name"]
                with st.sidebar.expander(f"{call_emoji} Call: {call_id}"):
                    st.write(f"Call ID: {message['call_id']}")
                    st.write(f"Assistant Message Number: {assistant_message_num}")
                    st.write(f"Function: {message['name']}")
                    st.write(f"Arguments: {message['arguments']}")
                    continue
            if "type" in message and message["type"] == "function_call_output":
                call_id = message["call_id"]
                with st.sidebar.expander(f"{output_emoji} Output: {call_id}"):
                    st.write(f"Call ID: {message['call_id']}")
                    st.write(f"Assistant Message Number: {assistant_message_num}")
                    st.write(
                        ReportDisplayer.clean_markdown(
                            f"Output:\n\n{message['output']}"
                        )
                    )
                    continue

            try:
                role = message["role"]
            except KeyError:
                if "type" in message and message["type"] == "reasoning":
                    logger.warning(f"Found message with no role: {message}")
                else:
                    st.error(f"Unexpected message role. Message: {message}")
                continue

            with st.chat_message(role):
                if role == "assistant":
                    assistant_message_num += 1
                content = message["content"]
                if isinstance(content, list):
                    text = content[0]["text"]
                else:
                    text = content
                st.write(ReportDisplayer.clean_markdown(text))

    @classmethod
    def display_chat_bar_and_gather_files(cls) -> str | None:
        if chat_input := st.chat_input(accept_file=True):
            prompt = chat_input.text

            input_files = chat_input.files
            if "chat_files" not in st.session_state.keys():
                st.session_state.chat_files = []
            session_files = st.session_state.chat_files

            if input_files:
                files_to_upload = []
                for file in input_files:
                    files_to_upload.append(
                        FileToUpload(file_data=file, file_name=file.name)
                    )
                new_files = HostedFile.upload_files_to_openai(files_to_upload)
                session_files.extend(new_files)
                for new_file in new_files:
                    prompt += f"\n\n*[User uploaded file: {new_file.file_name}]*"

            st.session_state.chat_files = session_files
        else:
            prompt = None
        return prompt

    @classmethod
    async def process_prompt(
        cls, prompt: str | None, active_tools: list[AgentTool]
    ) -> None:
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
        if st.session_state.messages[-1]["role"] != "assistant":
            with MonetaryCostManager(10) as cost_manager:
                start_time = time.time()
                await cls.generate_response(prompt, active_tools)
                st.session_state.last_chat_cost = cost_manager.current_usage
                end_time = time.time()
                st.session_state.last_chat_duration = end_time - start_time
            st.rerun()

    @classmethod
    async def generate_response(
        cls,
        prompt_input: str | None,
        active_tools: list[AgentTool],
    ) -> None:
        if not prompt_input:
            return

        chat_files: list[HostedFile] = st.session_state.chat_files
        file_instructions = "You have access to the following files:\n"
        for file in chat_files:
            file_instructions += (
                f"- File Name: {file.file_name} | File ID: {file.file_id}\n\n"
            )

        instructions = clean_indents(
            f"""
            # Instructions
            You are a helpful assistant hired to help with forecasting related tasks.
            - When a tool gives you answers that are cited, ALWAYS include the links in your responses. Keep the links inline as much as you can.
            - If you can, you infer the inputs to tools rather than ask for them.
            - If a tool call fails, you say so rather than giving a back up answer. ALWAYS state errors. NEVER give a back up answer.
            - Whenever possible, please paralelize your tool calls and split tasks into parallel subtasks. However, don't do this if tasks are dependent on each other (e.g. you need metaculus question information BEFORE running a forecast)
            - By default, restate ALL the output that tools give you in readable markdown to the user. Do this even if the tool output is long.
            - Format your response as Markdown parsable in streamlit.write() function
            - If the forecast_question_tool is available, always use this when forecasting unless someone asks you not to.

            # Payment
            Please first prioritize the usefulness of your response and choosing good tools to answer the user's question. You are being hired for this capability.
            You will also receive a large bonus if you can consistently cite exactly the links given to you in full from tool calls (this includes not add any links that are not in tool calls)

            {"# Files" if chat_files else ""}
            {file_instructions if chat_files else ""}
            """
        )

        model_choice = st.session_state["model_choice"]

        agent = AiAgent(
            name="Chat App Agent",
            instructions=instructions,
            model=AgentSdkLlm(model=model_choice),
            tools=active_tools,  # type: ignore
            handoffs=[],
        )

        with general_trace_or_span("Chat App") as chat_trace:
            result = AgentRunner.run_streamed(
                agent, st.session_state.messages, max_turns=20
            )
            streamed_text = ""
            with st.chat_message("assistant"):
                placeholder = st.empty()
            with st.spinner("Thinking..."):
                async for event in result.stream_events():
                    if event.type == "raw_response_event" and isinstance(
                        event.data, ResponseTextDeltaEvent
                    ):
                        streamed_text += event.data.delta
                    placeholder.write(streamed_text)

                    new_reasoning = event_to_tool_message(event)
                    if new_reasoning:
                        st.sidebar.write(new_reasoning)

            # logger.info(f"Chat finished with output: {streamed_text}")
            st.session_state.messages = result.to_input_list()
            st.session_state.trace_id = chat_trace.trace_id
            cls._update_last_message_if_gemini_bug(model_choice)

        ForecastDatabaseManager.add_general_report_to_database(
            question_text=prompt_input,
            background_info=None,
            resolution_criteria=None,
            fine_print=None,
            prediction=None,
            explanation=streamed_text,
            page_url=None,
            price_estimate=None,
            run_type=ForecastRunType.WEB_APP_CHAT,
        )

    @classmethod
    def _update_last_message_if_gemini_bug(cls, model_choice: str) -> None:
        last_3_messages = st.session_state.messages[-3:]
        for message in last_3_messages:
            if (
                "type" in message
                and message["type"] == "function_call_output"
                and "gemini" in model_choice
                and "question_details" in message["call_id"]
            ):
                last_message = st.session_state.messages[-1]
                output = message["output"]
                last_message["content"][0][
                    "text"
                ] += f"\n\n---\n\nNOTICE: There is a bug in gemini tool calling in OpenAI agents SDK, here is the content. Consider using openrouter/anthropic/claude-sonnet-4:\n\n {output}."

    @classmethod
    def clear_chat_history(cls) -> None:
        st.session_state.messages = [cls.DEFAULT_MESSAGE]
        st.session_state.trace_id = None
        st.session_state.chat_files = []


if __name__ == "__main__":
    ChatPage.main()
