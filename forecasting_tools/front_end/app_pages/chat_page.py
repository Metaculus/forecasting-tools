from __future__ import annotations

import logging
import os
import time

import streamlit as st
from agents import Agent, RunItem, Runner, Tool, trace
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel

from forecasting_tools.agents_and_tools.misc_tools import (
    create_tool_for_forecasting_bot,
    get_general_news_with_asknews,
    grab_open_questions_from_tournament,
    grab_question_details_from_metaculus,
    perplexity_pro_search,
    perplexity_quick_search,
    smart_searcher_search,
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
from forecasting_tools.ai_models.agent_wrappers import AgentSdkLlm, AgentTool
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecast_bots.bot_lists import (
    get_all_important_bot_classes,
)
from forecasting_tools.front_end.helpers.app_page import AppPage
from forecasting_tools.front_end.helpers.report_displayer import (
    ReportDisplayer,
)
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


DEFAULT_MODEL: str = (
    "openrouter/google/gemini-2.5-pro-preview"  # "openrouter/anthropic/claude-sonnet-4"
)


class ChatSession(BaseModel, Jsonable):
    name: str
    messages: list[dict]
    model_choice: str = DEFAULT_MODEL
    trace_id: str | None = None
    last_chat_cost: float | None = None
    last_chat_duration: float | None = None


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
        st.sidebar.button(
            "Clear Chat History", on_click=cls.clear_chat_history
        )
        cls.display_model_selector()
        active_tools = cls.display_tools()
        cls.display_chat_metadata()
        cls.display_premade_examples()
        st.sidebar.write("---")
        cls.display_messages(st.session_state.messages)

        if prompt := st.chat_input():
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )
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
    def display_messages(cls, messages: list[dict]) -> None:
        assistant_message_num = 0
        st.sidebar.write("**Tool Calls and Outputs:**")
        for message in messages:
            output_emoji = "🔍"
            call_emoji = "📞"
            if "type" in message and message["type"] == "function_call":
                call_id = message["name"]
                with st.sidebar.expander(f"{call_emoji} Call: {call_id}"):
                    st.write(f"Function: {message['name']}")
                    st.write(f"Arguments: {message['arguments']}")
                    st.write(f"Call ID: {message['call_id']}")
                    st.write(
                        f"Assistant Message Number: {assistant_message_num}"
                    )
                    continue
            if "type" in message and message["type"] == "function_call_output":
                call_id = message["call_id"]
                with st.sidebar.expander(f"{output_emoji} Output: {call_id}"):
                    st.write(f"Call ID: {message['call_id']}")
                    st.write(
                        f"Assistant Message Number: {assistant_message_num}"
                    )
                    st.write(f"Output:\n\n{message['output']}")
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
    def display_debug_mode(cls) -> None:
        local_streamlit_mode = (
            os.getenv("LOCAL_STREAMLIT_MODE", "false").lower() == "true"
        )
        if local_streamlit_mode:
            if st.sidebar.checkbox("Debug Mode", value=True):
                st.session_state["debug_mode"] = True
            else:
                st.session_state["debug_mode"] = False

    @classmethod
    def display_model_selector(cls) -> None:
        if "model_choice" not in st.session_state.keys():
            st.session_state["model_choice"] = DEFAULT_MODEL
        model_name: str = st.session_state["model_choice"]
        model_choice = st.sidebar.text_input(
            "Litellm compatible model used for chat (not tools)",
            value=model_name,
        )
        if "o1-pro" in model_choice or "gpt-4.5" in model_choice:
            raise ValueError(
                "o1 pro and gpt-4.5 are not available for this application."
            )
        st.session_state["model_choice"] = model_choice

        # # TODO: When future versions of openai-agents come out, check if AgentSdkLlm works
        # st.session_state["model_choice"] = DEFAULT_MODEL

    @classmethod
    def get_chat_tools(cls) -> list[Tool]:
        return [
            TopicGenerator.find_random_headlines_tool,
            QuestionDecomposer.decompose_into_questions_tool,
            QuestionOperationalizer.question_operationalizer_tool,
            perplexity_pro_search,
            get_general_news_with_asknews,
            smart_searcher_search,
            grab_question_details_from_metaculus,
            grab_open_questions_from_tournament,
            TopicGenerator.get_headlines_on_random_company_tool,
            perplexity_quick_search,
            InfoHazardIdentifier.info_hazard_identifier_tool,
        ]

    @classmethod
    def display_tools(cls) -> list[Tool]:
        default_tools: list[Tool] = cls.get_chat_tools()
        bot_options = get_all_important_bot_classes()

        active_tools: list[Tool] = []
        with st.sidebar.expander("Select Tools"):
            bot_choice = st.selectbox(
                "Select a bot for forecast_question_tool (Main Bot is best)",
                [bot.__name__ for bot in bot_options],
            )
            bot = next(
                bot for bot in bot_options if bot.__name__ == bot_choice
            )
            default_tools = [
                create_tool_for_forecasting_bot(bot)
            ] + default_tools

            tool_names = [tool.name for tool in default_tools]
            all_checked = all(
                st.session_state.get(f"tool_{name}", True)
                for name in tool_names
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

        with st.sidebar.expander("Tool Explanations"):
            for tool in active_tools:
                if isinstance(tool, AgentTool):
                    property_description = ""
                    for property_name, metadata in tool.params_json_schema[
                        "properties"
                    ].items():
                        description = metadata.get(
                            "description", "No description provided"
                        )
                        field_type = metadata.get("type", "No type provided")
                        property_description += f"- {property_name}: {description} (type: {field_type})\n"
                    st.write(
                        clean_indents(
                            f"""
                            **{tool.name}**

                            {clean_indents(tool.description)}

                            {clean_indents(property_description)}

                            ---

                            """
                        )
                    )
        return active_tools

    @classmethod
    def display_chat_metadata(cls) -> None:
        with st.sidebar.expander("Chat Metadata"):
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
                            st.session_state.last_chat_cost = (
                                session.last_chat_cost
                            )
                        if session.last_chat_duration:
                            st.session_state.last_chat_duration = (
                                session.last_chat_duration
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
                    ChatSession.save_object_list_to_file_path(
                        saved_sessions, save_path
                    )
                    st.sidebar.success(
                        f"Chat session '{chat_session.name}' saved."
                    )
                    st.rerun()

    @classmethod
    async def generate_response(
        cls,
        prompt_input: str | None,
        active_tools: list[Tool],
    ) -> None:
        if not prompt_input:
            return

        instructions = clean_indents(
            """
            You are a helpful assistant.
            - When a tool gives you answers that are cited, ALWAYS include the links in your responses. Keep the links inline as much as you can.
            - If you can, you infer the inputs to tools rather than ask for them.
            - If a tool call fails, you say so rather than giving a back up answer.
            - Whenever possible, please paralelize your tool calls and split tasks into parallel subtasks. However, don't do this if tasks are dependent on each other (e.g. you need metaculus question information BEFORE running a forecast)
            - By default, restate ALL the output that tools give you in readable markdown to the user. Do this even if the tool output is long.
            - Format your response as Markdown parsable in streamlit.write() function
            - If the forecast_question_tool is available, always use this when forecasting unless someone asks you not to.
            """
        )

        model_choice = st.session_state["model_choice"]

        agent = Agent(
            name="Assistant",
            instructions=instructions,
            model=AgentSdkLlm(model=model_choice),
            tools=active_tools,
            handoffs=[],
        )

        with trace("Chat App") as chat_trace:
            result = Runner.run_streamed(
                agent, st.session_state.messages, max_turns=20
            )
            streamed_text = ""
            with st.chat_message("assistant"):
                placeholder = st.empty()
            with st.spinner("Thinking..."):
                async for event in result.stream_events():
                    new_reasoning = ""
                    if event.type == "raw_response_event" and isinstance(
                        event.data, ResponseTextDeltaEvent
                    ):
                        streamed_text += event.data.delta
                    elif event.type == "run_item_stream_event":
                        new_reasoning = (
                            f"{cls._grab_text_of_item(event.item)}\n\n"
                        )
                    # elif event.type == "agent_updated_stream_event":
                    #     reasoning_text += f"Agent updated: {event.new_agent.name}\n\n"
                    placeholder.write(streamed_text)
                    if new_reasoning:
                        st.sidebar.write(new_reasoning)

        # logger.info(f"Chat finished with output: {streamed_text}")
        st.session_state.messages = result.to_input_list()
        st.session_state.trace_id = chat_trace.trace_id
        cls._update_last_message_if_gemini_bug(model_choice)

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
    def _grab_text_of_item(cls, item: RunItem) -> str:
        text = ""
        if item.type == "message_output_item":
            content = item.raw_item.content[0]
            if content.type == "output_text":
                # text = content.text
                text = ""  # the text is already streamed
            elif content.type == "output_refusal":
                text = content.refusal
            else:
                text = "Error: unknown content type"
        elif item.type == "tool_call_item":
            tool_name = getattr(item.raw_item, "name", "unknown_tool")
            tool_args = getattr(item.raw_item, "arguments", {})
            text = f"Tool call: {tool_name}({tool_args})"
        elif item.type == "tool_call_output_item":
            output = getattr(item, "output", str(item.raw_item))
            text = f"Tool output:\n\n{output}"
        elif item.type == "handoff_call_item":
            handoff_info = getattr(item.raw_item, "name", "handoff")
            text = f"Handoff call: {handoff_info}"
        elif item.type == "handoff_output_item":
            text = f"Handoff output: {str(item.raw_item)}"
        elif item.type == "reasoning_item":
            text = f"Reasoning: {str(item.raw_item)}"
        return text

    @classmethod
    def clear_chat_history(cls) -> None:
        st.session_state.messages = [cls.DEFAULT_MESSAGE]
        st.session_state.trace_id = None


if __name__ == "__main__":
    ChatPage.main()
