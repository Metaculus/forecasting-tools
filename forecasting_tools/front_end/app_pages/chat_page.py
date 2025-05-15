import logging

import streamlit as st
from agents import Agent, RunItem, Runner, Tool
from openai.types.responses import ResponseTextDeltaEvent

from forecasting_tools.agents_and_tools.misc_tools import (
    create_tool_for_forecasting_bot,
    get_general_news_with_asknews,
    grab_open_questions_from_tournament,
    grab_question_details_from_metaculus,
    perplexity_search,
    smart_searcher_search,
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
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import AgentSdkLlm
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

logger = logging.getLogger(__name__)


class ChatMessage:
    def __init__(self, role: str, content: str, tool_output: str = "") -> None:
        self.role = role
        self.content = content
        self.tool_output = tool_output

    def to_open_ai_message(self) -> dict:
        return {"role": self.role, "content": self.content}


class ChatPage(AppPage):
    PAGE_DISPLAY_NAME: str = "💬 Chatbot"
    URL_PATH: str = "/chat"
    ENABLE_HEADER: bool = False
    ENABLE_FOOTER: bool = False
    DEFAULT_MESSAGE: ChatMessage = ChatMessage(
        role="assistant",
        content="How may I assist you today?",
        tool_output="",
    )

    @classmethod
    async def _async_main(cls) -> None:
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [cls.DEFAULT_MESSAGE]

        st.sidebar.button(
            "Clear Chat History", on_click=cls.clear_chat_history
        )
        if "last_chat_cost" not in st.session_state.keys():
            st.session_state.last_chat_cost = 0
        if st.session_state.last_chat_cost > 0:
            st.sidebar.markdown(
                f"**Last Chat Cost:** ${st.session_state.last_chat_cost:.7f}"
            )
        model_choice = cls.display_model_selector()
        active_tools = cls.display_tools()
        cls.display_messages(st.session_state.messages)

        if prompt := st.chat_input():
            st.session_state.messages.append(
                ChatMessage(role="user", content=prompt)
            )
            with st.chat_message("user"):
                st.write(prompt)

        if st.session_state.messages[-1].role != "assistant":
            with MonetaryCostManager(10) as cost_manager:
                new_messages = await cls.generate_response(
                    prompt, active_tools, model_choice
                )
                st.session_state.last_chat_cost = cost_manager.current_usage
            st.session_state.messages.extend(new_messages)
            st.rerun()

    @classmethod
    def display_model_selector(cls) -> str:
        model_choice = st.sidebar.text_input(
            "Select a model",
            value="openrouter/google/gemini-2.5-pro-preview",  # "gemini/gemini-2.5-pro-preview-03-25"
        )
        if "o1-pro" in model_choice or "gpt-4.5" in model_choice:
            raise ValueError(
                "o1 pro and gpt-4.5 are not available for this application."
            )
        return model_choice

    @classmethod
    def display_tools(cls) -> list[Tool]:
        default_tools: list[Tool] = [
            TopicGenerator().find_random_headlines_tool,
            QuestionDecomposer().decompose_into_questions_tool,
            QuestionOperationalizer().question_operationalizer_tool,
            perplexity_search,
            get_general_news_with_asknews,
            smart_searcher_search,
            grab_question_details_from_metaculus,
            grab_open_questions_from_tournament,
            TopicGenerator().get_headlines_on_random_company_tool,
        ]

        bot_options = get_all_important_bot_classes()
        bot_choice = st.sidebar.selectbox(
            "Select a bot for forecast_question_tool",
            [bot.__name__ for bot in bot_options],
        )
        bot = next(bot for bot in bot_options if bot.__name__ == bot_choice)
        default_tools.append(create_tool_for_forecasting_bot(bot))

        active_tools: list[Tool] = []
        with st.sidebar:
            for tool in default_tools:
                tool_active = st.checkbox(tool.name, value=True)
                if tool_active:
                    active_tools.append(tool)

        return active_tools

    @classmethod
    def display_messages(cls, messages: list[ChatMessage]) -> None:
        for i, message in enumerate(messages):
            with st.chat_message(message.role):
                st.write(message.content)

            if message.tool_output.strip():
                with st.sidebar:
                    with st.expander(f"Tool Calls Message {i//2}"):
                        st.write(message.tool_output)

    @classmethod
    async def generate_response(
        cls,
        prompt_input: str | None,
        active_tools: list[Tool],
        model_choice: str,
    ) -> list[ChatMessage]:
        if prompt_input is None:
            return [
                ChatMessage(
                    role="assistant",
                    content="You didn't enter any message",
                )
            ]

        instructions = clean_indents(
            """
            You are a helpful assistant.
            When a tool gives you answers that are cited, ALWAYS include the links in your responses.

            If you can, you infer the inputs to tools rather than ask for them.

            If a tool call fails, you say so rather than giving a back up answer.

            Whenever possible, please parralelize your tool calls.
            """
        )

        agent = Agent(
            name="Assistant",
            instructions=instructions,
            model=AgentSdkLlm(model=model_choice),
            tools=active_tools,
            handoffs=[],
        )

        openai_messages = [
            m.to_open_ai_message() for m in st.session_state.messages
        ]
        result = Runner.run_streamed(agent, openai_messages, max_turns=20)
        streamed_text = ""
        reasoning_text = ""
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
                    new_reasoning = f"{cls._grab_text_of_item(event.item)}\n\n"
                    reasoning_text += new_reasoning  # TODO: Don't define this as reasoning, but as a new tool-role message
                # elif event.type == "agent_updated_stream_event":
                #     reasoning_text += f"Agent updated: {event.new_agent.name}\n\n"
                placeholder.write(streamed_text)
                if new_reasoning:
                    st.sidebar.write(new_reasoning)

        logger.info(f"Chat finished with output: {streamed_text}")
        new_messages = [
            ChatMessage(
                role="assistant",
                content=ReportDisplayer.clean_markdown(streamed_text),
                tool_output=ReportDisplayer.clean_markdown(reasoning_text),
            )
        ]
        return new_messages

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


if __name__ == "__main__":
    ChatPage.main()
