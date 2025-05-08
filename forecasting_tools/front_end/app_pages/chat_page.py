import logging

import streamlit as st
from agents import Agent, Handoff, RunItem, Runner, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from openai.types.responses import ResponseTextDeltaEvent

from forecasting_tools.front_end.helpers.app_page import AppPage
from forecasting_tools.front_end.helpers.tools import get_tools_for_chat_app

logger = logging.getLogger(__name__)


class ChatPage(AppPage):
    PAGE_DISPLAY_NAME: str = "💬 Chatbot"
    URL_PATH: str = "/chat"
    ENABLE_HEADER: bool = False
    ENABLE_FOOTER: bool = False
    DEFAULT_MESSAGE: dict = {
        "role": "assistant",
        "content": "How may I assist you today?",
    }
    MODEL_NAME: str = "openrouter/google/gemini-2.5-pro-preview"

    @classmethod
    async def _async_main(cls) -> None:
        set_tracing_disabled(disabled=True)

        # Store LLM generated responses
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [cls.DEFAULT_MESSAGE]

        st.sidebar.button(
            "Clear Chat History", on_click=cls.clear_chat_history
        )
        cls.display_messages(st.session_state.messages)

        # User-provided prompt
        if prompt := st.chat_input():
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )
            with st.chat_message("user"):
                st.write(prompt)

        # Generate a new response if last message is not from assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            new_messages = await cls.generate_response(prompt)
            st.session_state.messages.extend(new_messages)
            st.rerun()

    @classmethod
    def display_messages(cls, messages: list[dict]) -> None:
        for i, message in enumerate(messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])

            if "reasoning" in message:
                reasoning: str = message["reasoning"].strip()
                if reasoning:
                    with st.sidebar:
                        with st.expander(f"Tool Calls Message {i//2}"):
                            st.write(reasoning)

    @classmethod
    async def generate_response(cls, prompt_input: str | None) -> list[dict]:
        if prompt_input is None:
            return [
                {
                    "role": "assistant",
                    "content": "You didn't enter any message",
                }
            ]
        chat_history_as_string = ""
        for dict_message in st.session_state.messages:
            if dict_message["role"] == "user":
                chat_history_as_string += (
                    "User: " + dict_message["content"] + "\n\n"
                )
            else:
                chat_history_as_string += (
                    "Assistant: " + dict_message["content"] + "\n\n"
                )
        agent = Agent(
            name="Assistant",
            instructions="You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'.",
            model=LitellmModel(model=cls.MODEL_NAME),
            tools=get_tools_for_chat_app(),
            handoffs=cls._all_handoffs(),
        )

        result = Runner.run_streamed(
            agent, f"{chat_history_as_string} {prompt_input} \n\n Assistant: "
        )
        streamed_text = ""
        reasoning_text = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Thinking..."):
                async for event in result.stream_events():
                    if event.type == "raw_response_event" and isinstance(
                        event.data, ResponseTextDeltaEvent
                    ):
                        streamed_text += event.data.delta
                    elif event.type == "run_item_stream_event":
                        reasoning_text += (
                            f"{cls._grab_text_of_item(event.item)}\n\n"
                        )
                    placeholder.write(streamed_text)
        logger.info(f"Chat finished with output: {streamed_text}")
        new_messages = [
            {
                "role": "assistant",
                "content": streamed_text,
                "reasoning": reasoning_text,
            }
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
        return text

    @classmethod
    def clear_chat_history(cls) -> None:
        st.session_state.messages = [cls.DEFAULT_MESSAGE]

    @classmethod
    def _all_handoffs(cls) -> list[Agent | Handoff]:
        debate_agent = Agent(
            name="Debate Agent",
            instructions="You are a debate agent. You act as a debate partner.",
            model=LitellmModel(model="gemini/gemini-2.5-pro-preview-03-25"),
        )
        return [debate_agent]


if __name__ == "__main__":
    ChatPage.main()
