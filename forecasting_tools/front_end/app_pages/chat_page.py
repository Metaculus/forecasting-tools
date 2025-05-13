import logging

import streamlit as st
from agents import Agent, Handoff, RunItem, Runner, Tool
from agents.extensions.models.litellm_model import LitellmModel
from openai.types.responses import ResponseTextDeltaEvent

from forecasting_tools.agents_and_tools.question_generators.question_decomposer import (
    QuestionDecomposer,
)
from forecasting_tools.agents_and_tools.question_generators.question_operationalizer import (
    QuestionOperationalizer,
)
from forecasting_tools.agents_and_tools.question_generators.topic_generator import (
    TopicGenerator,
)
from forecasting_tools.agents_and_tools.research_tools import (
    get_general_news,
    perplexity_search,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.front_end.helpers.app_page import AppPage

logger = logging.getLogger(__name__)


class ChatMessage:
    def __init__(self, role: str, content: str, reasoning: str = "") -> None:
        self.role = role
        self.content = content
        self.reasoning = reasoning

    def to_open_ai_message(self) -> dict:
        return {"role": self.role, "content": self.content}

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        return cls(
            role=data.get("role", "assistant"),
            content=data.get("content", ""),
            reasoning=data.get("reasoning", ""),
        )


class ChatPage(AppPage):
    PAGE_DISPLAY_NAME: str = "💬 Chatbot"
    URL_PATH: str = "/chat"
    ENABLE_HEADER: bool = False
    ENABLE_FOOTER: bool = False
    DEFAULT_MESSAGE: ChatMessage = ChatMessage(
        role="assistant",
        content="How may I assist you today?",
        reasoning="",
    )
    MODEL_NAME: str = "openrouter/google/gemini-2.5-pro-preview"

    @classmethod
    async def _async_main(cls) -> None:
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [cls.DEFAULT_MESSAGE]

        st.sidebar.button(
            "Clear Chat History", on_click=cls.clear_chat_history
        )
        cls.display_messages(st.session_state.messages)

        if prompt := st.chat_input():
            st.session_state.messages.append(
                ChatMessage(role="user", content=prompt)
            )
            with st.chat_message("user"):
                st.write(prompt)

        if st.session_state.messages[-1].role != "assistant":
            new_messages = await cls.generate_response(prompt)
            st.session_state.messages.extend(new_messages)
            st.rerun()

    @classmethod
    def display_messages(cls, messages: list[ChatMessage]) -> None:
        for i, message in enumerate(messages):
            with st.chat_message(message.role):
                st.write(message.content)

            if message.reasoning.strip():
                with st.sidebar:
                    with st.expander(f"Tool Calls Message {i//2}"):
                        st.write(message.reasoning)

    @classmethod
    async def generate_response(
        cls, prompt_input: str | None
    ) -> list[ChatMessage]:
        if prompt_input is None:
            return [
                ChatMessage(
                    role="assistant",
                    content="You didn't enter any message",
                )
            ]

        agent = Agent(
            name="Assistant",
            instructions=clean_indents(
                """
                You are a helpful assistant.
                When a tool gives you answers that are cited,
                you include the links in your responses.

                If you can, you infer the inputs to tools rather than ask for them.

                If a tool call fails, you say so rather than giving a back up answer.
                """
            ),
            model=LitellmModel(model=cls.MODEL_NAME),
            tools=get_tools_for_chat_app(),
            handoffs=get_agents_for_chat_app(),
        )

        openai_messages = [
            m.to_open_ai_message() for m in st.session_state.messages
        ]
        result = Runner.run_streamed(agent, openai_messages)
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
                        reasoning_text += f"{cls._grab_text_of_item(event.item)}\n\n"  # TODO: Don't define this as reasoning, but as a new tool-role message
                    # elif event.type == "agent_updated_stream_event":
                    #     reasoning_text += f"Agent updated: {event.new_agent.name}\n\n"
                    placeholder.write(streamed_text)
                    st.sidebar.write(reasoning_text)
        logger.info(f"Chat finished with output: {streamed_text}")
        new_messages = [
            ChatMessage(
                role="assistant",
                content=streamed_text,
                reasoning=reasoning_text,
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


def get_agents_for_chat_app() -> list[Agent | Handoff]:
    return []


def get_tools_for_chat_app() -> list[Tool]:
    return [
        QuestionDecomposer().decompose_into_questions_tool,
        TopicGenerator().generate_random_topics_tool,
        QuestionOperationalizer().question_title_to_simple_question_tool,
        perplexity_search,
        get_general_news,
    ]


if __name__ == "__main__":
    ChatPage.main()
