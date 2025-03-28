from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import streamlit as st
from pydantic import BaseModel

from forecasting_tools.front_end.helpers.app_page import AppPage
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class Example(BaseModel, Jsonable):
    short_name: str | None = None
    notes: str | None = None
    input: dict[str, Any]
    output: dict[str, Any]


class ToolPage(AppPage, ABC):
    OUTPUT_TYPE: type[Jsonable] = Jsonable
    INPUT_TYPE: type[Jsonable] = Jsonable
    EXAMPLES_FILE_PATH: str | None = NotImplemented

    def __init_subclass__(cls: type[ToolPage], *args, **kwargs) -> None:
        super().__init_subclass__(*args, **kwargs)
        is_abstract = ABC in cls.__bases__
        if not is_abstract:
            if cls.OUTPUT_TYPE is Jsonable or not issubclass(
                cls.OUTPUT_TYPE, Jsonable
            ):
                raise NotImplementedError(
                    "You forgot to define OUTPUT_TYPE as a subclass of Jsonable"
                )
            if cls.INPUT_TYPE is Jsonable or not issubclass(
                cls.INPUT_TYPE, Jsonable
            ):
                raise NotImplementedError(
                    "You forgot to define INPUT_TYPE as a subclass of Jsonable"
                )
            if cls.EXAMPLES_FILE_PATH is NotImplemented:
                raise NotImplementedError(
                    "You forgot to define EXAMPLES_FILE_PATH"
                )

    @classmethod
    async def _async_main(cls) -> None:
        st.title(cls.PAGE_DISPLAY_NAME)
        await cls._display_intro_text()
        await cls._display_example_buttons_expander()
        input_to_tool = await cls._get_input()
        if input_to_tool:
            assert isinstance(input_to_tool, cls.INPUT_TYPE)
            output = await cls._run_tool(input_to_tool)
            assert isinstance(output, cls.OUTPUT_TYPE)
            await cls._save_run(input_to_tool, output)
        outputs = await cls._get_saved_outputs()
        if outputs:
            await cls._display_outputs(outputs)

    @classmethod
    @abstractmethod
    async def _display_intro_text(cls) -> None:
        pass

    @classmethod
    async def _display_example_buttons_expander(cls) -> None:
        examples = await cls._get_examples()
        if examples:
            with st.expander("📋 Premade Examples", expanded=False):
                cols = st.columns(len(examples))
                for index, example in enumerate(examples):
                    with cols[index]:
                        await cls._display_single_example_button(
                            example, index
                        )

    @classmethod
    async def _display_single_example_button(
        cls, example: Example, example_number: int
    ) -> None:
        button_label = f"Show Example {example_number + 1}"
        if example.short_name:
            button_label += f": {example.short_name}"
        example_clicked = st.button(button_label, use_container_width=True)
        if example.notes:
            st.markdown(
                f"<div style='text-align: center'>{example.notes}</div>",
                unsafe_allow_html=True,
            )
        if example_clicked:
            input_to_tool = cls.INPUT_TYPE.from_json(example.input)
            output = cls.OUTPUT_TYPE.from_json(example.output)
            await cls._save_run(input_to_tool, output, is_premade_example=True)

    @classmethod
    async def _get_examples(cls) -> list[Example]:
        if cls.EXAMPLES_FILE_PATH is None:
            return []
        examples = Example.load_json_from_file_path(cls.EXAMPLES_FILE_PATH)
        return examples

    @classmethod
    @abstractmethod
    async def _get_input(cls) -> Any:
        pass

    @classmethod
    @abstractmethod
    async def _run_tool(cls, input: Any) -> Jsonable:
        pass

    @classmethod
    async def _save_run(
        cls,
        input_to_tool: Jsonable,
        output: Jsonable,
        is_premade_example: bool = False,
    ) -> None:
        assert isinstance(output, cls.OUTPUT_TYPE)
        await cls._save_output_to_session_state(output)

        if not is_premade_example:
            try:
                await cls._save_run_to_file(input_to_tool, output)
            except Exception as e:
                logger.error(f"Error saving output to file: {e}")

        # Allow manipulation of the output when prepping to save it to database without affecting the original
        assert isinstance(input_to_tool, BaseModel)
        assert isinstance(output, BaseModel)
        input_to_tool = input_to_tool.model_copy(deep=True)
        output = output.model_copy(deep=True)

        try:
            await cls._save_run_to_coda(
                input_to_tool, output, is_premade_example
            )
        except Exception as e:
            logger.error(f"Error saving output to Coda: {e}")

    @classmethod
    @abstractmethod
    async def _save_run_to_coda(
        cls,
        input_to_tool: Jsonable,
        output: Jsonable,
        is_premade_example: bool,
    ) -> None:
        pass

    @classmethod
    async def _save_output_to_session_state(cls, output: Jsonable) -> None:
        assert isinstance(output, cls.OUTPUT_TYPE)
        session_state_key = cls.__get_saved_outputs_key()
        if session_state_key not in st.session_state:
            st.session_state[session_state_key] = []
        st.session_state[session_state_key].insert(0, output)

    @classmethod
    async def _save_run_to_file(
        cls, input_to_tool: Jsonable, output: Jsonable
    ) -> None:
        assert isinstance(output, cls.OUTPUT_TYPE)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = f"logs/forecasts/streamlit/{timestamp}_{cls.__name__}.json"
        Example.save_object_list_to_file_path(
            [Example(input=input_to_tool.to_json(), output=output.to_json())],
            file_path,
        )

    @classmethod
    async def _get_saved_outputs(cls) -> list[Jsonable]:
        session_state_key = cls.__get_saved_outputs_key()
        if session_state_key not in st.session_state:
            st.session_state[session_state_key] = []
        outputs = st.session_state[session_state_key]
        validated_outputs = [
            cls.OUTPUT_TYPE.from_json(output.to_json()) for output in outputs
        ]
        return validated_outputs

    @classmethod
    def __get_saved_outputs_key(cls) -> str:
        return f"{cls.__name__}_saved_outputs"

    @classmethod
    @abstractmethod
    async def _display_outputs(cls, outputs: list[Jsonable]) -> None:
        pass
