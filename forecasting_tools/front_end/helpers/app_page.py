from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

import streamlit as st
from streamlit.navigation.page import StreamlitPage

logger = logging.getLogger(__name__)


class AppPage(ABC):
    PAGE_DISPLAY_NAME: str = NotImplemented
    URL_PATH: str = NotImplemented
    IS_DEFAULT_PAGE: bool = False
    ENABLE_FOOTER: bool = True
    ENABLE_HEADER: bool = True

    def __init_subclass__(cls: type[AppPage], *args, **kwargs) -> None:
        super().__init_subclass__(*args, **kwargs)
        is_abstract = ABC in cls.__bases__
        if not is_abstract:
            if cls.PAGE_DISPLAY_NAME is NotImplemented:
                raise NotImplementedError(
                    "You forgot to define PAGE_DISPLAY_NAME"
                )
            if cls.URL_PATH is NotImplemented:
                raise NotImplementedError("You forgot to define URL_PATH")

    @classmethod
    def main(cls) -> None:
        if cls.ENABLE_HEADER:
            cls.header()
        asyncio.run(cls._async_main())
        if cls.ENABLE_FOOTER:
            cls.footer()

    @classmethod
    @abstractmethod
    async def _async_main(cls) -> None:
        pass

    @classmethod
    def convert_to_streamlit_page(cls) -> StreamlitPage:
        page = st.Page(
            cls.main,
            title=cls.PAGE_DISPLAY_NAME,
            icon=None,
            url_path=cls.URL_PATH,
            default=cls.IS_DEFAULT_PAGE,
        )
        return page

    @classmethod
    def header(cls):
        pass

    @classmethod
    def footer(cls):
        st.markdown("---")
        st.write(
            "This is the demo site for the "
            "[forecasting-tools python package](https://github.com/CodexVeritas/forecasting-tools)."
        )
        st.write(
            "Give feedback on the [Forecasting Tools Discord](https://discord.gg/Dtq4JNdXnw) or email "
            "me at ben [at] metaculus [dot com]. "
            "Let me know what I can do to make this a tool you will want to use "
            "every day! Let me know if you want to chat and we can find a time!"
        )
        st.write(
            "Queries made to the website are saved to a database and may be "
            "reviewed to help improve the tool"
        )
