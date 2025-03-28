import asyncio
from typing import Callable

import streamlit as st


class CustomAuth:
    DEFAULT_PASSPHRASE = "debias"

    @staticmethod
    def add_access_control(
        allowed_passphrases: list[str] = [DEFAULT_PASSPHRASE],
    ) -> Callable:
        def decorator(func: Callable) -> Callable:
            if not asyncio.iscoroutinefunction(func):
                raise TypeError("Function must be a coroutine.")

            async def wrapper(*args, **kwargs) -> None:
                st.sidebar.title("Access Control")
                entered_passphrase = st.sidebar.text_input(
                    "Enter passphrase", type="password"
                )
                if entered_passphrase in allowed_passphrases:
                    await func(*args, **kwargs)
                else:
                    st.warning(
                        "Passphrase required to access this feature. See side bar."
                    )
                    st.sidebar.warning(
                        "Incorrect passphrase. Please try again."
                    )

            return wrapper

        return decorator
