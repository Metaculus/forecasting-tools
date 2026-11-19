import asyncio
import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_coroutine_synchronously(coroutine: Coroutine[Any, Any, T]) -> T:
    """
    Runs a coroutine to completion from sync code. If this thread already has a running
    event loop (e.g. Jupyter, or an async function calling a sync wrapper), the coroutine
    runs on a new loop in a worker thread, and the running loop is blocked until it finishes.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    logger.warning(
        f"Running '{coroutine.__qualname__}' in a worker thread because an event loop "
        "is already running in this thread. The running loop is blocked until it "
        "finishes. Await the async version of this function instead if there is one."
    )
    caller_context = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(caller_context.run, asyncio.run, coroutine).result()
