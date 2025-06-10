import asyncio
import logging
import os

from hyperbrowser import Hyperbrowser
from hyperbrowser.models import (
    CreateSessionParams,
    CuaTaskData,
    StartCuaTaskParams,
)
from pydantic import BaseModel

from forecasting_tools.agents_and_tools.hosted_file import HostedFile
from forecasting_tools.ai_models.agent_wrappers import agent_tool
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents

logger = logging.getLogger(__name__)


class ComputerUseResponse(BaseModel):
    hyperbrowser_task_data: CuaTaskData
    hosted_files: list[HostedFile]
    live_url: str | None
    downloads_url: str | None
    final_answer: str
    hyperbrowser_session_id: str


class ComputerUse:

    def __init__(self) -> None:
        self.hb_client = Hyperbrowser(
            api_key=os.getenv("HYPERBROWSER_API_KEY")
        )

    async def answer_prompt(self, prompt: str) -> ComputerUseResponse:
        session = self.hb_client.sessions.create(
            CreateSessionParams(save_downloads=True)
        )
        session_id = session.id
        logger.info(f"Hyperbrowser Session ID: {session_id}")

        instructions = clean_indents(
            f"""
            You are a browser use agent helping with a user prompt. Please help the user with their request while keeping the following

            Rules:
            - If the user asks you to download something in their instructions, you should download the file (do not stop halfway to ask if they are sure they want to)
            - Do not stop halfway to ask any questions. Go all the way to the end of the task, unless you find it is impossible (in which case say so and then stop).
            - If you are asked to download something, and you successfully click the download button, say that you successfully downloaded the file and describe the screen you were last on when you finished (and detailed descriptions of any graphs/tables/filters that were on the screen)

            User Request:
            {prompt}
            """
        )

        resp = self.hb_client.agents.cua.start_and_wait(
            StartCuaTaskParams(task=instructions, session_id=session_id)
        )
        live_url = resp.live_url
        logger.info(f"Hyperbrowser Live URL: {live_url}")
        data = resp.data
        if data is None:
            raise RuntimeError("No response from Hyperbrowser")
        final_result = data.final_result
        if final_result is None:
            raise RuntimeError("No response from Hyperbrowser")
        downloads_response = self.hb_client.sessions.get_downloads_url(
            session.id
        )
        while downloads_response.status == "in_progress":
            logger.info("Waiting for downloads zip to be ready...")
            await asyncio.sleep(1)
            downloads_response = self.hb_client.sessions.get_downloads_url(
                session.id
            )

        download_url = downloads_response.downloads_url
        logger.info(f"Hyperbrowser Downloads URL: {download_url}")
        self.hb_client.sessions.stop(session.id)

        if download_url:
            hosted_files = HostedFile.upload_zipped_files(download_url)
        else:
            hosted_files = []

        return ComputerUseResponse(
            hyperbrowser_task_data=data,
            final_answer=final_result,
            hosted_files=hosted_files,
            live_url=live_url,
            downloads_url=download_url,
            hyperbrowser_session_id=session_id,
        )

    @agent_tool
    @staticmethod
    def computer_use_tool(prompt: str) -> str:
        """
        This tool has access to a browser and is specialized for hard to navigate internet tasks.
        Don't use this tool for simple searches, but instead to do things like:
        1. Naviagate to a site and download a file
        2. Examine things on sites that need to be viewed visually
        3. See what is on a specific page
        4. etc.

        Include any relevant URLs in the prompt and try to give a detailed plan of what you want the agent to do.
        """
        computer_use = ComputerUse()
        response = asyncio.run(computer_use.answer_prompt(prompt))

        text_log = "# Steps Taken"
        for i, step in enumerate(response.hyperbrowser_task_data.steps):
            action_text = ""
            if step.output:
                for output in step.output:
                    if "name" in output:
                        action_text += f"**Name**: {output['name']}\n"
                    if "action" in output:
                        action_text += f"**Action**: {output['action']}\n"
                    if "summary" in output:
                        action_text += f"**Summary**: {output['summary']}\n"
            if step.output_text:
                action_text += f"**Output Text**: {step.output_text}\n"

            text_log += clean_indents(
                f"""
                ## Step {i+1}
                {action_text}
                """
            )
        text_log += f"\n---\n# Final Answer\n{response.final_answer}"
        if response.downloads_url:
            text_log += f"\n- **Downloads URL:** {response.downloads_url}"
        for file in response.hosted_files:
            text_log += f"\n- **Downloaded File:** Name: {file.file_name} | OpenAI File ID: {file.file_id}"
        return text_log
