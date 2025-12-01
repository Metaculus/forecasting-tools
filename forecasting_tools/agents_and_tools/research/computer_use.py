import asyncio
import logging
import os

from hyperbrowser import AsyncHyperbrowser
from hyperbrowser.models import CreateSessionParams, CuaTaskData, StartCuaTaskParams
from pydantic import BaseModel

from forecasting_tools.agents_and_tools.other.hosted_file import HostedFile
from forecasting_tools.ai_models.agent_wrappers import agent_tool
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)


class ComputerUseResult(BaseModel):
    hyperbrowser_task_data: CuaTaskData
    hosted_files: list[HostedFile]
    downloads_url: str | None
    final_answer: str
    hyperbrowser_session_id: str
    recording_url: str | None

    @property
    def as_string(self) -> str:
        text_log = "# Computer Use Agent Steps"
        for i, step in enumerate(self.hyperbrowser_task_data.steps):
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
        text_log += f"\n\n# Final Answer\n{self.final_answer}"
        if self.downloads_url:
            text_log += f"\n- **Downloads URL:** [url]({self.downloads_url})"
        for file in self.hosted_files:
            text_log += f"\n- **Downloaded File:** Name: {file.file_name} | OpenAI File ID: {file.file_id}"
        if self.hyperbrowser_session_id:
            text_log += f"\n- **Session ID:** {self.hyperbrowser_session_id}"
        if self.recording_url:
            text_log += f"\n- **Recording URL:** [url]({self.recording_url})"
        return text_log


class ComputerUse:

    def __init__(self) -> None:
        api_key = os.getenv("HYPERBROWSER_API_KEY")
        if not api_key:
            raise ValueError("HYPERBROWSER_API_KEY is not set")
        self.hb_client = AsyncHyperbrowser(api_key=api_key)
        logger.warning("Cost tracking not supported for ComputerUse currently")

    async def answer_prompt(self, prompt: str) -> ComputerUseResult:
        session = await self.hb_client.sessions.create(
            CreateSessionParams(save_downloads=True, enable_web_recording=True)
        )
        session_id = session.id
        logger.info(f"Hyperbrowser Session ID: {session_id}")

        instructions = clean_indents(
            f"""
            You are a browser use agent helping with a user prompt. Please help the user with their request while following the rules below.

            # Rules:
            - If Downloading:
                - If the user asks you to download something in their instructions, you should download the file (do not stop halfway to ask if they are sure they want to)
                - If you are asked to download something, and you successfully click the download button, say that you successfully downloaded the file and describe the screen you were last on when you finished (and detailed descriptions of any graphs/tables/filters that were on the screen)
                - Any files you download will be returned as urls and Hosted file IDs automatically (you don't need to do anything additional other than click the download button). If you are asked to return files in a specified format, just say that you will return these (i.e. you don't support other formats) and finish your task.
                - When in doubt, if you clicked the download button assume it worked
                - If you ever see that the down arrow next to the profile button is blue in the top bar of chrome, this means you successfully downloaded a file.
                - If you can't download what is being looked for (and will give up), and there is a table/graph you can analyze instead, please try to answer the question with visual inspection. Please state that you are only doing a visual inspection.
            - If visually inspecting something:
                - Prioritize finishing the task asked of you and following any instructions given to you
                - By default give a lot of detail in your answer. Describe everything visually you see on the screen related to your answer.
                - If relevant please give exact quotations of the words on the screen if it helps answer the question
            - If Stuck:
                - Do not stop halfway to ask any questions. Go all the way to the end of the task, unless you find it is impossible (in which case say so and then stop).
                - If you get to the point where you are confused and waiting for a while. Just give up and describe what you did and where you ended.
                - Never ask follow up questions (you won't get any answers). If you are stuck, just say you give up and why (rather than phrasing it as a question).
            - As much as possible verbalize what you are doing at each step you take.

            # User Request:
            {prompt}
            """
        )

        resp = await self.hb_client.agents.cua.start_and_wait(
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
        downloads_response = await self.hb_client.sessions.get_downloads_url(session.id)
        while downloads_response.status == "in_progress":
            logger.info("Waiting for downloads zip to be ready...")
            await asyncio.sleep(1)
            downloads_response = await self.hb_client.sessions.get_downloads_url(
                session.id
            )

        download_url = downloads_response.downloads_url
        logger.info(f"Hyperbrowser Downloads URL: {download_url}")
        await self.hb_client.sessions.stop(session.id)

        if download_url:
            hosted_files = HostedFile.upload_zipped_files(download_url)
        else:
            hosted_files = []

        recording_data = await self.hb_client.sessions.get_recording_url(session.id)
        recording_url = recording_data.recording_url
        logger.info(f"Hyperbrowser Recording URL: {recording_url}")

        return ComputerUseResult(
            hyperbrowser_task_data=data,
            final_answer=final_result,
            hosted_files=hosted_files,
            downloads_url=download_url,
            hyperbrowser_session_id=session_id,
            recording_url=recording_url,
        )

    @agent_tool
    @staticmethod
    def computer_use_tool(prompt: str) -> str:
        """
        This tool has access to a browser and is specialized for hard to navigate internet tasks.
        Don't use this tool for simple searches, but instead to do things like:
        1. Navigate to a site and download a file
        2. Examine things on sites that need to be viewed visually
        3. See what is on a specific page
        4. etc.

        Include any relevant URLs in the prompt and try to give a detailed plan of what you want the agent to do. Links and Hosted file IDs of downloaded data will be returned automatically with the answer (so don't ask for file paths). This tool can take 10+ minutes to run.
        """
        computer_use = ComputerUse()
        result = asyncio.run(computer_use.answer_prompt(prompt))
        return result.as_string
