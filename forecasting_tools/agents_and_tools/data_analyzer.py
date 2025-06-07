import asyncio

from openai import OpenAI

from forecasting_tools.agents_and_tools.misc_tools import (
    perplexity_quick_search,
)
from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AiAgent,
    CodingTool,
    agent_tool,
)


class DataAnalyzer:

    def __init__(self) -> None:
        pass

    async def run_data_analysis(self, instructions: str) -> str:
        # client = OpenAI()

        # resp = client.responses.create(
        #     model="gpt-4.1",
        #     tools=[
        #         {
        #             "type": "code_interpreter",
        #             "container": { "type": "auto" }
        #         }
        #     ],
        #     instructions="You are a personal math tutor. When asked a math question, write and run code to answer the question.",
        #     input="I need to solve the equation 3x + 11 = 14. Can you help me?",
        # )

        # return resp.output_text

        client = OpenAI()
        file = client.files.create(
            file=open("temp/bot_questions.csv", "rb"), purpose="assistants"
        )

        agent = AiAgent(
            name="Data Analyzer",
            instructions=instructions,
            model="gpt-4o",
            tools=[
                perplexity_quick_search,
                CodingTool(
                    tool_config={
                        "type": "code_interpreter",
                        "container": {
                            "type": "auto",
                            "file_ids": [file.id],
                        },
                    }
                ),
            ],
            handoffs=[],
        )
        result = await AgentRunner.run(
            agent, "Please follow your custom instructions.", max_turns=10
        )
        return result.final_output

        # client = openai.OpenAI()

        # file = client.files.create(
        #     file=open("temp/bot_questions.csv", "rb"),
        #     purpose='assistants'
        # )

        # # Step 1: Create an Assistant
        # assistant = client.beta.assistants.create(
        #     name="Data Analyst Assistant",
        #     instructions="You are a personal Data Analyst Assistant",
        #     model="gpt-4o",
        #     tools=[{"type": "code_interpreter"}],
        #     file_ids=[file.id]
        # )

        # # Step 2: Create a Thread
        # thread = client.beta.threads.create()

        # # Step 3: Add a Message to a Thread
        # message = client.beta.threads.messages.create(
        #     thread_id=thread.id,
        #     role="user",
        #     content=instructions
        # )

        # # Step 4: Run the Assistant
        # run = client.beta.threads.runs.create(
        #     thread_id=thread.id,
        #     assistant_id=assistant.id,
        #     instructions=instructions
        # )

        # answer = run.model_dump_json(indent=4)

        # while True:
        #     # Wait for 5 seconds
        #     time.sleep(5)

        #     # Retrieve the run status
        #     run_status = client.beta.threads.runs.retrieve(
        #         thread_id=thread.id,
        #         run_id=run.id
        #     )
        #     answer += run_status.model_dump_json(indent=4)

        #     # If run is completed, get messages
        #     if run_status.status == 'completed':
        #         messages = client.beta.threads.messages.list(
        #             thread_id=thread.id
        #         )

        #         # Loop through messages and print content based on role
        #         for msg in messages.data:
        #             role = msg.role
        #             content = msg.content[0].text.value
        #             answer += f"{role.capitalize()}: {content}"
        #         break
        #     else:
        #         answer += "Waiting for the Assistant to process..."
        #         time.sleep(5)
        # return answer

    @agent_tool
    @staticmethod
    def data_analysis_tool(instruction: str) -> str:
        """
        This tool takes in instructions, and runs code to follow the instructions.
        """
        data_analysis = DataAnalyzer()
        return asyncio.run(data_analysis.run_data_analysis(instruction))
