from abc import ABC

from agents import Agent, AgentOutputSchema, Tool
from agents.extensions.models.litellm_model import LitellmModel

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents


class BaseAgentTool(ABC):
    """
    Flexible agent <-> tool conversion class.
    Override either agent or Tool with your needed agent or tool,
    and then use the other fields to convert from filled in property.
    """

    @property
    def agent(self) -> Agent:
        raise RuntimeError(
            "`agent` is not defined. Please call `readable_agent` if you want to use a tool like an agent."
        )

    @property
    def tool(self) -> Tool:
        return self.agent.as_tool(
            tool_name=None,
            tool_description=self.agent.handoff_description,
        )

    @property
    def readable_agent(self) -> Agent:
        readable_agent = Agent(
            name=f"{self.__class__.__name__} Markdownifier",
            instructions=clean_indents(
                f"""
                You are the secretary of a {self.agent.name} and take json outputthey give and turn it into readable markdown.
                You include ALL the information they give, with no additional information.
                However you make it readable and easy-to-understand Markdown
                """
            ),
            tools=[self.tool],
            model=LitellmModel(model="gpt-4o-mini"),
            output_type=AgentOutputSchema(str),
        )
        return readable_agent
