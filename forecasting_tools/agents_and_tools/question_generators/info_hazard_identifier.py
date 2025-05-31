import asyncio
from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from forecasting_tools.ai_models.agent_wrappers import agent_tool
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.forecast_helpers.structure_output import (
    structure_output,
)


class HazardRating(Enum):
    NO = "No"
    KINDA_YES = "Kinda Yes"
    YES = "Yes"
    STRONGLY_YES = "Strongly Yes"

    @property
    def passes_threshold(self) -> bool:
        return self in [HazardRating.YES, HazardRating.STRONGLY_YES]


class InfoHazardIdentification(BaseModel):
    is_harmful: HazardRating
    self_fulfilling_or_defeating_for_individual: HazardRating
    self_fulfilling_or_defeating_for_society: HazardRating
    reasoning: str

    @property
    def is_hazard(self) -> bool:
        if (
            self.is_harmful.passes_threshold
            or self.self_fulfilling_or_defeating_for_individual.passes_threshold
            or self.self_fulfilling_or_defeating_for_society.passes_threshold
        ):
            return True
        return False


class InfoHazardIdentifier:
    def __init__(
        self,
        model: str | GeneralLlm = "openrouter/perplexity/sonar-reasoning-pro",
    ) -> None:
        self.model: GeneralLlm = GeneralLlm.to_llm(model)

    async def identify_info_hazard(
        self,
        question_text: str,
        other_question_info: str | None = None,
        additional_context: str | None = None,
    ) -> InfoHazardIdentification:
        input_prompt = clean_indents(
            f"""
            # Instructions
            You are an info hazard identifier for forecasting questions and whether they should be posted on platforms like Metaculus, Polymarket, or similar. Your job is to analyze the provided question and context, and determine if there are any information hazards present and explain why or why not.

            You will check 3 different types of info hazards:
            1. Dangerous information
            2. Self-fulfilling/defeating information on the individual level
            3. Self-fulfilling/defeating information on the society level

            For each you will identify if the hazard is:
            - No
            - Kinda Yes
            - Yes
            - Strongly Yes

            The vast majority of the questions will fall into the No or Yes category. Save Strongly Yes for rare cases.

            ## What is an info hazard?
            Information hazards are risks posed by the spread of factual information that could lead to harmful outcomes if they reach certain actors or audiences. These range from technical vulnerabilities that could compromise systems, to extreme cases such as information about how to build a bioweapon. Questions should also seek to avoid harmful self-fulfilling or self-negating effects, where the aggregate forecast could itself influence whether the event in question occurs. Information hazards and circular effects are often difficult to assess and can vary substantially in the level of risk posed.

            ## Steps
            1. Forecast the question by giving the following information:
                - The time left until the outcome to the question is known.
                - The status quo outcome if nothing changed.
                - A brief description of a scenario that results in a No outcome.
                - A brief description of a scenario that results in a Yes outcome.
                - You reasoning and a final answer
            2. Check if question is harmful
                - If the user is given the answer to the question, could they use that information to harm themselves or others?
                - If either positive or negative answer to this question becomes widely known, would it cause a net negative effect?
                - If just knowing the answer could harm people in some way?
                - If so, then there is a harmful information hazard.
            3. Check if question has self-fulfilling/defeating information on the individual level
                - Ask "If the user (or his community) is told that the probability of event X is low, what will be the probability of event X?"
                - Ask "If the user (or his community) is told that the probability of event X is high, what will be the probability of event X?"
                - If there is a significant difference between the two answers, then there is a self-fulfilling information hazard on the individual level
            5. Check if question has self-fulfilling/defeating information on the society level
                - Ask "If everyone in the world is told that the probability of event X is low, what will be the probability of event X?"
                - Ask "If everyone in the world is told that the probability of event X is high, what will be the probability of event X?"
                - If there is a significant difference between the two answers, then there is a self-fulfilling information hazard on the society level

            Also remember that today is {datetime.now().strftime("%Y-%m-%d")}.

            ## Examples

            ### Dangerous information
            - "If I tried, could I synthesize a lethal toxin using common household chemicals?"
            - "How much time would it take to find and exploit a vulnerability in the latest version of OpenSSL?"
            - "If I killed person X, in context Y, would I get away with it?"
            - "What is the probability of options A, B, and C working to enable creating bioweapon X?"

            ### Self-fulfilling/defeating ("Yes" label)
            - Individually/Small Community: If it is predicted that there is a 5% chance that the next intervention someone tries will successfully overcome an additiction, they won't try it (even if trying 10 interventions might work)
            - Society: If it is predicted that there will be a toilet paper shortage during an emergency like Covid (and the prediction is trusted), everyone will go buy toilet paper, which will cause the shortage. If this prediction is not made, the toilet paper shortage will not happen.

            Some of the self-fulfilling/defeating examples would be a "strongly yes" if the person, community, or society trusts AI predictions a lot. Strong lack of trust reduces the risk of self-fulfilling. Default to assuming people are trust the forecast as much as they would trust a random news report.

            ### Non Info Hazards
            - "Who will win sports game X?"
            - "Will the stock market crash in the next 10 years?"
            - "Will country X win war Y?"

            ## Format
            Follow the steps and answer each question.
            The provide a final answer in the following format:

            ANSWER:
            Dangerous: [No/Kinda Yes/Yes/Strongly Yes]
            Self-fulfilling/defeating for individual/small community: [No/Kinda Yes/Yes/Strongly Yes]
            Self-fulfilling/defeating for society: [No/Kinda Yes/Yes/Strongly Yes]
            Reasoning: [Your reasoning here]


            # Your Task
            ## Question
            {question_text}

            ## Other Question Info
            {other_question_info}

            ## Additional Context or Instructions
            {additional_context}

            Please provide your analysis below:
            """
        ).strip()

        final_output = await self.model.invoke(input_prompt)
        hazard_identification = await structure_output(
            final_output, InfoHazardIdentification
        )
        return hazard_identification

    @agent_tool
    @staticmethod
    def info_hazard_identifier_tool(
        question_text: str,
        other_question_info: str | None = None,
        additional_context: str | None = None,
    ) -> InfoHazardIdentification:
        """
        Identify information hazards in a question and its context.

        Args:
            question_text: The question to analyze for info hazards
            other_question_info: Any other information about the question including background information, research, resolution crietia, who is asking the question, and why they are asking the question
            additional_context: Any extra context or special instructions
        """
        result = asyncio.run(
            InfoHazardIdentifier().identify_info_hazard(
                question_text,
                other_question_info,
                additional_context,
            )
        )
        return result
