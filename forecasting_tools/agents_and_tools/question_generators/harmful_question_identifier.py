import asyncio
from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from forecasting_tools.ai_models.agent_wrappers import agent_tool
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.misc import clean_indents


class HarmRating(Enum):
    NO = "No"
    KINDA_YES = "Kinda Yes"
    YES = "Yes"
    STRONGLY_YES = "Strongly Yes"

    @property
    def passes_threshold(self) -> bool:
        return self in [HarmRating.YES, HarmRating.STRONGLY_YES]


class HarmfulQuestionIdentification(BaseModel):
    original_question: str
    is_directly_harmful: HarmRating
    self_fulfilling_or_defeating_for_individual: HarmRating
    self_fulfilling_or_defeating_for_society: HarmRating
    alternative_questions: list[str]
    reasoning: str

    @property
    def is_hazard(self) -> bool:
        if (
            self.is_directly_harmful.passes_threshold
            or self.self_fulfilling_or_defeating_for_individual.passes_threshold
            or self.self_fulfilling_or_defeating_for_society.passes_threshold
        ):
            return True
        return False

    @property
    def as_string(self) -> str:
        alternative_questions = ""
        for question in self.alternative_questions:
            alternative_questions += f"- {question}\n"
        return clean_indents(
            f"""
            ## Question: {self.original_question}
            - **Is directly Harmful**: {self.is_directly_harmful.value}
            - **Self-fulfilling/defeating for individual**: {self.self_fulfilling_or_defeating_for_individual.value}
            - **Self-fulfilling/defeating for society**: {self.self_fulfilling_or_defeating_for_society.value}

            **Potential Alternative Questions**
            {alternative_questions}

            **Reasoning**
            {self.reasoning}
            """
        )


class HarmfulQuestionIdentifier:
    def __init__(
        self,
        model: str | GeneralLlm = "openrouter/perplexity/sonar-reasoning-pro",
    ) -> None:
        self.model: GeneralLlm = GeneralLlm.to_llm(model)

    async def identify(
        self,
        question_text: str,
        other_question_info: str | None = None,
        additional_context: str | None = None,
    ) -> HarmfulQuestionIdentification:
        # TODO: Make the prompting here focused more on self-fulfilling prophecies (not "info-hazard") and make sure that results still are good.
        input_prompt = clean_indents(
            f"""
            # Instructions
            You are an self-fulfilling prophecy and info hazard identifier for forecasting questions and whether they should be posted on platforms like Metaculus, Polymarket, or similar. Your job is to analyze the provided question and context, and determine if there are any self-fulfilling prophecies or information hazards present and explain why or why not.

            You will check 3 different types of harmful aspects:
            1. Dangerous information
            2. Self-fulfilling/defeating information on the individual level
            3. Self-fulfilling/defeating information on the society level

            For each you will identify if the harmful aspect is:
            - No
            - Kinda Yes
            - Yes
            - Strongly Yes

            The vast majority of the questions will fall into the No or Yes category. Save Strongly Yes for rare cases.

            Finally you will provide an alternative way to ask the question that would not result in an self-fulfilling prophecy or info hazard.

            ## What is a harmful question?
            - Questions should seek to avoid harmful self-fulfilling or self-negating effects, where the aggregate forecast could itself influence whether the event in question occurs.
            - Information hazards are risks posed by the spread of factual information that could lead to harmful outcomes if they reach certain actors or audiences. These range from technical vulnerabilities that could compromise systems, to extreme cases such as information about how to build a bioweapon.
            Information hazards and circular effects are often difficult to assess and can vary substantially in the level of risk posed.

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
            4. Check if question has self-fulfilling/defeating information on the society level
                - Ask "If everyone in the world is told that the probability of event X is low, what will be the probability of event X?"
                - Ask "If everyone in the world is told that the probability of event X is high, what will be the probability of event X?"
                - If there is a significant difference between the two answers, then there is a self-fulfilling information hazard on the society level
            5. Come up with an alternative way to ask the question that would not result in an info hazard.
                - Make sure its in a format that could be asked on Metaculus, Polymarket, etc (e.g. about a future event and resolves to "yes/no" a set of options or a number)
                - Give multiple options
                - Apply the same evaluation to your ideas to make sure that they are not info hazards. Don't give ideas if they are info hazards.

            Also remember that today is {datetime.now().strftime("%Y-%m-%d")}.

            ## Examples

            ### Dangerous information
            - "If I tried, could I synthesize a lethal toxin using common household chemicals?"
            - "How much time would it take to find and exploit a vulnerability in the latest version of OpenSSL?"
            - "If I killed person X, in context Y, would I get away with it?"
            - "What is the probability of options A, B, and C working to enable creating bioweapon X?"

            ### Self-fulfilling/defeating ("Yes" label)
            - Individually/Small Community: If it is predicted that there is a 5% chance that the next intervention someone tries will successfully overcome an addiction, they won't try it (even if trying 10 interventions might work)
            - Society: If it is predicted that there will be a toilet paper shortage during an emergency like Covid (and the prediction is trusted), everyone will go buy toilet paper, which will cause the shortage. If this prediction is not made, the toilet paper shortage will not happen.

            Some of the self-fulfilling/defeating examples would be a "strongly yes" if the person, community, or society trusts AI predictions a lot. Strong lack of trust reduces the risk of self-fulfilling. Default to assuming people trust the forecast as much as they would trust a random news report.

            ### Reframing questions
            *Example 1*
            - Original: "Will there be a toilet paper shortage in the next month?"
            - Reframed:
                - "If people believe there will be a toilet paper shortage in the next month, will there be a toilet paper shortage in the next month?"
                - "If people don't believe there will be a toilet paper shortage in the next month, will there be a toilet paper shortage in the next month?"
                - "What will the average number of toilet paper shortages be per year over the next 3 years?"
                - "By how much will the ratio of 'toilet paper rolls produced in US' to 'US citizens' change in the next month?"
            - Notes: Target things that will defuse the self-fulfilling/defeating effect by helping people recognize or see through it.

            *Example 2*
            - Original: "How much time would it take to find and exploit a vulnerability in the latest version of OpenSSL?"
            - Reframed:
                - "How many OpenSSL vulnerabilities will be found in the next 3 years?"
                - "What would the number of OpenSSL vulnerabilities be in the next 3 years given policy X, Y, and Z are taken" (fill in policy)

            ### Non Info Hazards
            - "Who will win sports game X?"
            - "Will the stock market crash in the next 10 years?"
            - "Will country X win war Y?"

            ## Format
            Follow the steps and answer each question.
            The provide a final answer in the following format:

            ANSWER:
            Question being evaluated: [Question being evaluated here]
            Dangerous: [No/Kinda Yes/Yes/Strongly Yes]
            Self-fulfilling/defeating for individual/small community: [No/Kinda Yes/Yes/Strongly Yes]
            Self-fulfilling/defeating for society: [No/Kinda Yes/Yes/Strongly Yes]
            Alternative way to ask the questions: [List of alternative ways to ask the question here]
            Reasoning: [Your reasoning here (formatted as a bullet point for each major step)]

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
            final_output, HarmfulQuestionIdentification
        )
        return hazard_identification

    @agent_tool
    @staticmethod
    def harmful_question_identifier_tool(
        question_text: str,
        other_question_info: str | None = None,
        additional_context: str | None = None,
    ) -> str:
        """
        Identify self-fulfilling prophecies and harmful information in a question and its context.

        Args:
            question_text: The question to analyze for self-fulfilling prophecies and info hazards
            other_question_info: Any other information about specific aspects of the question including background information, research, resolution criteria, who is asking the question, and why they are asking the question
            additional_context: Any extra context or special instructions
        """
        result = asyncio.run(
            HarmfulQuestionIdentifier().identify(
                question_text,
                other_question_info,
                additional_context,
            )
        )
        return result.as_string
