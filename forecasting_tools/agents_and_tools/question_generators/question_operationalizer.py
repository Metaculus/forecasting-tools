import asyncio
import random

from forecasting_tools.agents_and_tools.question_generators.simple_question import (
    SimpleQuestion,
)
from forecasting_tools.ai_models.agent_wrappers import agent_tool
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.misc import clean_indents


class QuestionOperationalizer:
    def __init__(
        self,
        model: str | GeneralLlm = "openrouter/perplexity/sonar-reasoning-pro",
    ) -> None:
        self.model: GeneralLlm = GeneralLlm.to_llm(model)
        self.example_full_questions = DataOrganizer.load_questions_from_file_path(
            "forecasting_tools/agents_and_tools/question_generators/q3_q4_quarterly_questions.json"
        )
        self.example_simple_questions = (
            SimpleQuestion.full_questions_to_simple_questions(
                self.example_full_questions
            )
        )
        self.random_example_question_sample = random.sample(
            self.example_simple_questions, 5
        )

    async def operationalize_question(
        self,
        question_title: str,
        related_research: str | None,
        additional_context: str | None = None,
    ) -> SimpleQuestion:
        examples = "\n".join(
            [str(question) for question in self.random_example_question_sample]
        )
        input_prompt = clean_indents(
            f"""
            # Instructions
            You are a forecasting question operationalizer. Your job is to take a question title and turn it into a full forecasting question with all necessary details.

            ## Key Guidelines:
            - Make sure the question is uncertain (so the answer provides useful insights)
                - For binary questions, forecasted probabilities should be between 10% and 90%
                - For numeric questions, the range should not be an obvious number
                - For multiple choice questions, probability for each option should not be more than 80% or less than 5%
            - The question should be clear, specific, and resolvable with public information
                - Good: "Will SpaceX launch a rocket on May 2nd 2023?"
                - Bad: "Will Elon mention his intention to launch in a private meeting by the end of 2023?"
            - Resolution criteria should be unambiguous (everyone should agree on the answer once it resolves)
            - Make sure the time horizon is appropriate - not too short or too long
            - Don't forget to INCLUDE links and citations wherever you can. If you reference the previous research, include the links this research cited using markdown.

            Follow the format provided in the examples and adhere strictly to the schema requirements.

            ## Fields
            {SimpleQuestion.get_field_descriptions()}

            ## Examples of good questions
            {examples}

            ## Your Task
            ### Question Title
            Please operationalize the following question title into a full forecasting question.
            {question_title}

            ### Additional Context
            {additional_context}

            ### Your previous research
            {related_research}

            Now please operationalize the question title into a full forecasting question.
            """
        ).strip()

        final_output = await self.model.invoke(input_prompt)
        if not final_output:
            raise RuntimeError(
                f"LLM answer is an empty string. The prompt was: {input_prompt}"
            )

        questions = await structure_output(final_output, SimpleQuestion)
        if not questions:
            raise ValueError("No question generated from the title.")
        return questions

    @agent_tool
    @staticmethod
    def question_operationalizer_tool(
        question_title: str, related_research: str
    ) -> SimpleQuestion:
        """
        Convert a question title to a fully formed SimpleQuestion object.

        Args:
            question_title: The title of the question to operationalize
            related_research: Include as much research as possible to help make a good question (especially include possible resolution sources)
        """

        result = asyncio.run(
            QuestionOperationalizer().operationalize_question(
                question_title,
                related_research,
            )
        )
        return result
