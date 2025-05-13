import random

from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from forecasting_tools.agents_and_tools.question_generators.simple_question import (
    SimpleQuestion,
)
from forecasting_tools.agents_and_tools.research_tools import (
    get_general_news,
    perplexity_search,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.forecast_helpers.structure_output import (
    structure_output,
)


class QuestionOperationalizer:
    def __init__(
        self,
        model: str = "o4-mini",
    ) -> None:
        self.model: str = GeneralLlm.to_model_name(model)
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

    def _get_agent_instructions(self) -> str:
        examples = "\n".join(
            [str(question) for question in self.random_example_question_sample]
        )
        prompt = clean_indents(
            f"""
            # Instructions
            You are a forecasting question operationalizer. Your job is to take a question title and turn it into a full forecasting question with all necessary details.

            ## Steps
            1. Research general information about the question
            2. Ideate on potential resolution sources and make 3 parallel queries to dive deeper
            3. Determine a resolution source and criteria
            4. If the question is numeric, please also research reasonable upper and lower bounds (Keep the bounds very wide)
            5. State your final question and elaborate on all fields

            ## Key Guidelines:
            - Make sure the question is uncertain:
            - For binary questions, probabilities should be between 10% and 90%
            - For numeric questions, the range should not be an obvious number
            - For multiple choice questions, probability for each option should not be more than 80% or less than 5%
            - The question should be clear, specific, and resolvable with public information
            - Resolution criteria should be unambiguous
            - Make sure the time horizon is appropriate - not too short or too long

            Follow the format provided in the examples and adhere strictly to the schema requirements.

            ## Fields
            {SimpleQuestion.get_field_descriptions()}

            ## Examples of good questions
            {examples}
            """
        ).strip()
        return prompt

    async def question_title_to_simple_question(
        self,
        question_title: str,
    ) -> SimpleQuestion:
        input_prompt = f"Please turn this question title into a full forecasting question: \n'{question_title}'. "

        agent = Agent(
            name="Question Operationalizer Agent",
            instructions=self._get_agent_instructions(),
            tools=[get_general_news, perplexity_search],
            model=LitellmModel(model=self.model),
            output_type=None,
        )

        result = await Runner.run(agent, input_prompt)
        final_output = result.final_output

        questions = await structure_output(str(final_output), SimpleQuestion)
        if not questions:
            raise ValueError("No question generated from the title.")
        return questions

    @function_tool
    @staticmethod
    def question_title_to_simple_question_tool(
        question_title: str,
    ) -> SimpleQuestion:
        """
        Convert a question title to a fully formed SimpleQuestion object.

        Args:
            question_title: The title of the question to operationalize
            num_weeks_till_resolution: Number of weeks until the question should resolve

        Returns:
            A SimpleQuestion object with all fields filled out
        """
        import asyncio

        return asyncio.run(
            QuestionOperationalizer().question_title_to_simple_question(
                question_title,
            )
        )
