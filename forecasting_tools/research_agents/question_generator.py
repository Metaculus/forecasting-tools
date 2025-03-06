import logging
from datetime import datetime

from pydantic import BaseModel

from forecasting_tools.ai_models.general_llm import GeneralLlm

logger = logging.getLogger(__name__)


class SimpleQuestion(BaseModel):
    question_text: str
    resolution_criteria: str
    background_information: str
    expected_resolution_date: datetime


class QuestionGenerator:

    def __init__(self, model: GeneralLlm | str = "o1"):
        if isinstance(model, str):
            self.model = GeneralLlm(model=model)
        else:
            self.model = model

    async def generate_questions(
        self, number_of_questions: int = 3, prompt: str = ""
    ) -> list[SimpleQuestion]:
        pass


# class QuestionGenerator:

#     @classmethod
#     async def get_example_questions(cls) -> list[ShortQuestion]:
#         q1_filter = ApiFilter(
#             allowed_tournaments=[MetaculusApi.Q3_2024_QUARTERLY_CUP],
#             allowed_statuses=["open", "closed", "resolved"],
#         )
#         q4_filter = ApiFilter(
#             allowed_tournaments=[MetaculusApi.Q4_2024_QUARTERLY_CUP],
#             allowed_statuses=["open", "closed", "resolved"],
#         )
#         q1_questions = await MetaculusApi.get_questions_matching_filter(
#             q1_filter
#         )
#         q4_questions = await MetaculusApi.get_questions_matching_filter(
#             q4_filter
#         )
#         questions = q1_questions + q4_questions
#         short_questions = []
#         for question in questions:
#             assert question.resolution_criteria is not None
#             assert question.background_info is not None
#             assert question.scheduled_resolution_time is not None
#             short_questions.append(
#                 ShortQuestion(
#                     question_text=question.question_text,
#                     resolution_criteria=question.resolution_criteria,
#                     background_information=question.background_info,
#                     expected_resolution_date=question.scheduled_resolution_time,
#                 )
#             )
#         return short_questions

#     @classmethod
#     async def run(cls) -> None:
#         number_of_questions = 3
#         use_perplexity = False
#         short_questions = await cls.get_example_questions()

#         resolution_criteria_explanation = clean_indents(
#             """
#             Resolution criteria are highly specific way to resolve this that will always be super obvious in retrospect.
#             Resolution criteria should pass the clairvoynce test such that after the event happens there is no debate about whether it happened or not.
#             It should be meaningful and pertain to the intent of the question
#             Ideally you give a link for where this information can be found (i.e. something on a page that is clearly a 'yes' or 'no' because of a number shown or text shown).
#             However other methods apply.
#             """
#         )
#         prompt = clean_indents(
#             f"""
#             Search the web and find {number_of_questions} of important news items that a Metaculus question could be written about and make questions about them.
#             Please list out the questions. This should be a json list parsable in python.
#             Questions should resolve between 1 week and 3 months from now.

#             {resolution_criteria_explanation}

#             Here are some example questions:
#             {short_questions}

#             Here is the schema for the questions you should return. Remember to make it parsable in python:
#             {SmartSearcher.get_schema_format_instructions_for_pydantic_type(ShortQuestion)}
#             """
#         )

#         logger.info(
#             "Prompt:-----------------------------------------------------"
#         )
#         logger.info(prompt)
#         model = None
#         if use_perplexity:
#             model = GeneralLlm(model="perplexity/sonar-deep-research")
#         else:
#             # smart_searcher_model = GeneralLlm(
#             #     model="claude-3-7-sonnet-latest",
#             #     temperature=1,
#             #     max_tokens=64000,
#             #     timeout=240,
#             #     thinking={
#             #         "type": "enabled",
#             #         "budget_tokens": 20000
#             #     },
#             # )
#             smart_searcher_model = GeneralLlm(model="o1", temperature=0)
#             model = SmartSearcher(
#                 model=smart_searcher_model,
#                 num_searches_to_run=5,
#                 num_sites_per_search=10,
#                 use_brackets_around_citations=False,
#             )
#         questions = await model.invoke_and_return_verified_type(
#             prompt, list[ShortQuestion]
#         )

#         logger.info(
#             "Questions:-----------------------------------------------------"
#         )
#         for question in questions:
#             logger.info(question.model_dump_json())

#         tasks: list[Coroutine[Any, Any, ShortQuestion]] = []
#         for question in questions:
#             prompt = clean_indents(  # TODO: Try with resolution criteria prompt instructions
#                 f"""
#                 The below question has not been reviewed yet and the resolution criteria is probably not very good.

#                 Here is the question:
#                 {question.model_dump_json()}

#                 Please improve the resolution criteria and ideally add a link to it.
#                 Look for clear places that could help resolve the question.
#                 You have to be more than 100% confident that the resolution criteria will be unambiguous in retrospect.
#                 Consider ways that this could go wrong.

#                 Here is the schema for the question you should return:
#                 {SmartSearcher.get_schema_format_instructions_for_pydantic_type(ShortQuestion)}
#                 """
#             )
#             logger.info(
#                 "Prompt:-----------------------------------------------------"
#             )
#             logger.info(prompt)
#             tasks.append(
#                 model.invoke_and_return_verified_type(prompt, ShortQuestion)
#             )

#         refined_question = await asyncio.gather(*tasks, return_exceptions=True)
#         logger.info(
#             "Refined Question:-----------------------------------------------------"
#         )
#         for result in refined_question:
#             if isinstance(result, Exception):
#                 logger.error(f"Error: {result}")
#             else:
#                 logger.info(result.model_dump_json())

#     asyncio.run(run())
