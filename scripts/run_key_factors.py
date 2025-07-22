import asyncio
import logging
from datetime import datetime

from forecasting_tools.agents_and_tools.deprecated.general_researcher import (
    GeneralResearcher,
)
from forecasting_tools.agents_and_tools.research.key_factors_researcher import (
    KeyFactorsResearcher,
    ScoredKeyFactor,
)
from forecasting_tools.data_models.questions import MetaculusQuestion, QuestionState
from forecasting_tools.helpers.metaculus_api import MetaculusApi
from forecasting_tools.util import file_manipulation
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


post_id_or_string_question = [
    # "How many units of humanoid robots sold for performing in-home daily tasks by 2030?",
    # "How many doctor positions will be filled by humans in 2030?",
    # "How many full time programmer positions will be filled by humans in 2030?",
    # "What percent of the current number of human jobs will AI systems be performing end-to-end at a human level in 2030?",
    # "Can AIs outperform superforecasters by 2030?",
    # "How many named machine learning algorithm advancements will have been autonomously discovered by AI systems without human intervention by 2030?",
    # "How many U.S. remote workers will delegate complex work (lasting >1 hour per task) at least 3 times per week to autonomous AI agents that can independently execute multi-step workflows and produce complete deliverables without continuous human supervision by 2030?",
    # "Global number of people living on less than $10k / year in 2030?",
    # "What will be the global 80/20 wealth concentration in 2030?",
    # "US average annual GDP growth rate from 2027 to 2030?",
    # "Global average annual GDP growth rate from 2027 to 2030?",
    # "How many hours per week will an average person work in 2030?",
    # "What will be the average number of years of schooling in 2030?",
    # "Civilian unemployment rate in 2030?",
    # "How many self-driving vehicles will be on the roads in developed countries in 2030?",
    # "What percent of customer service calls will be entirely handled by AI systems in 2030?",
    # "What will the global labor share of gross domestic product be in 2030?",
    # "What will the total world unemployment rate be in 2030?",
    # "What will the global median income or consumption per day be in 2030 in 2017 USD adjusted for purchasing power?",
    2534,  # https://www.metaculus.com/questions/2534/
    # 22519,  # https://www.metaculus.com/questions/22519/
    6614,  # https://www.metaculus.com/questions/6614/
    18546,  # https://www.metaculus.com/questions/18546/
    22244,  # https://www.metaculus.com/questions/22244/
    6046,  # https://www.metaculus.com/questions/6046/
    3244,  # https://www.metaculus.com/questions/3244/
    11588,  # https://www.metaculus.com/questions/11588/
    22954,  # https://www.metaculus.com/questions/22954/
    3366,  # https://www.metaculus.com/questions/3366/
    # 27811,  # https://www.metaculus.com/questions/27811/
    7444,  # https://www.metaculus.com/questions/7444/
]


async def run_key_factors_on_tournament(
    tournament_id: int | None = None,
) -> None:
    if tournament_id is None:
        questions = post_id_or_string_question
    else:
        tournament_questions = MetaculusApi.get_all_open_questions_from_tournament(
            tournament_id,
        )
        open_questions = [
            question
            for question in tournament_questions
            if question.state == QuestionState.OPEN
        ]
        questions = open_questions

    logger.info(f"Processing {len(questions)} questions")

    results = []
    for question in questions:
        result = asyncio.run(_process_question(question))
        results.append(result)
        logger.info(f"Processed question {question}\n{result}")

    date_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = f"logs/forecasts/key_factors/key_factors_{date_string}.json"
    file_manipulation.write_json_file(file_path, results)

    results = file_manipulation.load_json_file(file_path)
    formatted_output = _extract_all_markdown_info_from_key_results_file(results)

    output_path = f"logs/forecasts/key_factors/key_factors_{date_string}.txt"
    file_manipulation.log_to_file(output_path, formatted_output)

    logger.info(f"Question backgrounds written to {output_path}")


async def _process_question(question: int | str | MetaculusQuestion) -> dict:
    if isinstance(question, str):
        general_search_responder = GeneralResearcher(question)
        generated_background_info = (
            await general_search_responder.respond_with_markdown()
        )

        metaculus_question = MetaculusQuestion(
            question_text=question,
            background_info=generated_background_info,
            page_url="",
            api_json={},
        )
    if isinstance(question, int):
        metaculus_question = MetaculusApi.get_question_by_post_id(question)
    else:
        assert isinstance(question, MetaculusQuestion)
        metaculus_question = question

    background_info = metaculus_question.background_info

    key_factors = await KeyFactorsResearcher.find_and_sort_key_factors(
        metaculus_question, num_key_factors_to_return=5
    )
    key_factor_markdown = ScoredKeyFactor.turn_key_factors_into_markdown_list(
        key_factors
    )
    return_value = {
        "combined_markdown": f"## Background Information\n\n{background_info}\n\n## Key Factors\n\n{key_factor_markdown}\n\n*The above has been researched by AI and may have flaws*",
        "key_factors_markdown": key_factor_markdown,
        "background_info": background_info,
        "key_factors": [key_factor.model_dump_json() for key_factor in key_factors],
        "question": metaculus_question.to_json(),
    }
    return return_value


def _extract_all_markdown_info_from_key_results_file(
    results: list[dict],
) -> str:
    formatted_text = ""

    for result in results:
        question = MetaculusQuestion.from_json(result.get("question", {}))
        background = result.get("key_factors_markdown", "")

        if question and background:
            formatted_text += f"[Question]: {question.question_text}\n\n"
            formatted_text += f"[URL]: {question.page_url}\n\n"
            formatted_text += f"[Key Factors]:\n{background}\n\n"
            formatted_text += "-" * 80 + "\n\n"

    return formatted_text


if __name__ == "__main__":
    # Q4 2024 Quarterly cup is 3672
    CustomLogger.setup_logging()
    input_tournament_id = input("Enter tournament ID: ")
    if input_tournament_id:
        asyncio.run(run_key_factors_on_tournament(int(input_tournament_id)))
    else:
        asyncio.run(run_key_factors_on_tournament())
