from __future__ import annotations

import logging

from forecasting_tools.agents_and_tools.base_rates.base_rate_researcher import (
    BaseRateReport,
    BaseRateResearcher,
)
from forecasting_tools.agents_and_tools.deprecated.configured_llms import (
    BasicLlm,
)
from forecasting_tools.agents_and_tools.deprecated.general_researcher import (
    GeneralResearcher,
)
from forecasting_tools.agents_and_tools.deprecated.question_responder import (
    QuestionResponder,
)
from forecasting_tools.agents_and_tools.deprecated.question_router import (
    QuestionRouter,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import (
    clean_indents,
    strip_code_block_markdown,
)
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.util import async_batching

logger = logging.getLogger(__name__)


class ResearchCoordinator:

    def __init__(
        self,
        question: MetaculusQuestion,
    ) -> None:
        self.question = question

    async def create_full_markdown_research_report(
        self,
        num_of_background_questions: int,
        num_base_rate_questions: int,
        num_base_rate_questions_with_deep_research: int,
    ) -> str:
        background_markdown = await self.generate_background_markdown(
            num_of_background_questions,
        )
        base_rate_markdown = await self.generate_base_rate_markdown(
            num_base_rate_questions,
            num_base_rate_questions_with_deep_research,
            background_markdown,
        )
        combined_markdown = background_markdown + "\n\n" + base_rate_markdown
        return combined_markdown

    async def make_list_of_base_rate_reports(
        self,
        number_of_base_rate_reports: int,
        generate_background_markdown: bool = True,
    ) -> list[BaseRateReport]:
        if generate_background_markdown:
            num_context_questions = 3
            background_markdown = await self.generate_background_markdown(
                num_context_questions
            )
        else:
            background_markdown = "No background information was generated"
        base_rate_questions = await self.brainstorm_base_rate_questions(
            number_of_base_rate_reports, background_markdown
        )
        base_rate_tasks = [
            BaseRateResearcher(question).make_base_rate_report()
            for question in base_rate_questions
        ]
        base_rate_reports, _ = (
            async_batching.run_coroutines_while_removing_and_logging_exceptions(
                base_rate_tasks
            )
        )
        return base_rate_reports

    async def generate_background_markdown(
        self,
        num_background_questions: int,
        additional_context: str | None = None,
    ) -> str:
        questions = await self.brainstorm_background_questions(
            num_background_questions, additional_context
        )
        answers = await self.answer_question_list(questions, GeneralResearcher)
        logger.info("Generated background markdown.")
        return await self.__create_question_answer_markdown_section(
            questions, answers, question_prepend="Q"
        )

    async def generate_base_rate_markdown(
        self,
        num_base_rate_questions: int,
        num_base_rate_questions_with_deep_research: int,
        additional_context: str,
    ) -> str:
        questions = await self.brainstorm_base_rate_questions(
            num_base_rate_questions, additional_context
        )
        deep_questions, shallow_questions = (
            await self.pick_best_base_rate_questions(
                num_base_rate_questions_with_deep_research, questions
            )
        )
        deep_answers = await self.answer_question_list(
            deep_questions, BaseRateResearcher
        )
        shallow_answers = await self.answer_question_list(
            shallow_questions, GeneralResearcher
        )
        combined_questions = deep_questions + shallow_questions
        combined_answers = deep_answers + shallow_answers
        markdown = await self.__create_question_answer_markdown_section(
            combined_questions, combined_answers, question_prepend="B"
        )
        logger.info("Generated base rate markdown.")
        return markdown

    async def pick_best_base_rate_questions(
        self,
        num_base_rate_questions_to_pick: int,
        questions: list[str],
    ) -> tuple[list[str], list[str]]:
        assert num_base_rate_questions_to_pick <= len(questions)
        assert num_base_rate_questions_to_pick >= 0
        if num_base_rate_questions_to_pick == 0:
            return [], questions
        if num_base_rate_questions_to_pick == len(questions):
            return questions, []
        deep_questions = await self.__pick_out_best_base_rate_questions(
            num_base_rate_questions_to_pick, questions
        )
        shallow_questions = [q for q in questions if q not in deep_questions]
        return deep_questions, shallow_questions

    async def brainstorm_background_questions(
        self,
        num_background_questions: int,
        additional_context: str | None = None,
    ) -> list[str]:
        logger.info(
            f"Running forecasts on question `{self.question.question_text}`"
        )
        prompt = clean_indents(
            f"""
            # Instructions
            You are a superforecaster forecasting on Metaculus, and are brainstorming questions to research to help you make a better prediction.
            What questions would you ask to help you make a better prediction?

            Consider questions about
            - Recent news
            - Anything important players or key decision makers have said about the event
            - How outcomes are decided (e.g. if predicting something like an election, or a committee decision)
            - General stats related to the event (max, min, average, variation, current value, growth rate, etc.)
            - Anything else you can think of that might help you make a better prediction.

            Use your judgement to decide if you should ask more of one type of question over another (e.g. a numeric question would benefit from more stat questions)
            Give your answer as a list of questions. Make sure this list is parsable by python with quotes around the questions. You should give the list and only the list.
            Also consider other creative ways to aggregate information. For instance if you are prediciting things related to popularity (like Eurovsion results) you might want to ask the view count of key competitors on YouTube.
            Please come up with {num_background_questions} questions.

            # Example
            Lets say the question is "Will SpaceX launch a rocket in 2023?"
            Answer:
            [
            "What is SpaceX?",
            "What is the recent news of SpaceX?",
            "How do rockets launches work?",
            "What is the recent news of SpaceX regarding rocket launches?",
            "What has SpaceX and Elon Musk said about rocket launches this year?",
            "Has SpaceX already launched a rocket in 2023?",
            "What are developments in the space industry in 2023 especially around rocket launches?",
            "Is there anyone not wanting SpaceX to launch a rocket in 2023?",
            "What reasons might there be for SpaceX not launching a rocket in 2023?",
            "What reasons might there be for SpaceX successfully launching a rocket in 2023?",
            "What will the weather be like in 2023 for SpaceX rocket launches? Is there planned to be bad weather on planned launch dates?"
            "What is the max number of rockets SpaceX has launched in a year?"
            "What is the average number of rockets SpaceX has launched in a year?"
            ]

            # Question Info
            {self.question.give_question_details_as_markdown()}

            # Previous Research
            {additional_context or "No previous research was provided"}

            # Your Turn
            Please come up with {num_background_questions} questions.
            """
        )
        model = BasicLlm(temperature=0.8)
        questions_to_get_context: list[str] = (
            await model.invoke_and_return_verified_type(prompt, list[str])
        )

        logger.info(
            f"Brainstormed {len(questions_to_get_context)} questions to get context"
        )
        full_questions_to_get_context = [
            f"In the context of the larger question '{self.question.question_text}', {question}"
            for question in questions_to_get_context
        ]
        return full_questions_to_get_context

    async def brainstorm_base_rate_questions(
        self,
        num_base_rate_questions: int,
        additional_context: str | None = None,
    ) -> list[str]:
        prompt = clean_indents(
            f"""
            You are a superforecaster forecasting on Metaculus.

            # Instructions
            You are trying to fine questions related to base rates you can compare to? Please make {num_base_rate_questions} questions.
            You'll also be given some additional research you've done thus far that can be used to find better base cases.
            Give your answer as a list of questions. Make sure this list is parsable by python with quotes around the questions. There should be no comma after the last line of the list. You should give the list and only the list.

            # Examples
            ## Example 1
            Lets say the question is "Will SpaceX launch a rocket in 2023?"
            Lets say you found that SpaceX has said they will launch a rocket in 2023.
            You would answer:
            [
            "How often has SpaceX launched rockets over the last 5 years?",
            "How often has SpaceX launched rockets in the past since its founding to today?",
            "How often has SpaceX said they said they will launch and not launched since its founding to today?"
            ]

            ## Example 2
            Lets say the question is "Will the US GDP be higher than 20 trillion for the year of 2023?"
            Lets say you also found there was a recession in 2022.
            Lets say there was an AI tech boom in 2023.
            Lets say that last year (2022) the US GDP was 19 trillion.
            You would answer:
            [
            "How often has the US yearly GDP increased by more than 1 trillion in the past 10 years?",
            "How often has the US yearly GDP increased by 5% or more in the last 50 years after a large tech industry change similar to the AI boom?",
            "How often has the US yearly GDP increased by more than 1 trillion in the last 50 years after a recession?"
            ]

            ## Example 3
            Lets say the question is "Before October 1, 2024, will Stripe announce on the news section of its website that it is planning an IPO?"
            Lets say also that you found that a Stripe CEO said in a podcast that they are planning an IPO this year.
            You would answer:
            [
            "Over the history of Stripe, how often has Stripe announced on the news section of its website that it is planning an IPO?",
            "How often has Stripe announced on the news section of its website that it is planning an IPO after a CEO said they are planning an IPO this year?",
            "How often has a company announced on the news section of its website that it is planning an IPO after a CEO said they are planning an IPO in that year year?"
            ]

            ## Example 4
            Lets say the question is "Will the CDC report 21 to 100 total human cases of H5 in the United States on October 1, 2024?"
            lets say that the CDC has already reported 3 cases of H5 in the US in 2024.
            Lets say that H5 is a bird flu and is hard for humans to catch
            You would answer:
            [
            "Over the last 100 years how often has the CDC reported 21 to 100 total human cases of H5 in the US after they have already reported 3 cases in a year?",
            "Over the last 50 years, how often has there been an increase of 18 human cases of a animal-born disease (like H5) after there have already been a few (~3) human cases in a year?",
            "How often have there been H5 pandemics in the past 100 years?"
            ]

            ## Example 5
            Lets say the question is "What will Apple's stock price be on October 1, 2024?"
            Lets say you learned that the lowest Apple's stock price was is $0.1 and the highest was $300
            Lets say Apple's stock price is $280 right now.
            Lets say today is September 1, 2024
            You would answer:
            [
            "Over the last 1 year, how often has Apple's stock price been above $280?",
            "Over the last 1 year, how often has Apple's stock price been above $150?",
            "Over the last 20 years, how often has Apple's stock price risen by more than $25 in a 1 month period?",
            "Over the last 20 years, how often has Apple's stock price fallen by more than $25 in a 1 month period?"
            ]
            NOTICE: With a question about 'what number will something be' you should guess some numbers you think are likely, then ask how often its been those numbers

            # Question Info
            {self.question.give_question_details_as_markdown()}

            # Research You've Done So Far
            {additional_context or "No Additional Context was given"}

            # Final Comments
            Please come up with {num_base_rate_questions} questions to find a base rate to compare to.
            """
        )

        model = BasicLlm(temperature=0.8)
        base_rate_questions: list[str] = (
            await model.invoke_and_return_verified_type(prompt, list[str])
        )

        logger.info(
            f"Brainstormed {len(base_rate_questions)} questions for baserate"
        )
        question_text_prepend = self.__get_question_context_prepend()
        full_questions_to_get_context = [
            f"{question_text_prepend}{question}"
            for question in base_rate_questions
        ]
        return full_questions_to_get_context

    async def answer_question_list(
        self,
        questions: list[str],
        responder_type: type[QuestionResponder] | None = None,
    ) -> list[str]:
        question_router = QuestionRouter()
        if responder_type is None:
            answering_question_coroutines = [
                question_router.answer_question_with_markdown_using_routing(
                    question
                )
                for question in questions
            ]
        else:
            answering_question_coroutines = [
                responder_type(question).respond_with_markdown()
                for question in questions
            ]
        exception_handled_coroutines = (
            async_batching.wrap_coroutines_to_return_not_raise_exceptions(
                answering_question_coroutines
            )
        )
        unverified_answers: list[str | Exception] = (
            async_batching.run_coroutines(exception_handled_coroutines)
        )
        verified_answers = []
        for question, answer in zip(questions, unverified_answers):
            if isinstance(answer, Exception):
                logger.warning(
                    f"Error in answering question `{question}`: {answer}"
                )
                verified_answer = "Error in generating answer"
            elif isinstance(answer, str):
                verified_answer = answer
            else:
                raise ValueError(
                    f"answer is not a string or exception: {type(answer)}"
                )
            verified_answers.append(verified_answer)

        logger.info(
            f"Answered {len(verified_answers)} questions using {responder_type.__name__ if responder_type else 'QuestionRouter'}"
        )
        return verified_answers

    async def summarize_full_research_report(
        self, research_as_markdown: str
    ) -> str:
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster working to summarize research you've done.
            Your goal is to summarize a research report the superforecasters have done.
            They have only 1min to read your summary, so make it concise and specific. Bring up signal not noise.

            # Instructions
            Please make a markdown report with three sections:
            1. Research Overview: Give 2 paragraphs summarazing the research done. Surface things people would want to know for a forecast.
            2. Possible Base Rates: Make one bullet point for each unique possible base rate. Prioritize numbers, and do some calculations to find historical rates if possible (e.g. if you find there are 3 successful X out of 10 X total, then state your calculation and say 3 successful X out of 10 X total is 30% success rate).
            3. Pros Section: Make one bullet point for each unique pro. These should be inside view adjustments that would move your forecast up.
            4. Cons Section: Make one bullet point for each unique con. These should be outside view adjustments that would move your forecast down.

            Please cite from which question you got your information from (e.g. [Q2] for question 2, or [B1] for base rate question 1).
            Please try to prioritize things that people might miss on their own
            Don't use any of your own information. Only use information from the research report.

            # Compensation
            The superforecaster you work with is compensated based on the accuracy of their predictions, and you will get a cut of their compensation. You can make up to $1000 if you can increase their score with accurate information. Remember they are successful as much as they are properly able to assess uncertainty (i.e. overconfidence can be just as bad as underconfidence). Potray things as they are, and avoid misinformation, biased wording, point out potential biases in the data, etc.

            # Example
            An example is given in triple backticks below for the question of "Will North Korea conduct a nuclear test before 2025?".

            ```
            #### Research Overview
            Recent activities and statements by North Korean officials indicate a continued focus on their nuclear program, with plans to increase weapon-grade nuclear materials and arsenal. The nearly complete Experimental Light Water Reactor (ELWR) at Yongbyon could significantly boost plutonium production. North Korea has declared itself an "irreversible" nuclear power, preparing for a "real war" and threatening to turn the Pacific into a "firing range" [Q1]. These developments suggest a heightened commitment to advancing nuclear capabilities, potentially influencing the likelihood of a seventh nuclear test before 2025.
            North Korea has historically timed nuclear tests to coincide with significant international events, such as U.S. elections, to maximize attention and influence diplomatic dynamics. This pattern suggests a potential nuclear test around the 2024 U.S. Presidential Election to leverage international focus and influence the incoming administration's policy stance [Q2]. However, the Economist Intelligence Unit predicts North Korea may refrain from nuclear tests in 2023-24 due to economic dependence on China, which opposes further nuclear development [Q3].

            #### Possible Base Rates
            - Since North Korea began nuclear testing in 2006, it has conducted nuclear tests within three months of a U.S. Presidential Election on two occasions. There have been 4 U.S. Presidential Elections between 2006 and today (Oct 20 2024). 2/4 = 50% chance of a test within three months of a U.S. Presidential Election [B1].
            - Historically, North Korea has conducted six nuclear tests, with varying degrees of advance warning from credible sources [B2].
            - North Korea has a history of conducting nuclear tests following public statements about advancing their nuclear capabilities [B3].

            #### Pros
            - North Korea has a history of timing its nuclear tests and missile launches to coincide with significant international events, including U.S. elections, to maximize global attention and potentially influence diplomatic dynamics [Q2].
            - The president of South Korea said that he expects North Korea to conduct a nuclear test before 2025 [Q3].

            #### Cons
            - The Economist Intelligence Unit (EIU) predicts that North Korea may refrain from conducting nuclear tests in 2023-24 due to its economic dependence on China, which opposes further nuclear development [Q3].
            - Though there is a pattern of signs or reports of activity at nuclear sites, the exact timing of tests has been upredictable in the past and can vary between 0 to 3 months of the related indicating event if it happens at all [B2].
            ```

            # Question details
            Below are the details of the question the superforecaster is predicting on:

            {self.question.give_question_details_as_markdown()}

            # Research Report To Summarize
            Below in the triple back tick code block, is the research report you need to summarize.

            ```
            {research_as_markdown}
            ```

            Now please summarize the research report above using the markdown template given to you. Just fill in the template and give the markdown report, do not include any other text. Your summary will be published as is.
            """
        )
        model = BasicLlm(temperature=0)
        summary_markdown = await model.invoke(prompt)
        cleaned_summary_markdown = strip_code_block_markdown(summary_markdown)
        return cleaned_summary_markdown

    async def __pick_out_best_base_rate_questions(
        self,
        num_base_rate_questions_with_deep_research: int,
        base_rate_questions: list[str],
    ) -> list[str]:
        number_of_questions_to_pick = (
            num_base_rate_questions_with_deep_research
        )
        prompt = clean_indents(
            f"""
            You are a superforecaster forecasting on Metaculus.

            # Question Details
            {self.question.give_question_details_as_markdown()}

            # Instructions
            You have brainstormed some base rate questions. Now you need to pick out the best ones to use as base rates. You should pick the ones that are most likely to be useful and easy to find information on. Don't pick any that are hard to find information on, or are not a good reference class for the question.
            Give your answer as a list of questions. Make sure this list is parsable by python with quotes around the questions. You should give the list and only the list.
            You should pick only {number_of_questions_to_pick} questions.


            # Example
            If you were given the list
            [
            "How often has SpaceX launched rockets over the last 5 years?",
            "How often has SpaceX launched rockets in the past since its founding to today?",
            "How often has SpaceX said they said they will launch and not launched since its founding to today?"
            ]
            and told to pick 1 question, you would answer:
            [
            "How often has SpaceX launched rockets over the last 5 years?"
            ]

            # The List of Base Rate Questions
            Please pick {number_of_questions_to_pick} questions from the list below:

            {base_rate_questions}
            """
        )
        model = BasicLlm(temperature=0)
        picked_questions: list[str] = (
            await model.invoke_and_return_verified_type(prompt, list[str])
        )
        assert len(picked_questions) == number_of_questions_to_pick
        return picked_questions

    def __get_question_context_prepend(self) -> str:
        return f"In the context of the larger question '{self.question.question_text}', "

    async def __create_question_answer_markdown_section(
        self,
        questions_to_get_context: list[str],
        answers: list[str],
        question_prepend: str,
    ) -> str:
        context_prepend = self.__get_question_context_prepend()
        questions_without_context_prepended = [
            question.replace(context_prepend, "")
            for question in questions_to_get_context
        ]
        question_answer_pairs = list(
            zip(questions_without_context_prepended, answers)
        )
        questions_with_answer_as_markdown = [
            f"## {question_prepend}{i + 1}: {pair[0]}\n  Answer:\n {pair[1]}\n\n"
            for i, pair in enumerate(question_answer_pairs)
        ]
        combined_question_answer_markdown = "\n".join(
            questions_with_answer_as_markdown
        )
        return combined_question_answer_markdown
