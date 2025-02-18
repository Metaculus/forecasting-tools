from datetime import datetime
from typing import NamedTuple

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.forecast_helpers.smart_searcher import SmartSearcher
from forecasting_tools.research_agents.general_researcher import (
    GeneralResearcher,
)


class EstimationResult(NamedTuple):
    count: int
    explanation: str


class Estimator:

    def __init__(
        self,
        type_of_thing_to_estimate: str,
        previous_research: str | None = None,
    ):
        """
        type_of_thing_to_estimate: The type of thing you are estimating the size of like "number of piano tuners in NYC"
        """
        self.__type_of_thing_to_estimate = type_of_thing_to_estimate
        self.__previous_research = previous_research

    async def estimate_size(self) -> EstimationResult:
        research_to_use: str | None = self.__previous_research
        if not self.__previous_research:
            research_to_use = await GeneralResearcher(
                f"What is information that would be useful for estimating the following: {self.__type_of_thing_to_estimate}?"
            ).respond_with_markdown()

        prompt = clean_indents(
            f"""
            # Instructions
            You are an super expert genius estimator and superforecaster, child of Fermi himself, trying to figure out how often something has happened between two dates.

            Please follow the below steps:
            1) List out all the facts you know. These are direct information or numbers from the internet. Each one MUST have a at least one citation.
            2) Using the facts, run a fermi estimate (or other estimation technique) to get your best guesstimate of the number of events.
            3) Give your final answer
            Make sure your answer is given in json following the example. Return only the json and nothing else.

            # What else you know
            Today is: {datetime.now().strftime("%Y-%m-%d")}
            A friend has given you the following information:
            ```
            {research_to_use}
            ```
            Note, you are encouraged to reuse your friend's citations, but they must use letters rather than numbers (e.g. [A](https://example.com) not [1](https://example.com))
            When reusing citations, you must use the URL and should include the text fragment as well.
            If you reusea a citation you MUST say the exact same thing that your friend did. You didn't see the original context, and don't want to misquote.


            # Payment
            You are being paid hourly, and a manager will evaluate your answer. If it passes inspection, you will be paid a $100 bonus.
            You will lose $10 from your final pay for every bad citation you give.

            # Example
            For example: If you were asked to estimate the "Number of piano tuners in NYC?"
            {{
                "facts": [
                    "The population of New York City is approximately 8.4 million [A](https://www.census.gov/quickfacts/newyorkcitynewyork#:~:text=The%20population%20of%20New%20York%20City%20is%20approximately%208.4%20million)",
                    "There is roughly 1 piano per 50 people in the United States [3]",
                    "Pianos should be tuned 2-4 times a year [B](https://www.steinway.com/news/features/care-and-maintenance-of-a-steinway#:~:text=Pianos%20should%20be%20tuned%202%2D4%20times%20a%20year)",
                    "A piano tuner can tune 2-4 pianos per day [2]",
                    "Piano tuners typically work 5 days a week, 50 weeks a year [7]"
                ],
                "reasoning_steps": [
                    "Estimate number of pianos in NYC: 8.4 million / 50 = 168,000 pianos",
                    "Estimate annual tunings: 168,000 * 3 (average of 2-4) = 504,000 tunings/year",
                    "Tunings per tuner per year: 3 (avg) * 5 days * 50 weeks = 750 tunings/year/tuner",
                    "Number of tuners needed: 504,000 / 750 ≈ 672 tuners"
                ],
                "answer": 672
            }}

            # Closing Thoughts
            Please estimate the size of the reference class '{self.__type_of_thing_to_estimate}'
            Remember to only give JSON
            Give an integer as your answer.
            Remember that you should use numbers for normal citations, and ONLY use letters for the citations that your friend gave you.
            """
        )
        model = SmartSearcher(temperature=0)
        estimation = await model.invoke_and_return_verified_type(prompt, dict)

        facts_as_markdown = "\n".join(
            [f"- {fact}" for fact in estimation["facts"]]
        )
        estimation_reasoning = "\n".join(
            [
                f"{i+1}. {step}"
                for i, step in enumerate(estimation["reasoning_steps"])
            ]
        )
        number_of_hits = estimation["answer"]
        estimation_as_markdown = clean_indents(
            f"""
            I estimate that there are {number_of_hits} '{self.__type_of_thing_to_estimate}'.

            **Facts**:
            {facts_as_markdown}

            **Estimation Steps and Assumptions**:
            {estimation_reasoning}
            """
        )
        if not self.__previous_research:
            estimation_as_markdown += (
                f"\n\n**Background Research**: {research_to_use}"
            )
        assert isinstance(number_of_hits, int)
        return EstimationResult(number_of_hits, estimation_as_markdown)
