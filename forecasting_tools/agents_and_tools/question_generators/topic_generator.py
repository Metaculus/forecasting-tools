import asyncio
import logging
import random
from datetime import datetime

from pydantic import BaseModel

from forecasting_tools.agents_and_tools.research.smart_searcher import SmartSearcher
from forecasting_tools.ai_models.agent_wrappers import agent_tool
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.helpers.asknews_searcher import AskNewsSearcher
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)


class OrgInfo(BaseModel):
    symbol: str
    name: str
    url: str
    overview: str


class TopicGenerator:
    LANGUAGES = [
        "en_US",
        "ja_JP",
        "de_DE",
        "en_GB",
        "fr_FR",
        "es_ES",
        "it_IT",
        "pt_BR",
        "ru_RU",
        "zh_CN",
        "ar_EG",
        "hi_IN",
        "ko_KR",
    ]

    @classmethod
    async def generate_random_topic(
        cls,
        model: GeneralLlm | SmartSearcher | str = "openrouter/openai/gpt-4o",
        number_of_topics: int = 10,
        additional_instructions: str = "",
    ) -> list[str]:
        from faker import Faker

        if isinstance(model, str):
            model = GeneralLlm(model=model, temperature=1, timeout=40)

        fake = Faker(cls.LANGUAGES)

        random_text = clean_indents(
            f"""
            Job: {fake.job()}
            Country 1: {fake.country()}
            Country 2: {fake.country()}
            State/Province (not necessarily in above country): {fake.state()}
            City (not necessarily in above state): {fake.city()}
            Word: {fake.word()}
            Sentence: {fake.sentence()}
            Paragraph: {fake.paragraph()}
            Text: {fake.text(max_nb_chars=50)}
            News headline: {fake.sentence().rstrip('.')}
            Company ticker symbol: {fake.lexify(text='???', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
            """
        )

        prompt = clean_indents(
            f"""
            # Instructions
            Using the ideas below (some of which are abstract or randomly generated)
            come up with {number_of_topics} topics for a forecasting questions about the future.
            These will be used to make questions for superforecasters
            Make sure all ideas come from the material below (do not copy the initial ideas)
            Make sure to put everything in English (except for proper nouns)

            Try to choose something interesting and meaningful.

            {additional_instructions}

            Return your response as a list of dictionaries with a "topic" key.

            # Example result w/o citations:
            ```json
            [
                {{"topic": "Lithuanian politics"}},
                {{"topic": "Gun violence in Olklahoma"}},
                {{"topic": "News on Japenese elections"}},
                {{"topic": "Sports results and news in Russian Hockey"}},
                {{"topic": "Number of new houses built in the US"}},
                {{"topic": "Mining jobs in Canada and related news"}},
                {{"topic": "News related to company with ticker symbol XYZ"}},
            ]
            ```

            # Example result w/ citations:
            ```json
            [
                {{"topic": "March Madness", "citations": "[1] [2] [7]"}},
                {{"topic": "Japenese Obon Festival", "citations": "[3] [8] [9]"}},
                {{"topic": "National Hocky tournament in Russia", "citations": "[4] [11] [12]"}},
                {{"topic": "Current Housing Crisis in Oklahoma", "citations": "[13] [14] [15]"}},
                {{"topic": "Corona Outbreak in Europe", "citations": "[13] [14] [15]"}},
                {{"topic": "Recent AI initiative of the TransFord Institute", "citations": "[19] [20] [21]"}},
            ]
            ```

            # Material to adapt:
            {random_text}

            Now please generate a list of topics (in json format) that could be interesting and meaningful.
            """
        )

        topic_dicts = await model.invoke_and_return_verified_type(
            prompt, list[dict[str, str]]
        )
        final_topics = []
        for topic in topic_dicts:
            text = topic["topic"]
            citations = topic.get("citations", "")
            final_topics.append(f"{text} {citations}")

        return final_topics

    @classmethod
    async def generate_random_news_items(
        cls,
        model: GeneralLlm | str = "openrouter/openai/gpt-4o",
        number_of_items: int = 10,
    ) -> list[str]:
        num_topics = 2
        num_news_items_per_topic = number_of_items // num_topics

        topics = await cls.generate_random_topic(
            model=model,
            additional_instructions=(
                "Pick topics related to breaking news"
                " (e.g. if your material is related to basketball"
                " and march madness is happening choose this as a topic)."
                " Add citations to show the topic is recent and relevant."
                " Consider searching for 'latest news in <place>' or 'news related to <upcoming holidays/tournaments/events>'."
                f" Today is {datetime.now().strftime('%Y-%m-%d')} if you already know of something specific in an area to find juice."
            ),
            number_of_topics=num_topics,
        )

        results = await asyncio.gather(
            *[
                cls.topic_to_news_item(
                    topic,
                    number_of_items=num_news_items_per_topic,
                    model=model,
                )
                for topic in topics
            ]
        )
        news_items = []
        for topic_results in results:
            news_items.extend(topic_results)

        return news_items

    @classmethod
    async def topic_to_news_item(
        cls,
        topic: str,
        number_of_items: int = 5,
        model: GeneralLlm | str = "openrouter/openai/gpt-4o",
    ) -> list[str]:
        if isinstance(model, str):
            model = GeneralLlm(model=model, temperature=1, timeout=40)

        ask_news_results = await AskNewsSearcher().get_formatted_news_async(topic)
        prompt = clean_indents(
            f"""
            # Instructions
            Please extract {number_of_items} news items from the following text:

            Return your response as a list of strings with the related url

            # Example result:
            ```json
            [
                {{"topic": "Senator Joe Shmoe joins the race for presidency in the US", "url": "https://www.nyt.com/breaking-news/us/senator-joe-shmoe-joins-the-race-for-presidency"}},
                {{"topic": "Russia attacks Ukraine with new drone technology", "url": "https://www.bbc.com/events/russia-attacks-ukraine"}},
                {{"topic": "Nicuragua start first nuclear power plant", "url": "https://www.nuclearnews.com/events/nicuragua-starts-first-nuclear-power-plant"}},
                {{"topic": "Deadly outbreak of disease spreading through Europe", "url": "https://www.outbreaknews.com/events/deadly-outbreak-of-disease-spreading-through-europe"}},
                {{"topic": "Chinese officials visit Lebanon to discuss trade", "url": "https://www.tradeinkorea.com/events/chinese-officials-visit-lebanon-to-discuss-trade"}},
            ]
            ```

            # Search results
            {ask_news_results}

            # Final instructions
            Now return a json list of topics please
            """
        )
        topic_dicts = await model.invoke_and_return_verified_type(
            prompt, list[dict[str, str]]
        )
        topics = [
            f"{topic_dict['topic']} [link]({topic_dict['url']})"
            for topic_dict in topic_dicts
        ]
        return topics

    @classmethod
    async def get_news_on_random_company(
        cls,
        model: GeneralLlm | str = "gpt-4o",
        search_model: GeneralLlm | str = "openrouter/perplexity/sonar-pro",
    ) -> tuple[OrgInfo, list[str]]:
        from faker import Faker

        fake = Faker(cls.LANGUAGES)

        use_4_letters = random.random() < 0.5
        if use_4_letters:
            ticker_symbol = fake.lexify(
                text="????", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            )
        else:
            ticker_symbol = fake.lexify(
                text="???", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            )
        prompt = clean_indents(
            f"""
            Please figure out what org/entity the ticker symbol {ticker_symbol} is for (or if it doesn't exist an org with the closest matching ticker).
            Return the name of the org/entity, a relevant url, the chosen ticker symbol, and a short description of the org/entity.
            """
        )
        response = await GeneralLlm.to_llm(search_model).invoke(prompt)
        company = await structure_output(response, OrgInfo)
        logger.info(f"Company info: {company}")

        topics = await cls.topic_to_news_item(
            topic=f"News on company '{company.name}' with ticker symbol {company.symbol}",
            model=model,
            number_of_items=10,
        )
        return company, topics

    @agent_tool
    @staticmethod
    def find_random_headlines_tool() -> str:
        """
        By making search terms from a list of random things (random country, industry, etc), finds a list of 10 random headlines to help come up with ideas for questions to forecast.
        Output: List of 10 topics that include links to the source.
        """
        number_of_items = 10
        topics = asyncio.run(
            TopicGenerator.generate_random_news_items(number_of_items=number_of_items)
        )
        topic_list = ""
        for topic in topics:
            topic_list += f"- {topic}\n"
        return topic_list

    @agent_tool
    @staticmethod
    def get_headlines_on_random_company_tool() -> str:
        """
        By picking a randomly generated Ticker symbol, finds a list of news items on a company.
        """
        company, topics = asyncio.run(TopicGenerator.get_news_on_random_company())
        company_info = f"Company: {company.name} ({company.symbol})\n"
        company_info += f"Overview: {company.overview}\n"
        company_info += f"URL: {company.url}\n"
        topic_list = ""
        for topic in topics:
            topic_list += f"- {topic}\n"
        return f"{company_info}\n{topic_list}"
