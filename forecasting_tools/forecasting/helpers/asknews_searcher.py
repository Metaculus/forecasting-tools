import os

from asknews_sdk import AskNewsSDK


class AskNewsSearcher:

    @classmethod
    def get_formatted_news(cls, query: str) -> str:
        """
        Use the AskNews `news` endpoint to get news context for your query.
        The full API reference can be found here: https://docs.asknews.app/en/reference#get-/v1/news/search
        """
        ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
        ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")

        if not ASKNEWS_CLIENT_ID or not ASKNEWS_SECRET:
            raise ValueError("ASKNEWS_CLIENT_ID or ASKNEWS_SECRET is not set")

        ask = AskNewsSDK(
            client_id=ASKNEWS_CLIENT_ID,
            client_secret=ASKNEWS_SECRET,
            scopes=set(["news"]),
        )

        # get the latest news related to the query (within the past 48 hours)
        hot_response = ask.news.search_news(
            query=query,  # your natural language query
            n_articles=6,  # control the number of articles to include in the context, originally 5
            return_type="both",
            strategy="latest news",  # enforces looking at the latest news only
        )

        # get context from the "historical" database that contains a news archive going back to 2023
        historical_response = ask.news.search_news(
            query=query,
            n_articles=10,
            return_type="both",
            strategy="news knowledge",  # looks for relevant news within the past 60 days
        )

        hot_articles = hot_response.as_dicts
        historical_articles = historical_response.as_dicts
        formatted_articles = "Here are the relevant news articles:\n\n"

        if hot_articles:
            hot_articles = [article.__dict__ for article in hot_articles]
            hot_articles = sorted(
                hot_articles, key=lambda x: x["pub_date"], reverse=True
            )

            for article in hot_articles:
                pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
                formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

        if historical_articles:
            historical_articles = [
                article.__dict__ for article in historical_articles
            ]
            historical_articles = sorted(
                historical_articles, key=lambda x: x["pub_date"], reverse=True
            )

            for article in historical_articles:
                pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
                formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

        if not hot_articles and not historical_articles:
            formatted_articles += "No articles were found.\n\n"
            return formatted_articles

        return formatted_articles
