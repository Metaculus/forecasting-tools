from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.auto_optimizers.customizable_bot import CustomizableBot


class ControlPrompt:
    @classmethod
    def get_reasoning_prompt(cls) -> str:
        return _CONTROL_REASONING_PROMPT

    @classmethod
    def get_research_prompt(cls) -> str:
        return _CONTROL_AGENT_RESEARCH_PROMPT

    @classmethod
    def get_combined_prompt(cls) -> str:
        return clean_indents(
            f"""{_CONTROL_AGENT_RESEARCH_PROMPT}
            {CustomizableBot.RESEARCH_REASONING_SPLIT_STRING}
            {_CONTROL_REASONING_PROMPT}
            """
        )

    @classmethod
    def version(cls) -> str:
        return _VERSION


_VERSION = "2025Q2+tools"
_CONTROL_REASONING_PROMPT = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{question_text}

Question background:
{background_info}


This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

{fine_print}


Your research assistant says:
{research}

Today is {today}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A brief description of a scenario that results in a No outcome.
(d) A brief description of a scenario that results in a Yes outcome.

You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""

# _CONTROL_RESEARCH_PROMPT: str = (
#     PresetResearchStrategy.SEARCH_ASKNEWS_WITH_QUESTION_TEXT.value
# )


_CONTROL_AGENT_RESEARCH_PROMPT: str = """
You are an assistant to a superforecaster.
The superforecaster will give you a question they intend to forecast on.
To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
You do not produce forecasts yourself.

Question:
{question_text}

Please make only 1 search using the question text as a query (with Perplexity if available).
Completely restate what the search tool tells you in full without any additional commentary.
Don't use any other tools other than the 1 search with the question text as the query.
"""
