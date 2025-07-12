from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.auto_optimizers.customizable_bot import (
    CustomizableBot,
    PresetResearchStrategy,
)
from forecasting_tools.auto_optimizers.prompt_data_models import ResearchTool


class ControlPrompt:
    """
    Reasoning prompt is used for reasoning

    The research strategy is a string that can be matched to a default research strategy

    The combined prompt is the full prompt that is used for the bot

    The seed prompts is what should be given as initial examples for prompt optimization
    (e.g. we don't want the LLM to think that a simple string is a good prompt)
    """

    @classmethod
    def get_reasoning_prompt(cls) -> str:
        return _CONTROL_REASONING_PROMPT

    @classmethod
    def get_control_research_prompt(cls) -> str:
        return _CONTROL_RESEARCH_PROMPT

    @classmethod
    def get_control_combined_prompt(cls) -> str:
        return clean_indents(
            f"""{_CONTROL_RESEARCH_PROMPT}
            {CustomizableBot.RESEARCH_REASONING_SPLIT_STRING}
            {_CONTROL_REASONING_PROMPT}
            """
        )

    @classmethod
    def get_control_research_tools(cls) -> list[ResearchTool]:
        return []

    @classmethod
    def get_seed_research_prompt(cls) -> str:
        return _SEED_RESEARCH_PROMPT

    @classmethod
    def get_seed_combined_prompt(cls) -> str:
        return clean_indents(
            f"""{_SEED_RESEARCH_PROMPT}
            {CustomizableBot.RESEARCH_REASONING_SPLIT_STRING}
            {_CONTROL_REASONING_PROMPT}
            """
        )

    @classmethod
    def version(cls) -> str:
        return _VERSION


_VERSION = "2025Q2"
_CONTROL_REASONING_PROMPT = """You are a professional forecaster interviewing for a job.

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

_CONTROL_RESEARCH_PROMPT: str = (
    PresetResearchStrategy.SEARCH_ASKNEWS_WITH_QUESTION_TEXT.value
)


_SEED_RESEARCH_PROMPT: str = """Please make only 1 search using the tools you have available (always default to AskNews if available).

Use the query: {question_text}

Completely restate what the search tool tells you in full without any additional commentary.
Don't use any other tools other than the 1 search with the question text as the query.
Don't use Metaculus community prediction as a source. Metaculus is disallowed.
"""
