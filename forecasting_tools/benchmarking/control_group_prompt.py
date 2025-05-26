class ControlGroupPrompt:
    @classmethod
    def get_prompt(cls) -> str:
        return _CONTROL_GROUP_PROMPT

    @classmethod
    def version(cls) -> str:
        return _VERSION


_VERSION = "2025Q2"
_CONTROL_GROUP_PROMPT = """
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
