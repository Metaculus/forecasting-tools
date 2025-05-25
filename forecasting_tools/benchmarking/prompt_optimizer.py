import asyncio
import logging
from dataclasses import dataclass

from pydantic import BaseModel

from forecasting_tools.agents_and_tools.misc_tools import perplexity_pro_search
from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AgentSdkLlm,
    AiAgent,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.benchmarking.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.benchmarking.benchmarker import Benchmarker
from forecasting_tools.benchmarking.customizable_bot import CustomizableBot
from forecasting_tools.benchmarking.question_research_snapshot import (
    QuestionResearchSnapshot,
    ResearchType,
)
from forecasting_tools.forecast_helpers.structure_output import (
    structure_output,
)

logger = logging.getLogger(__name__)


CONTROL_GROUP_PROMPT = """
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


class QuestionIdea(BaseModel):
    short_name: str
    idea: str


@dataclass
class PromptConfig:
    prompt_template: str
    llm: GeneralLlm
    original_idea: QuestionIdea
    research_reports_per_question: int = 1
    predictions_per_research_report: int = 5


@dataclass
class EvaluatedPrompt:
    prompt_config: PromptConfig
    benchmark: BenchmarkForBot

    @property
    def score(self) -> float:
        return self.benchmark.average_expected_baseline_score

    @property
    def prompt_text(self) -> str:
        return self.prompt_config.prompt_template


@dataclass
class OptimizationResult:
    evaluated_prompts: list[EvaluatedPrompt]

    @property
    def best_prompt(self) -> EvaluatedPrompt:
        sorted_evaluated_prompts = sorted(
            self.evaluated_prompts, key=lambda x: x.score, reverse=True
        )
        return sorted_evaluated_prompts[0]

    @property
    def best_prompt_text(self) -> str:
        return self.best_prompt.prompt_text


class PromptOptimizer:
    """
    A class to optimize a prompt for a given set of evaluation questions.
    """

    def __init__(
        self,
        evaluation_questions: list[QuestionResearchSnapshot],
        num_prompts_to_try: int,
        forecast_llm: GeneralLlm,
        ideation_llm_name: str,
        evaluation_batch_size: int = 100,
        file_or_folder_to_save_benchmarks: str | None = None,
    ) -> None:
        self.evaluation_questions = evaluation_questions
        self.num_prompts_to_try = num_prompts_to_try
        self.forecast_llm = forecast_llm
        self.ideation_llm_name = ideation_llm_name
        self.research_type = ResearchType.ASK_NEWS_SUMMARIES
        self.concurrent_evaluation_batch_size = evaluation_batch_size
        self.file_or_folder_to_save_benchmarks = (
            file_or_folder_to_save_benchmarks
        )

    async def create_optimized_prompt(
        self,
    ) -> OptimizationResult:
        ideas = await self.get_prompt_ideas()
        tasks = [self.prompt_idea_to_prompt_string(idea) for idea in ideas]
        prompt_templates = await asyncio.gather(*tasks)
        configs = []
        for prompt, idea in zip(prompt_templates, ideas):
            configs.append(
                PromptConfig(
                    prompt_template=prompt,
                    llm=self.forecast_llm,
                    original_idea=idea,
                )
            )
        control_group_config = PromptConfig(
            prompt_template=CONTROL_GROUP_PROMPT,
            llm=self.forecast_llm,
            original_idea=QuestionIdea(
                short_name="Control Group",
                idea="The control group is a group of questions that are not optimized for the prompt. It is used to evaluate the performance of the optimized prompt.",
            ),
        )
        configs.append(control_group_config)
        evaluated_prompts = await self.evaluate_prompts(configs)
        sorted_evaluated_prompts = sorted(
            evaluated_prompts, key=lambda x: x.score, reverse=True
        )
        return OptimizationResult(evaluated_prompts=sorted_evaluated_prompts)

    async def get_prompt_ideas(self) -> list[QuestionIdea]:
        agent = AiAgent(
            name="Prompt Ideator",
            model=AgentSdkLlm(self.ideation_llm_name),
            instructions=clean_indents(
                f"""
                Please come up with {self.num_prompts_to_try} prompt ideas that asks a bot to forecast binary questions.
                There must be a final binary float given at the end, make sure to request for this.

                Consider general forecasting principles that are used in superforecasting.
                Give your ideas in a list format.

                If you need to, run up to 10 searches finding unique ways to approach the prompt.

                Return a list of ideas in the format:
                **1-6 word title**
                Idea 1

                **1-6 word title**
                Idea 2

                **1-6 word title**
                Idea 3
                """
            ),
            tools=[perplexity_pro_search],
        )
        output = await AgentRunner.run(
            agent, f"Please generate {self.num_prompts_to_try} prompt ideas"
        )
        ideas = await structure_output(output.final_output, list[QuestionIdea])
        logger.info(f"Generated {len(ideas)} prompt ideas: {ideas}")
        return ideas

    async def prompt_idea_to_prompt_string(
        self, prompt_idea: QuestionIdea
    ) -> str:
        agent = AiAgent(
            name="Prompt Implementor",
            model=AgentSdkLlm(self.ideation_llm_name),
            instructions=clean_indents(
                f"""
                You need to implement a prompt that asks a bot to forecast binary questions.
                There must be a final binary float given at the end, make sure to request for this.

                The prompt should implement the below idea:
                Name: {prompt_idea.short_name}
                Idea: {prompt_idea.idea}

                This is a template prompt, and so you should add the following variables to the prompt:
                {{question_text}}
                {{background_info}}
                {{resolution_criteria}}
                {{fine_print}}
                {{today}}
                {{research}}

                """
            ),
        )
        output = await AgentRunner.run(
            agent,
            "Please implement a prompt. Return the prompt and nothing but the prompt. The prompt will be run as is",
        )
        prompt = output.final_output
        logger.info(f"Generated prompt:\n{prompt}")
        return prompt

    async def evaluate_prompts(
        self, configurations: list[PromptConfig]
    ) -> list[EvaluatedPrompt]:
        bots = []
        for config in configurations:
            if config.research_reports_per_question != 1:
                raise NotImplementedError(
                    "Currently only supports one research report per question"
                )
            bot = CustomizableBot(
                prompt=config.prompt_template,
                research_snapshots=self.evaluation_questions,
                research_type=self.research_type,
                research_reports_per_question=config.research_reports_per_question,
                predictions_per_research_report=config.predictions_per_research_report,
                llms={
                    "default": config.llm,
                },
                publish_reports_to_metaculus=False,
                enable_summarize_research=False,
            )
            bots.append(bot)
        benchmarker = Benchmarker(
            forecast_bots=bots,
            questions_to_use=[
                snapshot.question for snapshot in self.evaluation_questions
            ],
            concurrent_question_batch_size=self.concurrent_evaluation_batch_size,
            file_path_to_save_reports=self.file_or_folder_to_save_benchmarks,
        )
        benchmarks = await benchmarker.run_benchmark()
        evaluated_prompts: list[EvaluatedPrompt] = []
        for config, benchmark in zip(configurations, benchmarks):
            benchmark.code = config.prompt_template
            benchmark.forecast_bot_class_name = (
                config.original_idea.short_name.replace(" ", "_")
            )
            evaluated_prompts.append(
                EvaluatedPrompt(prompt_config=config, benchmark=benchmark)
            )
        benchmarker.save_benchmarks_to_file_if_configured(benchmarks)
        return evaluated_prompts
