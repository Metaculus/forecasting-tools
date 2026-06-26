"""Base class for the public-baseline (population) forecasting bots.

A ``PopulationBaselineBot`` estimates the forecast that a *specific group of
people* would collectively produce if a randomized, representative sample of
that group were polled on the question. It is deliberately NOT trying to be
accurate about the world; it is trying to faithfully reproduce a group's
collective belief (including that group's biases).

Each concrete bot (public sentiment, expert opinion, credible news outlets,
left, center, right) only needs to define a ``PopulationSpec`` describing who is
being sampled and how to find evidence of their views. All of the agentic
machinery, prediction conversion, and comment formatting lives here.

The bots are built on PydanticAI agents that are each given a single Exa-based
quote-search tool and asked to return a structured ``*PopulationForecast`` object
containing the individual sources they sampled, each source's implied forecast,
and an aggregate implied forecast.

Each question is forecast by several independent agent runs ("branches"), one per
underlying model, mirroring the research-only bot's multi-sample aggregation but
with a different model per branch. The framework then aggregates the branch
forecasts (e.g. median) into the final prediction. Every branch's structured
object is rendered into the Metaculus comment so readers can see exactly which
sources produced which implied forecasts, per model.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

import pendulum
from pydantic import BaseModel, Field
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from forecasting_tools.agents_and_tools.research.exa_quote_searcher import (
    ExaQuoteSearcher,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.forecast_report import (
    ReasonedPrediction,
    ResearchWithPredictions,
)
from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOption,
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    ConditionalQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.official_bots.template_bot_2026_summer import (
    SummerTemplateBot2026,
)
from forecasting_tools.helpers.asknews_searcher import AskNewsSearcher
from forecasting_tools.helpers.metaculus_client import MetaculusClient
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)


class PopulationSpec(BaseModel):
    """Description of the group of people a baseline bot is sampling."""

    name: str
    short_name: str
    target_description: str
    sampling_method: str
    source_guidance: str
    interpretation_guidance: str


class DiscoveredSource(BaseModel):
    """A single source used to approximate one slice of the sampled group."""

    name: str = Field(
        description="Short name of the source, e.g. 'YouGov poll (Jun 2026)', "
        "'Dr. Jane Smith, virologist', or 'The Economist editorial'."
    )
    represents: str = Field(
        description="Which slice of the sampled group this source stands in for."
    )
    url: str | None = Field(
        default=None, description="A link to the source if one is available."
    )
    implied_forecast: str = Field(
        description="The forecast this source implies for THIS exact question, "
        "stated concretely (a probability, an outcome, or a number/range)."
    )
    confidence: str = Field(
        description="How strongly the source implies this forecast: "
        "'low', 'medium', or 'high'."
    )
    note: str = Field(
        description="One sentence on how this source's view was translated into "
        "the implied forecast."
    )


class BinaryPopulationForecast(BaseModel):
    scratchpad: str = Field(
        default="",
        description="Your private working notes: lay out how you sampled the "
        "group, what the evidence shows, and how you reasoned toward the "
        "group's implied probability BEFORE committing to the number.",
    )
    population_summary: str = Field(
        description="2-4 sentences on what the sampled group collectively believes "
        "about this question and why."
    )
    sources: list[DiscoveredSource]
    aggregate_probability: float = Field(
        ge=0,
        le=1,
        description="The group's aggregate implied probability that the question "
        "resolves YES (between 0 and 1).",
    )
    aggregate_rationale: str = Field(
        description="How the individual sources were weighted and combined."
    )


class OptionProbability(BaseModel):
    option_name: str
    probability: float = Field(ge=0, le=1)


class MultipleChoicePopulationForecast(BaseModel):
    scratchpad: str = Field(
        default="",
        description="Your private working notes reasoning toward the group's "
        "implied option probabilities BEFORE committing to numbers.",
    )
    population_summary: str
    sources: list[DiscoveredSource]
    option_probabilities: list[OptionProbability] = Field(
        description="The group's aggregate implied probability for each option. "
        "Use the exact option names provided and make the probabilities sum to ~1."
    )
    aggregate_rationale: str


class NumericPercentile(BaseModel):
    percentile: float = Field(
        ge=0,
        le=1,
        description="A cumulative probability between 0 and 1 (e.g. 0.1 for the "
        "10th percentile).",
    )
    value: float = Field(
        description="The value at this percentile, in the question's units."
    )


class NumericPopulationForecast(BaseModel):
    scratchpad: str = Field(
        default="",
        description="Your private working notes reasoning toward the group's "
        "implied distribution BEFORE committing to percentile values.",
    )
    population_summary: str
    sources: list[DiscoveredSource]
    percentiles: list[NumericPercentile] = Field(
        description="An increasing list of percentile/value pairs describing the "
        "group's aggregate distribution. Include at least the 0.1, 0.2, 0.4, 0.6, "
        "0.8 and 0.9 percentiles with wide intervals."
    )
    aggregate_rationale: str


class DatePercentilePoint(BaseModel):
    percentile: float = Field(ge=0, le=1)
    iso_date: str = Field(
        description="The date at this percentile in ISO format (YYYY-MM-DD)."
    )


class DatePopulationForecast(BaseModel):
    scratchpad: str = Field(
        default="",
        description="Your private working notes reasoning toward the group's "
        "implied date distribution BEFORE committing to percentile dates.",
    )
    population_summary: str
    sources: list[DiscoveredSource]
    percentiles: list[DatePercentilePoint] = Field(
        description="An increasing list of percentile/date pairs describing the "
        "group's aggregate distribution. Include at least the 0.1, 0.2, 0.4, 0.6, "
        "0.8 and 0.9 percentiles with wide intervals."
    )
    aggregate_rationale: str


class PopulationBaselineBot(SummerTemplateBot2026):
    """Estimates what a sampled group of people would collectively forecast.

    Subclasses provide ``population_spec``. The bot uses a PydanticAI agent with
    a single search tool to gather evidence of the group's views, returns a
    structured per-source breakdown, and converts the aggregate into the
    prediction type the framework expects.
    """

    population_spec: PopulationSpec
    # Max model round-trips per branch (a few searches + the final answer).
    _request_limit_per_question: int = 8
    # Hard ceiling on a single branch so a stalled OpenRouter/Exa/AskNews call
    # cannot hang the whole forecast (GeneralLlm.timeout is not wired into the
    # PydanticAI agent, so we enforce it ourselves).
    _agent_run_timeout_seconds: float = 240
    # Cap simultaneous agent runs (each fans out to OpenRouter + Exa) so bursts
    # of branches/questions don't trip OpenRouter "error" responses or Exa rate
    # limits. Shared across all questions and branches of the bot.
    _max_concurrent_agent_runs: int = 2
    _agent_run_limiter: asyncio.Semaphore = asyncio.Semaphore(2)

    def __init__(
        self,
        *,
        branch_llms: list[GeneralLlm] | None = None,
        research_reports_per_question: int = 1,
        use_research_summary_to_forecast: bool = False,
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to: str | None = None,
        skip_previously_forecasted_questions: bool = False,
        llms: dict[str, str | GeneralLlm | None] | None = None,
        enable_summarize_research: bool = False,
        parameters_to_exclude_from_config_dict: list[str] | None = None,
        extra_metadata_in_explanation: bool = False,
        required_successful_predictions: float = 0.5,
        metaculus_client: MetaculusClient | None = None,
    ) -> None:
        self.branch_llms = branch_llms or self._default_branch_llms()
        assert len(self.branch_llms) > 0, "Need at least one branch model"
        super().__init__(
            research_reports_per_question=research_reports_per_question,
            predictions_per_research_report=len(self.branch_llms),
            use_research_summary_to_forecast=use_research_summary_to_forecast,
            publish_reports_to_metaculus=publish_reports_to_metaculus,
            folder_to_save_reports_to=folder_to_save_reports_to,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
            llms=llms,
            enable_summarize_research=enable_summarize_research,
            parameters_to_exclude_from_config_dict=parameters_to_exclude_from_config_dict,
            extra_metadata_in_explanation=extra_metadata_in_explanation,
            required_successful_predictions=required_successful_predictions,
            metaculus_client=metaculus_client,
        )
        self._exa_quote_searcher = ExaQuoteSearcher()

    @staticmethod
    def _default_branch_llms() -> list[GeneralLlm]:
        agent_timeout = 5 * 60
        return [
            GeneralLlm(
                model="openrouter/anthropic/claude-sonnet-4.5",
                temperature=0.3,
                timeout=agent_timeout,
            ),
            GeneralLlm(
                model="openrouter/x-ai/grok-4.3",
                temperature=0.3,
                timeout=agent_timeout,
            ),
            GeneralLlm(
                model="openrouter/z-ai/glm-5.1",
                temperature=0.3,
                timeout=agent_timeout,
            ),
        ]

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        config_dict = super()._llm_config_defaults()
        config_dict["summarizer"] = None
        return config_dict

    async def run_research(self, question: MetaculusQuestion) -> str:
        return ""

    @staticmethod
    def _branch_label(branch_llm: GeneralLlm) -> str:
        return branch_llm.model.removeprefix("openrouter/")

    def _get_agent_model(self, branch_llm: GeneralLlm) -> OpenRouterModel:
        model_name = branch_llm.model.removeprefix("openrouter/")
        provider = OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY"))
        return OpenRouterModel(model_name, provider=provider)

    async def _run_agent(self, agent: Agent, prompt: str) -> object:
        async with self._agent_run_limiter:
            return await asyncio.wait_for(
                agent.run(
                    prompt,
                    usage_limits=UsageLimits(
                        request_limit=self._request_limit_per_question
                    ),
                ),
                timeout=self._agent_run_timeout_seconds,
            )

    async def _search_for_quotes(
        self,
        query: str,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> str:
        try:
            result = await self._exa_quote_searcher.search_for_quotes(
                query, include_domains, exclude_domains
            )
            if result and "No Exa search results" not in result:
                return result
            logger.info(f"Exa returned nothing for '{query}'; trying AskNews.")
        except Exception as exa_error:
            logger.warning(
                f"Exa quote search failed ({exa_error}); falling back to AskNews."
            )
        try:
            return await AskNewsSearcher().get_formatted_news_async(query)
        except Exception as fallback_error:
            logger.warning(f"AskNews fallback failed: {fallback_error}")
            return f"No search results available for '{query}'."

    def _build_agent(
        self, output_type: type[BaseModel], branch_llm: GeneralLlm
    ) -> Agent:
        async def search_for_quotes(
            query: str,
            include_domains: list[str] | None = None,
            exclude_domains: list[str] | None = None,
        ) -> str:
            """Search the web for evidence of what the target group thinks, returning quotable highlights.

            Use focused queries (polls, surveys, expert statements, op-eds,
            articles, social-media sentiment, etc.). For each result you get the
            title, URL, date, an article summary, and the most relevant highlight
            quotes you can cite verbatim. Optionally pass include_domains /
            exclude_domains (e.g. specific outlets) to target a slice of the group.
            """
            return await self._search_for_quotes(
                query, include_domains, exclude_domains
            )

        return Agent(
            self._get_agent_model(branch_llm),
            output_type=output_type,
            system_prompt=self._system_prompt(),
            tools=[search_for_quotes],
            retries=2,
        )

    async def _make_prediction(
        self, question: MetaculusQuestion, research: str
    ) -> ReasonedPrediction[object]:
        notepad = await self._get_notepad(question)
        async with self._note_pad_lock:
            branch_index = notepad.note_entries.get("branch_counter", 0)
            notepad.note_entries["branch_counter"] = branch_index + 1
            notepad.total_predictions_attempted += 1
        branch_llm = self.branch_llms[branch_index % len(self.branch_llms)]

        if isinstance(question, BinaryQuestion):
            return await self._run_forecast_on_binary(question, research, branch_llm)
        if isinstance(question, MultipleChoiceQuestion):
            return await self._run_forecast_on_multiple_choice(
                question, research, branch_llm
            )
        if isinstance(question, NumericQuestion):
            return await self._run_forecast_on_numeric(question, research, branch_llm)
        if isinstance(question, DateQuestion):
            return await self._run_forecast_on_date(question, research, branch_llm)
        if isinstance(question, ConditionalQuestion):
            return await self._run_forecast_on_conditional(question, research)
        raise ValueError(f"Unknown question type: {type(question)}")

    def _system_prompt(self) -> str:
        spec = self.population_spec
        return clean_indents(
            f"""
            You are a careful research analyst. Your job is to estimate the forecast
            that {spec.name} would collectively give for a specific question. You are
            NOT estimating what is most likely to actually happen, and you are NOT
            giving your own opinion.

            Concretely, approximate the result of taking a RANDOMIZED, REPRESENTATIVE
            sample of {spec.target_description}, asking each member to forecast the
            question, and aggregating their answers.

            How to sample this group:
            {spec.sampling_method}

            Finding evidence:
            Use the `search_for_quotes` tool (2-5 focused searches) to find concrete
            evidence of what this group currently believes about the question or its
            close neighbors. {spec.source_guidance}

            For every source you use, record: its name, which slice of the group it
            represents, its URL (if any), and the forecast it IMPLIES for THIS exact
            question. Translate vague sentiment, narratives, polling, or commentary
            into a concrete implied forecast that matches the question's resolution
            criteria. {spec.interpretation_guidance}

            Then aggregate across the sources you sampled, weighting each by how
            representative it is of {spec.target_description}, to get the group's
            aggregate implied forecast.

            Rules:
            - Faithfully represent this group, including its biases and blind spots,
              even if you personally believe they are wrong.
            - Ground your estimate in evidence you actually find. If evidence is thin,
              sample more broadly and reason explicitly about how this group tends to
              think about questions like this.
            - Try to include 3-8 distinct sources.
            - Keep it efficient: a few targeted searches, then answer.
            """
        )

    def _question_block(self, question: MetaculusQuestion) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return clean_indents(
            f"""
            Question: {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            {question.fine_print}

            Today's date: {today}
            """
        )

    async def _run_forecast_on_binary(
        self,
        question: BinaryQuestion,
        research: str,
        branch_llm: GeneralLlm | None = None,
    ) -> ReasonedPrediction[float]:
        branch_llm = branch_llm or self.branch_llms[0]
        agent = self._build_agent(BinaryPopulationForecast, branch_llm)
        prompt = clean_indents(
            f"""
            {self._question_block(question)}

            Estimate what {self.population_spec.name} would forecast as the probability
            that this question resolves YES.
            """
        )
        result = await self._run_agent(agent, prompt)
        forecast: BinaryPopulationForecast = result.output
        probability = max(0.01, min(0.99, forecast.aggregate_probability))
        reasoning = self._format_reasoning(
            branch_llm,
            forecast.scratchpad,
            forecast.population_summary,
            forecast.sources,
            forecast.aggregate_rationale,
            f"Aggregate implied probability of YES: {probability:.1%}",
        )
        return ReasonedPrediction(prediction_value=probability, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self,
        question: MultipleChoiceQuestion,
        research: str,
        branch_llm: GeneralLlm | None = None,
    ) -> ReasonedPrediction[PredictedOptionList]:
        branch_llm = branch_llm or self.branch_llms[0]
        agent = self._build_agent(MultipleChoicePopulationForecast, branch_llm)
        options_str = ", ".join(f'"{option}"' for option in question.options)
        prompt = clean_indents(
            f"""
            {self._question_block(question)}

            The allowed options are: [{options_str}]

            Estimate what {self.population_spec.name} would forecast as the probability
            of each option. Use the exact option names above and make the
            probabilities sum to approximately 1.
            """
        )
        result = await self._run_agent(agent, prompt)
        forecast: MultipleChoicePopulationForecast = result.output
        predicted_options = self._build_option_list(
            question, forecast.option_probabilities
        )
        final_line = "Aggregate implied probabilities: " + ", ".join(
            f"{option.option_name}: {option.probability:.1%}"
            for option in predicted_options.predicted_options
        )
        reasoning = self._format_reasoning(
            branch_llm,
            forecast.scratchpad,
            forecast.population_summary,
            forecast.sources,
            forecast.aggregate_rationale,
            final_line,
        )
        return ReasonedPrediction(
            prediction_value=predicted_options, reasoning=reasoning
        )

    @staticmethod
    def _build_option_list(
        question: MultipleChoiceQuestion,
        option_probabilities: list[OptionProbability],
    ) -> PredictedOptionList:
        lookup = {
            option.option_name.strip().lower(): option.probability
            for option in option_probabilities
        }
        raw_probabilities = [
            max(0.0, lookup.get(option.strip().lower(), 0.0))
            for option in question.options
        ]
        if sum(raw_probabilities) <= 0:
            raw_probabilities = [1.0 for _ in question.options]
        total = sum(raw_probabilities)
        normalized = [probability / total for probability in raw_probabilities]
        return PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=option, probability=probability)
                for option, probability in zip(question.options, normalized)
            ]
        )

    async def _run_forecast_on_numeric(
        self,
        question: NumericQuestion,
        research: str,
        branch_llm: GeneralLlm | None = None,
    ) -> ReasonedPrediction[NumericDistribution]:
        branch_llm = branch_llm or self.branch_llms[0]
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        agent = self._build_agent(NumericPopulationForecast, branch_llm)
        prompt = clean_indents(
            f"""
            {self._question_block(question)}

            Units for the answer: {question.unit_of_measure if question.unit_of_measure else "infer the appropriate units"}
            {lower_bound_message}
            {upper_bound_message}

            Estimate what {self.population_spec.name} would forecast for this number.
            Provide an increasing list of percentile/value pairs (at least the 0.1,
            0.2, 0.4, 0.6, 0.8 and 0.9 percentiles) with wide intervals to reflect
            the group's uncertainty.
            """
        )
        result = await self._run_agent(agent, prompt)
        forecast: NumericPopulationForecast = result.output
        percentiles = self._build_percentiles(forecast.percentiles)
        distribution = NumericDistribution.from_question(percentiles, question)
        final_line = "Aggregate implied distribution (percentile: value): " + ", ".join(
            f"{int(percentile.percentile * 100)}%: {percentile.value:g}"
            for percentile in percentiles
        )
        reasoning = self._format_reasoning(
            branch_llm,
            forecast.scratchpad,
            forecast.population_summary,
            forecast.sources,
            forecast.aggregate_rationale,
            final_line,
        )
        return ReasonedPrediction(prediction_value=distribution, reasoning=reasoning)

    async def _run_forecast_on_date(
        self,
        question: DateQuestion,
        research: str,
        branch_llm: GeneralLlm | None = None,
    ) -> ReasonedPrediction[NumericDistribution]:
        branch_llm = branch_llm or self.branch_llms[0]
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        agent = self._build_agent(DatePopulationForecast, branch_llm)
        prompt = clean_indents(
            f"""
            {self._question_block(question)}

            {lower_bound_message}
            {upper_bound_message}

            Estimate what {self.population_spec.name} would forecast for this date.
            Provide an increasing list of percentile/date pairs (at least the 0.1,
            0.2, 0.4, 0.6, 0.8 and 0.9 percentiles), dates in YYYY-MM-DD format, with
            wide intervals to reflect the group's uncertainty.
            """
        )
        result = await self._run_agent(agent, prompt)
        forecast: DatePopulationForecast = result.output
        percentiles = self._build_date_percentiles(forecast.percentiles)
        distribution = NumericDistribution.from_question(percentiles, question)
        final_line = "Aggregate implied dates (percentile: date): " + ", ".join(
            f"{int(percentile.percentile * 100)}%: "
            f"{datetime.fromtimestamp(percentile.value, tz=timezone.utc).date().isoformat()}"
            for percentile in percentiles
        )
        reasoning = self._format_reasoning(
            branch_llm,
            forecast.scratchpad,
            forecast.population_summary,
            forecast.sources,
            forecast.aggregate_rationale,
            final_line,
        )
        return ReasonedPrediction(prediction_value=distribution, reasoning=reasoning)

    @staticmethod
    def _build_percentiles(points: list[NumericPercentile]) -> list[Percentile]:
        percentiles = [
            Percentile(
                percentile=(
                    point.percentile
                    if point.percentile <= 1
                    else point.percentile / 100
                ),
                value=point.value,
            )
            for point in points
        ]
        return sorted(percentiles, key=lambda percentile: percentile.percentile)

    @staticmethod
    def _build_date_percentiles(points: list[DatePercentilePoint]) -> list[Percentile]:
        percentiles = [
            Percentile(
                percentile=(
                    point.percentile
                    if point.percentile <= 1
                    else point.percentile / 100
                ),
                value=pendulum.parse(point.iso_date).timestamp(),
            )
            for point in points
        ]
        return sorted(percentiles, key=lambda percentile: percentile.percentile)

    def _format_reasoning(
        self,
        branch_llm: GeneralLlm,
        scratchpad: str,
        population_summary: str,
        sources: list[DiscoveredSource],
        aggregate_rationale: str,
        final_line: str,
    ) -> str:
        spec = self.population_spec
        model_label = self._branch_label(branch_llm)
        table_rows = []
        for index, source in enumerate(sources, start=1):
            source_cell = (
                f"[{source.name}]({source.url})" if source.url else source.name
            )
            table_rows.append(
                f"| {index} | {source_cell} | {source.represents} | "
                f"{source.implied_forecast} | {source.confidence} | {source.note} |"
            )
        table = (
            "\n".join(table_rows)
            if table_rows
            else "| - | (no sources found) | - | - | - | - |"
        )
        max_scratchpad_chars = 2000
        trimmed_scratchpad = (scratchpad or "").strip()
        if len(trimmed_scratchpad) > max_scratchpad_chars:
            trimmed_scratchpad = (
                trimmed_scratchpad[:max_scratchpad_chars].rstrip() + "…"
            )
        scratchpad_section = (
            f"### Reasoning scratchpad\n{trimmed_scratchpad}\n\n"
            if trimmed_scratchpad
            else ""
        )
        return clean_indents(
            f"""
            ## {model_label} — what {spec.name} appears to forecast

            {population_summary}

            {scratchpad_section}### Sources sampled and their implied forecasts
            | # | Source | Represents | Implied forecast | Confidence | Note |
            |---|--------|-----------|------------------|-----------|------|
            {table}

            ### Aggregation
            {aggregate_rationale}

            **{final_line}**
            """
        )

    def _create_comment(
        self,
        question: MetaculusQuestion,
        research_prediction_collections: list[ResearchWithPredictions],
        aggregated_prediction,
        final_cost: float,
        time_spent_in_minutes: float,
    ) -> str:
        report_type = DataOrganizer.get_report_type_for_question_type(type(question))
        readable_prediction = report_type.make_readable_prediction(
            aggregated_prediction
        )
        spec = self.population_spec
        breakdowns = []
        for collection in research_prediction_collections:
            for prediction in collection.predictions:
                breakdowns.append(prediction.reasoning)
        combined_breakdowns = "\n\n---\n\n".join(breakdowns)
        branch_models = ", ".join(
            self._branch_label(branch_llm) for branch_llm in self.branch_llms
        )
        comment = clean_indents(
            f"""
            # {spec.name.upper()} BASELINE FORECAST
            *Question*: {question.question_text}
            *Estimated forecast of the {spec.short_name}*: {readable_prediction}
            *What this estimates*: the forecast a randomized, representative sample of
            {spec.target_description} would give if asked this question. This is a
            public-baseline proxy, not a best-guess of the true outcome.
            *Bot Name*: {self.__class__.__name__}
            *Model branches (aggregated)*: {branch_models}

            This forecast is the aggregate of {len(self.branch_llms)} independent agent
            runs, one per model above. Each run searched for evidence of what
            {spec.short_name} believes (via Exa quote search) and recorded the sources
            it sampled and the forecast each implies. The per-model breakdowns below
            show those sources and implied forecasts; the headline figure is the
            aggregate across all branches.

            {combined_breakdowns}

            ---
            *Note*: This is an experimental, low-cost agentic baseline bot built on
            PydanticAI. Cost/time metadata is not tracked for the agent's calls and is
            therefore omitted.
            """
        )
        max_comment_size = 150000
        if len(comment) > max_comment_size:
            comment = (
                comment[:2000]
                + "\n\n---\n\n The comment size exceeded max size and has been truncated"
            )
        return comment
