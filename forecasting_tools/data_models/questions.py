from __future__ import annotations

import copy
import logging
import textwrap
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

import pendulum
import typeguard
from pydantic import AliasChoices, BaseModel, Field, model_validator

from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.timestamped_predictions import (
    BinaryTimestampedPrediction,
    NumericTimestampedDistribution,
    TimeStampedPrediction,
)
from forecasting_tools.util.jsonable import Jsonable
from forecasting_tools.util.misc import (
    add_timezone_to_dates_in_base_model,
    clean_indents,
)

logger = logging.getLogger(__name__)


class QuestionState(Enum):
    UPCOMING = "upcoming"
    OPEN = "open"
    RESOLVED = "resolved"
    CLOSED = "closed"


class CanceledResolution(Enum):
    ANNULLED = "annulled"
    AMBIGUOUS = "ambiguous"


class OutOfBoundsResolution(Enum):
    ABOVE_UPPER_BOUND = "above_upper_bound"
    BELOW_LOWER_BOUND = "below_lower_bound"
    # NOTE: Sometimes the numeric resolution is a also number that is above/below bounds. OutOfBoundsResolution should be used when the resolution is known to be out of bounds but its exact value is unknown.


BinaryResolution = bool | CanceledResolution
NumericResolution = float | CanceledResolution | OutOfBoundsResolution
DateResolution = datetime | CanceledResolution | OutOfBoundsResolution
MultipleChoiceResolution = str | CanceledResolution
ResolutionType = (
    BinaryResolution | NumericResolution | DateResolution | MultipleChoiceResolution
)

QuestionBasicType = Literal[
    "binary", "numeric", "multiple_choice", "date", "discrete", "conditional"
]

ConditionalSubQuestionType = Literal["parent", "child", "yes", "no"]


class Category(BaseModel, Jsonable):
    id: int
    name: str
    slug: str | None = None
    emoji: str | None = None
    description: str | None = None
    type: Literal["category"] = "category"


class MetaculusQuestion(BaseModel, Jsonable):
    question_text: str
    id_of_post: int | None = Field(
        default=None,
        validation_alias=AliasChoices("question_id", "post_id", "id_of_post"),
    )  # Posts can contain multiple questions (e.g. group questions or conditionals), which is why id_of_post is separate from id_of_question (id_of_post was originally misnamed as question_id). Post IDs are what is used in the URL.
    page_url: str | None = None
    id_of_question: int | None = None
    state: QuestionState | None = None
    num_forecasters: int | None = None
    num_predictions: int | None = None
    resolution_criteria: str | None = None
    fine_print: str | None = None
    background_info: str | None = None
    unit_of_measure: str | None = None  # TODO: Move this field to numeric questions
    close_time: datetime | None = (
        None  # Time that the question was closed to new forecasts
    )
    actual_resolution_time: datetime | None = None
    scheduled_resolution_time: datetime | None = None
    published_time: datetime | None = (
        None  # Time that the question was visible on the site
    )
    open_time: datetime | None = (
        None  # Time the question was able to be forecasted on by individuals
    )
    date_accessed: datetime = Field(default_factory=pendulum.now)
    already_forecasted: bool | None = None
    tournament_slugs: list[str] = Field(default_factory=list)
    default_project_id: int | None = None
    includes_bots_in_aggregates: bool | None = None
    cp_reveal_time: datetime | None = None  # Community Prediction Reveal Time
    question_weight: float | None = None
    resolution_string: str | None = None
    conditional_type: ConditionalSubQuestionType | None = None
    previous_forecasts: list[TimeStampedPrediction] | None = None
    group_question_option: str | None = (
        None  # For group questions like "How many people will die of coronovirus in the following periouds" it would be "September 2024", "All of 2025", etc
    )
    question_ids_of_group: list[int] | None = None
    api_json: dict = Field(
        description=(
            "The API JSON response used to create the question. "
            "For group questions, a fake 'question' entry may be made to help with group question expansion "
            "(full group question info is under 'group_of_questions')"
        ),
        default_factory=dict,
    )
    custom_metadata: dict = Field(
        default_factory=dict
    )  # Additional metadata not tracked above or through the Metaculus API
    categories: list[Category] = Field(default_factory=list)

    @model_validator(mode="after")
    def add_timezone_to_dates(self) -> MetaculusQuestion:
        return add_timezone_to_dates_in_base_model(self)

    @classmethod
    def from_metaculus_api_json(cls, post_api_json: dict) -> MetaculusQuestion:
        post_id = post_api_json["id"]
        logger.debug(f"Processing Post ID {post_id}")

        question_json: dict = post_api_json["question"]
        json_state = question_json["status"]
        question_state = QuestionState(json_state)

        try:
            history = question_json["my_forecasts"]["history"]
            is_forecasted = history is not None and len(history) > 0
        except Exception:
            is_forecasted = False

        try:
            tournaments: list[dict] = post_api_json["projects"]["tournament"]  # type: ignore
            tournament_slugs = [str(t["slug"]) for t in tournaments]
        except KeyError:
            try:
                question_series: list[dict] = post_api_json["projects"]["question_series"]  # type: ignore
                tournament_slugs = [str(q["slug"]) for q in question_series]
            except KeyError:
                tournament_slugs = []

        group_question_option = question_json.get("label", None)
        if group_question_option is not None and group_question_option.strip() == "":
            group_question_option = None

        try:
            categories_dict_list = post_api_json["projects"]["category"]
        except KeyError:
            categories_dict_list = []
        categories = [
            Category(**category_dict) for category_dict in categories_dict_list
        ]

        published_time = cls._parse_api_date(post_api_json.get("published_at"))

        question = MetaculusQuestion(
            # NOTE: Reminder - When adding new fields, consider if group questions
            #       need to be parsed differently (i.e. if the field information is part of the post_json)
            #       Also, anything that filters on the question level needs a local filter added to MetaculusClient (since the site does not filter subquestions)
            state=question_state,
            question_text=question_json["title"],
            id_of_post=post_id,
            id_of_question=question_json["id"],
            background_info=question_json.get("description", None),
            fine_print=question_json.get("fine_print", None),
            resolution_criteria=question_json.get("resolution_criteria", None),
            unit_of_measure=question_json.get("unit", None),
            page_url=f"https://www.metaculus.com/questions/{post_id}",
            num_forecasters=post_api_json.get("nr_forecasters", None),
            num_predictions=post_api_json.get("forecasts_count", None),
            close_time=cls._parse_api_date(question_json.get("scheduled_close_time")),
            actual_resolution_time=cls._parse_api_date(
                question_json.get("actual_resolve_time")
            ),
            scheduled_resolution_time=cls._parse_api_date(
                question_json.get("scheduled_resolve_time")
            ),
            published_time=published_time,
            cp_reveal_time=cls._parse_api_date(question_json.get("cp_reveal_time")),
            open_time=cls._parse_api_date(question_json.get("open_time")),
            already_forecasted=is_forecasted,
            tournament_slugs=tournament_slugs,
            default_project_id=(
                post_api_json["projects"]["default_project"]["id"]
                if "projects" in post_api_json
                else None
            ),
            includes_bots_in_aggregates=question_json.get(
                "include_bots_in_aggregates", None
            ),
            question_weight=question_json.get("question_weight", None),
            resolution_string=question_json.get("resolution", None),
            group_question_option=group_question_option,
            api_json=post_api_json,
            conditional_type=question_json.get("conditional_type", None),
            categories=categories,
        )
        return question

    @classmethod
    def _parse_api_date(cls, date_value: str | float | None) -> datetime | None:
        if date_value is None:
            return None

        if isinstance(date_value, float) or isinstance(date_value, int):
            return pendulum.from_timestamp(date_value)

        parsed = pendulum.parse(date_value)
        assert isinstance(parsed, datetime)
        return parsed

    @classmethod
    def get_api_type_name(cls) -> QuestionBasicType:
        raise NotImplementedError(
            f"This function doesn't apply for base class {type(cls)}"
        )

    def give_question_details_as_markdown(self) -> str:
        today_string = pendulum.now(tz="UTC").strftime("%Y-%m-%d") + " (UTC)"
        question_details = textwrap.dedent(
            f"""
            The main question is:
            {self.question_text}

            Here is the resolution criteria:
            {self.resolution_criteria}

            Here is the fine print:
            {self.fine_print}

            Here is the background information:
            {self.background_info}

            Today is (YYYY-MM-DD):
            {today_string}
            """
        )
        return question_details.strip()

    @property
    def typed_resolution(
        self,
    ) -> ResolutionType | None:
        if self.resolution_string is None:
            return None

        assert isinstance(self.resolution_string, str)

        if self.resolution_string == "yes":
            return True
        elif self.resolution_string == "no":
            return False
        elif self.resolution_string in [v.value for v in CanceledResolution]:
            return CanceledResolution(self.resolution_string)
        elif self.resolution_string in [v.value for v in OutOfBoundsResolution]:
            return OutOfBoundsResolution(self.resolution_string)
        else:
            try:
                return float(self.resolution_string)
            except ValueError:
                try:
                    # Try parsing as ISO 8601 with timezone
                    parsed_datetime = pendulum.parse(self.resolution_string)
                    assert isinstance(parsed_datetime, datetime)
                    return parsed_datetime
                except Exception:
                    return self.resolution_string

    def get_question_type(
        self,
    ) -> QuestionBasicType:
        try:
            question_type = self.question_type  # type: ignore
        except Exception as e:
            raise AttributeError(
                f"Question type not found for {self.__class__.__name__}. Error: {e}"
            ) from e
        assert question_type == self.get_api_type_name()
        return question_type

    @property
    def timestamp_of_my_last_forecast(self) -> datetime | None:
        try:
            time_stamp: float = self.api_json["question"]["my_forecasts"]["latest"][
                "start_time"
            ]
            result = pendulum.from_timestamp(time_stamp)
        except Exception:
            result = None
        if result is not None and not self.already_forecasted:
            raise ValueError(
                f"There cannot be a last forecast time if the question is not already forecasted. Last forecast time: {result}"
            )
        return result

    @property
    def is_in_main_feed(self) -> bool | None:
        try:
            if self.api_json is None or len(self.api_json.keys()) == 0:
                return True
            visibility = self.api_json["projects"]["default_project"]["visibility"]
            is_in_main_feed = visibility == "normal"
            return is_in_main_feed
        except Exception:
            return None


class BinaryQuestion(MetaculusQuestion):
    question_type: Literal["binary"] = "binary"
    community_prediction_at_access_time: float | None = None
    previous_forecasts: list[BinaryTimestampedPrediction] | None = None

    @property
    def binary_resolution(self) -> BinaryResolution | None:
        resolution = typeguard.check_type(
            self.typed_resolution, BinaryResolution | None
        )
        return resolution

    @classmethod
    def from_metaculus_api_json(cls, api_json: dict) -> BinaryQuestion:
        normal_metaculus_question = super().from_metaculus_api_json(api_json)
        try:
            aggregations = api_json["question"]["aggregations"]
            try:
                recency_weighted_latest = aggregations["recency_weighted"]["latest"]  # type: ignore
                q2_center_community_prediction = recency_weighted_latest["centers"]  # type: ignore
            except KeyError:
                q2_center_community_prediction = aggregations["unweighted"]["latest"]["centers"]  # type: ignore
            assert len(q2_center_community_prediction) == 1
            community_prediction_at_access_time = q2_center_community_prediction[0]
            history = api_json["question"]["my_forecasts"]["history"]
            previous_forecasts = None
            if history:
                previous_forecasts = [
                    BinaryTimestampedPrediction(
                        prediction_in_decimal=forecast["forecast_values"][1],
                        timestamp=datetime.fromtimestamp(
                            forecast["start_time"], tz=timezone.utc
                        ),
                        timestamp_end=(
                            datetime.fromtimestamp(
                                forecast["end_time"], tz=timezone.utc
                            )
                            if forecast["end_time"]
                            else None
                        ),
                    )
                    for forecast in history
                ]
        except (KeyError, TypeError):
            community_prediction_at_access_time = None
            previous_forecasts = None
        return BinaryQuestion(
            community_prediction_at_access_time=community_prediction_at_access_time,
            previous_forecasts=previous_forecasts,
            **normal_metaculus_question.model_dump(exclude={"previous_forecasts"}),
        )

    @classmethod
    def get_api_type_name(cls) -> QuestionBasicType:
        return "binary"

    def get_cp_at_time(
        self,
        time: datetime,
    ) -> float | None:
        if time > pendulum.now():
            # NOTE: the community prediction often exists for future dates 1 week to 2
            #       months out since the CP is calculated for these dates as long as
            #       people's forecasts are not expired.
            #       We don't want to return a CP for a future date, so we return None.
            return None

        cp_history: list[dict] = self.api_json["question"]["aggregations"][
            "recency_weighted"
        ]["history"]
        assert time.tzinfo is not None

        time_timestamp = time.timestamp()

        for history_item in cp_history:
            start_time = history_item["start_time"]
            end_time = history_item["end_time"]

            if start_time <= time_timestamp and (
                not end_time or time_timestamp <= end_time
            ):
                centers = history_item["centers"]
                if len(centers) == 1:
                    return centers[0]
                else:
                    raise ValueError(
                        f"Expected 1 center, got {len(centers)} for time {time}"
                    )

        logger.warning(f"No confidence prediction found for time {time}")
        return None


class BoundedQuestionMixin:
    @classmethod
    def _get_bounds_from_api_json(
        cls, api_json: dict
    ) -> tuple[bool, bool, float, float, float | None]:
        try:
            open_upper_bound = api_json["question"]["open_upper_bound"]
            open_lower_bound = api_json["question"]["open_lower_bound"]
        except KeyError:
            logger.warning(
                "Open bounds not found in API JSON defaulting to 'open bounds'"
            )
            open_lower_bound = True
            open_upper_bound = True

        upper_bound = api_json["question"]["scaling"]["range_max"]
        lower_bound = api_json["question"]["scaling"]["range_min"]
        zero_point = api_json["question"]["scaling"]["zero_point"]

        try:
            upper_bound = float(upper_bound)
            lower_bound = float(lower_bound)
        except (TypeError, ValueError):
            logger.error(
                f"Error parsing bounds from API JSON. "
                f"Upper bound: {upper_bound}, Lower bound: {lower_bound}"
            )
            raise
        return (
            open_upper_bound,
            open_lower_bound,
            upper_bound,
            lower_bound,
            zero_point,
        )

    @classmethod
    def _get_cdf_size_from_json(cls, api_json: dict) -> int:
        try:
            outcome_count = api_json["question"]["scaling"]["inbound_outcome_count"]
            if outcome_count is None:
                outcome_count = 200
            cdf_size = outcome_count + 1  # Add 1 to account for this being a cdf
        except KeyError:
            logger.warning("CDF not found in API JSON using defaults")
            return 201
        return cdf_size

    @classmethod
    def _get_nominal_bounds_from_json(
        cls, api_json: dict
    ) -> tuple[float | None, float | None]:
        try:
            nominal_lower_bound = api_json["question"]["scaling"]["nominal_min"]
            nominal_upper_bound = api_json["question"]["scaling"]["nominal_max"]
        except KeyError:
            nominal_lower_bound = None
            nominal_upper_bound = None
        return nominal_lower_bound, nominal_upper_bound


class DateQuestion(MetaculusQuestion, BoundedQuestionMixin):
    question_type: Literal["date"] = "date"
    upper_bound: datetime
    lower_bound: datetime
    open_upper_bound: bool
    open_lower_bound: bool
    zero_point: float | None = None
    cdf_size: int = 201

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_bounds(cls, data: dict) -> dict:
        # Handle upper bound
        if "upper_bound_is_hard_limit" in data and "open_upper_bound" not in data:
            data["open_upper_bound"] = not data["upper_bound_is_hard_limit"]
        # Handle lower bound
        if "lower_bound_is_hard_limit" in data and "open_lower_bound" not in data:
            data["open_lower_bound"] = not data["lower_bound_is_hard_limit"]
        return data

    @property
    def date_resolution(self) -> DateResolution | None:
        resolution = typeguard.check_type(self.typed_resolution, DateResolution | None)
        return resolution

    @classmethod
    def from_metaculus_api_json(cls, api_json: dict) -> DateQuestion:
        normal_metaculus_question = super().from_metaculus_api_json(api_json)
        (
            open_upper_bound,
            open_lower_bound,
            unparsed_upper_bound,
            unparsed_lower_bound,
            zero_point,
        ) = cls._get_bounds_from_api_json(api_json)

        upper_bound = cls._parse_api_date(unparsed_upper_bound)
        lower_bound = cls._parse_api_date(unparsed_lower_bound)
        assert upper_bound is not None
        assert lower_bound is not None

        return DateQuestion(
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            open_upper_bound=open_upper_bound,
            open_lower_bound=open_lower_bound,
            zero_point=zero_point,
            cdf_size=cls._get_cdf_size_from_json(api_json),
            **normal_metaculus_question.model_dump(),
        )

    @classmethod
    def get_api_type_name(cls) -> QuestionBasicType:
        return "date"


class NumericQuestion(MetaculusQuestion, BoundedQuestionMixin):
    question_type: Literal["numeric"] = "numeric"
    upper_bound: float
    lower_bound: float
    open_upper_bound: bool
    open_lower_bound: bool
    zero_point: float | None = None
    cdf_size: int = (
        201  # Normal numeric questions have 201 points, but discrete questions have fewer
    )
    nominal_upper_bound: float | None = None
    nominal_lower_bound: float | None = None
    previous_forecasts: list[NumericTimestampedDistribution] | None = None

    @property
    def numeric_resolution(self) -> NumericResolution | None:
        resolution = typeguard.check_type(
            self.typed_resolution, NumericResolution | None
        )
        return resolution

    @staticmethod
    def _get_percentiles(
        continuous_range: list[float], forecast_values: list[float]
    ) -> list[Percentile]:
        zipped_list = list(zip(continuous_range, forecast_values))
        return [Percentile(value=x, percentile=y) for x, y in zipped_list]

    @classmethod
    def from_metaculus_api_json(cls, api_json: dict) -> NumericQuestion:
        normal_metaculus_question = super().from_metaculus_api_json(api_json)
        (
            open_upper_bound,
            open_lower_bound,
            upper_bound,
            lower_bound,
            zero_point,
        ) = cls._get_bounds_from_api_json(api_json)
        assert isinstance(upper_bound, float)
        assert isinstance(lower_bound, float)

        nominal_lower_bound, nominal_upper_bound = cls._get_nominal_bounds_from_json(
            api_json
        )

        try:
            continuous_range = api_json["question"]["scaling"]["continuous_range"]
            history = api_json["question"]["my_forecasts"]["history"]
            previous_forecasts = None
            if history:
                previous_forecasts = [
                    NumericTimestampedDistribution(
                        declared_percentiles=NumericQuestion._get_percentiles(
                            continuous_range, forecast["forecast_values"]
                        ),
                        open_upper_bound=open_upper_bound,
                        open_lower_bound=open_lower_bound,
                        upper_bound=upper_bound,
                        lower_bound=lower_bound,
                        zero_point=zero_point,
                        timestamp=datetime.fromtimestamp(
                            forecast["start_time"], tz=timezone.utc
                        ),
                        timestamp_end=(
                            datetime.fromtimestamp(
                                forecast["end_time"], tz=timezone.utc
                            )
                            if forecast["end_time"]
                            else None
                        ),
                    )
                    for forecast in history
                ]
        except KeyError:
            previous_forecasts = None

        return NumericQuestion(
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            open_upper_bound=open_upper_bound,
            open_lower_bound=open_lower_bound,
            zero_point=zero_point,
            cdf_size=cls._get_cdf_size_from_json(api_json),
            nominal_upper_bound=nominal_upper_bound,
            nominal_lower_bound=nominal_lower_bound,
            previous_forecasts=previous_forecasts,
            **normal_metaculus_question.model_dump(exclude={"previous_forecasts"}),
        )

    @classmethod
    def get_api_type_name(cls) -> QuestionBasicType:
        return "numeric"

    def give_question_details_as_markdown(self) -> str:
        original_details = super().give_question_details_as_markdown()
        final_details = (
            original_details
            + f"\n\nThe upper bound is {self.upper_bound} and the lower bound is {self.lower_bound}"
            + f"\nOpen upper bound is {self.open_upper_bound} and open lower bound is {self.open_lower_bound}"
            + f"\nThe zero point is {self.zero_point}"
        )
        return final_details.strip()


class DiscreteQuestion(NumericQuestion):
    question_type: Literal["discrete"] = "discrete"

    @classmethod
    def from_metaculus_api_json(cls, api_json: dict) -> DiscreteQuestion:
        normal_metaculus_question = super().from_metaculus_api_json(api_json)
        normal_metaculus_question.question_type = "discrete"  # type: ignore
        question = DiscreteQuestion(
            **normal_metaculus_question.model_dump(),
        )
        return question

    @classmethod
    def get_api_type_name(cls) -> QuestionBasicType:
        return "discrete"


class ConditionalQuestion(MetaculusQuestion):
    question_type: Literal["conditional"] = "conditional"
    parent: MetaculusQuestion
    child: MetaculusQuestion
    question_yes: MetaculusQuestion
    question_no: MetaculusQuestion

    @staticmethod
    def _conditional_questions_add_detail(
        question: MetaculusQuestion,
        parent: MetaculusQuestion,
        child: MetaculusQuestion,
        yes: bool,
    ) -> MetaculusQuestion:
        resolved = '"yes"' if yes else '"no"'
        question.question_text = f"`{child.question_text}`, if the question `{parent.question_text}` resolves to {resolved}"
        question.resolution_criteria = clean_indents(
            f"""
            IMPORTANT: This is a conditional forecasting question with two components:

            1. **Condition (Parent Question, NOT BEING FORECASTED RIGHT NOW)**: "{parent.question_text}"
            2. **Outcome (Child Question)**: "{child.question_text}"

            ## Your Task
            You are forecasting the CHILD question, assuming the PARENT question has resolved to {resolved}.
            Keep in mind, this is extremely important: you are **NOT** forecasting the PARENT QUESTION!
            You have been provided with research relating to the parent question, but you are NOT to forecast it.
            You shall assume the parent question is resolved to {resolved}, focus on the child question GIVEN THAT INFORMATION.
            Make sure to only include information relevant to the CHILD QUESTION in the final output, this is absolutely essential.

            ## Resolution Criteria for Parent Question (Assumed to Resolve {resolved})
            ```
            {parent.resolution_criteria}
            ```

            ## Resolution Criteria for Child Question (What You Are Actually Forecasting)
            ```
            {child.resolution_criteria}
            ```
        """
        )
        question.fine_print = clean_indents(
            f"""
            ## Fine Print for Parent Question (Assumed to Resolve {resolved})
            ```
            {parent.fine_print}
            ```

            ## Fine Print for Child Question (What You Are Actually Forecasting)
            ```
            {child.fine_print}
            ```
        """
        )
        question.background_info = clean_indents(
            f"""
            ## Background Information for Parent Question (Assumed to Resolve {resolved})
            ```
            {parent.background_info}
            ```

            Parent question URL: {parent.page_url}

            ## Background Information for Child Question (What You Are Actually Forecasting)
            ```
            {child.background_info}
            ```

            Child question URL: {child.page_url}
        """
        )
        return question

    @staticmethod
    def _unpack_individual_conditional_question(
        question_json: Any,
        post_json_from_api: Any,
        question_type: ConditionalSubQuestionType,
    ) -> MetaculusQuestion:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        new_question_json = copy.deepcopy(question_json)

        new_post_json = copy.deepcopy(post_json_from_api)
        new_post_json["question"] = new_question_json
        new_post_json["question"]["conditional_type"] = question_type

        question_obj = DataOrganizer.get_question_from_post_json(new_post_json)
        return question_obj

    @classmethod
    def from_metaculus_api_json(cls, api_json: dict) -> ConditionalQuestion:
        conditional = api_json["question"]
        api_json["question"] = {**api_json, **api_json["question"]}
        api_json["question"]["my_forecasts"] = (
            conditional["question_yes"]["my_forecasts"]
            or conditional["question_no"]["my_forecasts"]
        )
        base_question = MetaculusQuestion.from_metaculus_api_json(api_json)
        parent_question = cls._unpack_individual_conditional_question(
            conditional["condition"], api_json, "parent"
        )
        child_question = cls._unpack_individual_conditional_question(
            conditional["condition_child"], api_json, "child"
        )
        question_yes = cls._unpack_individual_conditional_question(
            conditional["question_yes"], api_json, "yes"
        )
        question_no = cls._unpack_individual_conditional_question(
            conditional["question_no"], api_json, "no"
        )
        question_yes = cls._conditional_questions_add_detail(
            question_yes, parent_question, child_question, True
        )
        question_no = cls._conditional_questions_add_detail(
            question_no, parent_question, child_question, False
        )
        questions_map = {
            "parent": parent_question,
            "child": child_question,
            "question_yes": question_yes,
            "question_no": question_no,
        }

        return ConditionalQuestion(
            **base_question.model_dump(exclude={"question_type"}), **questions_map
        )

    def get_all_subquestions(self) -> dict[str, MetaculusQuestion]:
        return {
            "parent": self.parent,
            "child": self.child,
            "question_yes": self.question_yes,
            "question_no": self.question_no,
        }

    @classmethod
    def get_api_type_name(cls) -> QuestionBasicType:
        return "conditional"

    @model_validator(mode="before")
    @classmethod
    def preprocess(cls, data):
        if isinstance(data, dict):
            if "question_text" not in data:
                data["question_text"] = data["title"]
                data["parent"] = data["conditional"]["condition"]
                data["parent"]["question_text"] = data["conditional"]["condition"][
                    "title"
                ]
                data["child"] = data["conditional"]["condition_child"]
                data["child"]["question_text"] = data["conditional"]["condition_child"][
                    "title"
                ]
                data["question_yes"] = data["conditional"]["question_yes"]
                data["question_yes"]["question_text"] = data["conditional"][
                    "question_yes"
                ]["title"]
                data["question_no"] = data["conditional"]["question_no"]
                data["question_no"]["question_text"] = data["conditional"][
                    "question_no"
                ]["title"]
        return data


class MultipleChoiceQuestion(MetaculusQuestion):
    question_type: Literal["multiple_choice"] = "multiple_choice"
    options: list[str]
    option_is_instance_of: str | None = None

    @property
    def mc_resolution(self) -> MultipleChoiceResolution | None:
        resolution = typeguard.check_type(
            self.typed_resolution, MultipleChoiceResolution | None
        )
        if isinstance(resolution, str) and resolution not in self.options:
            raise ValueError(
                f"Resolution {resolution} is not in options {self.options}"
            )
        return resolution

    @classmethod
    def from_metaculus_api_json(cls, api_json: dict) -> MultipleChoiceQuestion:
        normal_metaculus_question = super().from_metaculus_api_json(api_json)
        return MultipleChoiceQuestion(
            options=api_json["question"]["options"],
            option_is_instance_of=api_json["question"]["group_variable"],
            **normal_metaculus_question.model_dump(),
        )

    @classmethod
    def get_api_type_name(cls) -> QuestionBasicType:
        return "multiple_choice"

    def give_question_details_as_markdown(self) -> str:
        original_details = super().give_question_details_as_markdown()
        final_details = (
            original_details
            + f"\n\nThe final options you can choose are:\n {self.options}"
        )
        return final_details.strip()
