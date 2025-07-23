from __future__ import annotations

import logging
import textwrap
from datetime import datetime
from enum import Enum
from typing import Literal

import typeguard
from pydantic import AliasChoices, BaseModel, Field, model_validator

from forecasting_tools.util.jsonable import Jsonable

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

QuestionBasicType = Literal["binary", "numeric", "multiple_choice", "date"]


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
    unit_of_measure: str | None = None  # TODO: Move this field to continuous questions
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
    date_accessed: datetime = Field(default_factory=datetime.now)
    already_forecasted: bool = False
    tournament_slugs: list[str] = Field(default_factory=list)
    default_project_id: int | None = None
    includes_bots_in_aggregates: bool | None = None
    cp_reveal_time: datetime | None = None  # Community Prediction Reveal Time
    question_weight: float | None = None
    resolution_string: str | None = None
    group_question_option: str | None = (
        None  # For group questions like "How many people will die of coronovirus in the following periouds" it would be "September 2024", "All of 2025", etc
    )
    api_json: dict = Field(
        description=(
            "The API JSON response used to create the question. "
            "For group questions, a fake 'question' entry may be made to help with group question expansion "
            "(full group question info is under 'group_of_questions')"
        ),
        default_factory=dict,
    )

    @classmethod
    def from_metaculus_api_json(cls, post_api_json: dict) -> MetaculusQuestion:
        post_id = post_api_json["id"]
        logger.debug(f"Processing Post ID {post_id}")

        question_json: dict = post_api_json["question"]
        json_state = question_json["status"]
        question_state = QuestionState(json_state)

        try:
            forecast_values = question_json["my_forecasts"]["latest"][  # type: ignore
                "forecast_values"
            ]
            is_forecasted = forecast_values is not None
        except Exception:
            is_forecasted = False

        try:
            tournaments: list[dict] = post_api_json["projects"]["tournament"]  # type: ignore
            tournament_slugs = [str(t["slug"]) for t in tournaments]
        except KeyError:
            tournament_slugs = []

        question = MetaculusQuestion(
            # NOTE: Reminder - When adding new fields, consider if group questions
            #       need to be parsed differently (i.e. if the field information is part of the post_json)
            #       Also, anything that filters on the question level needs a local filter added to MetaculusApi (since the site does not filter subquestions)
            state=question_state,
            question_text=question_json["title"],
            id_of_post=post_id,
            id_of_question=question_json["id"],
            background_info=question_json.get("description", None),
            fine_print=question_json.get("fine_print", None),
            resolution_criteria=question_json.get("resolution_criteria", None),
            unit_of_measure=question_json.get("unit", None),
            page_url=f"https://www.metaculus.com/questions/{post_id}",
            num_forecasters=post_api_json["nr_forecasters"],
            num_predictions=post_api_json["forecasts_count"],
            close_time=cls._parse_api_date(question_json.get("scheduled_close_time")),
            actual_resolution_time=cls._parse_api_date(
                question_json.get("actual_resolve_time")
            ),
            scheduled_resolution_time=cls._parse_api_date(
                question_json.get("scheduled_resolve_time")
            ),
            published_time=cls._parse_api_date(post_api_json.get("published_at")),
            cp_reveal_time=cls._parse_api_date(question_json.get("cp_reveal_time")),
            open_time=cls._parse_api_date(question_json.get("open_time")),
            already_forecasted=is_forecasted,
            tournament_slugs=tournament_slugs,
            default_project_id=post_api_json["projects"]["default_project"]["id"],
            includes_bots_in_aggregates=question_json["include_bots_in_aggregates"],
            question_weight=question_json["question_weight"],
            resolution_string=question_json.get("resolution"),
            group_question_option=question_json.get("label"),
            api_json=post_api_json,
        )
        return question

    @classmethod
    def _parse_api_date(cls, date_value: str | float | None) -> datetime | None:
        if date_value is None:
            return None

        if isinstance(date_value, float):
            return datetime.fromtimestamp(date_value)

        date_formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d",
        ]

        assert isinstance(date_value, str)
        for date_format in date_formats:
            try:
                return datetime.strptime(date_value, date_format)
            except ValueError:
                continue

        raise ValueError(f"Unable to parse date: {date_value}")

    @classmethod
    def get_api_type_name(cls) -> QuestionBasicType:
        raise NotImplementedError(
            f"This function doesn't apply for base class {type(cls)}"
        )

    def give_question_details_as_markdown(self) -> str:
        today_string = datetime.now().strftime("%Y-%m-%d")
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
                    return datetime.fromisoformat(self.resolution_string)
                except ValueError:
                    return self.resolution_string

    def get_question_type(
        self,
    ) -> Literal["binary", "date", "numeric", "multiple_choice"]:
        try:
            return self.question_type  # type: ignore
        except Exception as e:
            raise AttributeError(
                f"Question type not found for {self.__class__.__name__}. Error: {e}"
            ) from e


class BinaryQuestion(MetaculusQuestion):
    question_type: Literal["binary"] = "binary"
    community_prediction_at_access_time: float | None = None

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
            q2_center_community_prediction = api_json["question"]["aggregations"]["recency_weighted"]["latest"]["centers"]  # type: ignore
            assert len(q2_center_community_prediction) == 1
            community_prediction_at_access_time = q2_center_community_prediction[0]
        except (KeyError, TypeError):
            community_prediction_at_access_time = None
        return BinaryQuestion(
            community_prediction_at_access_time=community_prediction_at_access_time,
            **normal_metaculus_question.model_dump(),
        )

    @classmethod
    def get_api_type_name(cls) -> QuestionBasicType:
        return "binary"


class BoundedQuestionMixin:
    @classmethod
    def _get_bounds_from_api_json(
        cls, api_json: dict
    ) -> tuple[bool, bool, float, float, float | None]:
        try:
            open_upper_bound = api_json["question"]["open_upper_bound"]  # type: ignore
            open_lower_bound = api_json["question"]["open_lower_bound"]  # type: ignore
        except KeyError:
            logger.warning(
                "Open bounds not found in API JSON defaulting to 'open bounds'"
            )
            open_lower_bound = True
            open_upper_bound = True

        upper_bound = api_json["question"]["scaling"]["range_max"]  # type: ignore
        lower_bound = api_json["question"]["scaling"]["range_min"]  # type: ignore
        zero_point = api_json["question"]["scaling"]["zero_point"]  # type: ignore

        assert isinstance(upper_bound, float), f"Upper bound is {upper_bound}"
        assert isinstance(lower_bound, float), f"Lower bound is {lower_bound}"
        return (
            open_upper_bound,
            open_lower_bound,
            upper_bound,
            lower_bound,
            zero_point,
        )


class DateQuestion(MetaculusQuestion, BoundedQuestionMixin):
    question_type: Literal["date"] = "date"
    upper_bound: datetime
    lower_bound: datetime
    open_upper_bound: bool
    open_lower_bound: bool
    zero_point: float | None = None

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

    @property
    def numeric_resolution(self) -> NumericResolution | None:
        resolution = typeguard.check_type(
            self.typed_resolution, NumericResolution | None
        )
        return resolution

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

        return NumericQuestion(
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            open_upper_bound=open_upper_bound,
            open_lower_bound=open_lower_bound,
            zero_point=zero_point,
            **normal_metaculus_question.model_dump(),
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
