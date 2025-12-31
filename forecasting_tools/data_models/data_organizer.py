import typeguard
from pydantic import BaseModel

from forecasting_tools.data_models.binary_report import BinaryPrediction, BinaryReport
from forecasting_tools.data_models.conditional_models import (
    ConditionalPrediction,
    PredictionAffirmed,
)
from forecasting_tools.data_models.conditional_report import ConditionalReport
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.multiple_choice_report import (
    MultipleChoiceReport,
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import (
    DateReport,
    DiscreteReport,
    NumericDistribution,
    NumericReport,
)
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    ConditionalQuestion,
    DateQuestion,
    DiscreteQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.util import file_manipulation


class TypeMapping(BaseModel):
    question_type: type[MetaculusQuestion]
    test_post_id: int
    report_type: type[ForecastReport] | None


PredictionTypes = (
    NumericDistribution | PredictedOptionList | float | ConditionalPrediction
)
QuestionTypes = (
    NumericQuestion
    | DateQuestion
    | MultipleChoiceQuestion
    | BinaryQuestion
    | DiscreteQuestion
    | ConditionalQuestion
)
ReportTypes = (
    NumericReport | MultipleChoiceReport | BinaryReport | DiscreteReport | DateReport
)


class DataOrganizer:
    __TYPE_MAPPING = [
        TypeMapping(
            question_type=NumericQuestion,
            test_post_id=14333,  # https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/
            report_type=NumericReport,
        ),
        TypeMapping(
            question_type=DiscreteQuestion,
            test_post_id=38880,  # https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/
            report_type=DiscreteReport,
        ),
        TypeMapping(
            question_type=DateQuestion,
            test_post_id=4110,  # https://www.metaculus.com/questions/4110/birthdate-of-oldest-living-human-in-2200/
            report_type=DateReport,
        ),
        TypeMapping(
            question_type=MultipleChoiceQuestion,
            test_post_id=22427,  # https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/
            report_type=MultipleChoiceReport,
        ),
        TypeMapping(
            question_type=BinaryQuestion,
            test_post_id=578,  # https://www.metaculus.com/questions/578/human-extinction-by-2100/
            report_type=BinaryReport,
        ),
        TypeMapping(
            question_type=ConditionalQuestion,
            test_post_id=40107,  # https://www.metaculus.com/questions/40107/conditional-someone-born-before-2001-lives-to-150/
            report_type=ConditionalReport,
        ),
    ]

    @classmethod
    def get_example_post_id_for_question_type(
        cls, question_type: type[MetaculusQuestion]
    ) -> int:
        assert issubclass(question_type, MetaculusQuestion)
        for mapping in cls.__TYPE_MAPPING:
            if mapping.question_type == question_type:
                return mapping.test_post_id
        raise ValueError(f"No question ID found for type {question_type}")

    @classmethod
    def get_report_type_for_question_type(
        cls, question_type: type[MetaculusQuestion]
    ) -> type[ForecastReport]:
        assert issubclass(question_type, MetaculusQuestion)
        for mapping in cls.__TYPE_MAPPING:
            if mapping.question_type == question_type:
                if mapping.report_type is None:
                    raise ValueError(f"No report type found for type {question_type}")
                return mapping.report_type
        raise ValueError(f"No report type found for type {question_type}")

    @classmethod
    def get_live_example_question_of_type(
        cls, question_type: type[MetaculusQuestion]
    ) -> MetaculusQuestion:
        from forecasting_tools.helpers.metaculus_client import MetaculusClient

        assert issubclass(question_type, MetaculusQuestion)
        question_id = cls.get_example_post_id_for_question_type(question_type)
        question = MetaculusClient().get_question_by_post_id(question_id)
        assert isinstance(question, question_type)
        return question

    @classmethod
    def get_all_report_types(cls) -> list[type[ForecastReport]]:
        return [
            mapping.report_type
            for mapping in cls.__TYPE_MAPPING
            if mapping.report_type is not None
        ]

    @classmethod
    def get_all_question_types(cls) -> list[type[MetaculusQuestion]]:
        return [mapping.question_type for mapping in cls.__TYPE_MAPPING]

    @classmethod
    def load_reports_from_file_path(cls, file_path: str) -> list[ForecastReport]:
        jsons = file_manipulation.load_json_file(file_path)
        reports = cls._load_objects_from_json(jsons, cls.get_all_report_types())  # type: ignore
        reports = typeguard.check_type(reports, list[ForecastReport])
        return reports

    @classmethod
    def load_questions_from_file_path(cls, file_path: str) -> list[MetaculusQuestion]:
        jsons = file_manipulation.load_json_file(file_path)
        questions = cls._load_objects_from_json(jsons, cls.get_all_question_types())  # type: ignore
        questions = typeguard.check_type(questions, list[MetaculusQuestion])
        return questions

    @classmethod
    def get_question_from_post_json(cls, post_json: dict) -> MetaculusQuestion:
        assert "question" in post_json, "Question key not found in API JSON"
        question_type_string = post_json["question"]["type"]  # type: ignore
        if question_type_string == BinaryQuestion.get_api_type_name():
            question_type = BinaryQuestion
        elif question_type_string == NumericQuestion.get_api_type_name():
            question_type = NumericQuestion
        elif question_type_string == DiscreteQuestion.get_api_type_name():
            question_type = DiscreteQuestion
        elif question_type_string == MultipleChoiceQuestion.get_api_type_name():
            question_type = MultipleChoiceQuestion
        elif question_type_string == DateQuestion.get_api_type_name():
            question_type = DateQuestion
        elif question_type_string == ConditionalQuestion.get_api_type_name():
            question_type = ConditionalQuestion
        else:
            raise ValueError(f"Unknown question type: {question_type_string}")
        question = question_type.from_metaculus_api_json(post_json)
        return question

    @classmethod
    def get_question_from_question_json(cls, question_json: dict) -> MetaculusQuestion:
        return cls.get_question_from_post_json(
            {"id": question_json["post_id"], "question": question_json}
        )

    @classmethod
    def save_reports_to_file_path(
        cls, reports: list[ForecastReport], file_path: str
    ) -> None:
        jsons = []
        for report in reports:
            jsons.append(report.to_json())
        file_manipulation.write_json_file(file_path, jsons)

    @classmethod
    def save_questions_to_file_path(
        cls, questions: list[MetaculusQuestion], file_path: str
    ) -> None:
        jsons = []
        for question in questions:
            jsons.append(question.to_json())
        file_manipulation.write_json_file(file_path, jsons)

    @classmethod
    def _load_objects_from_json(
        cls,
        jsons: list[dict],
        types: list[type[ForecastReport] | type[MetaculusQuestion]],
    ) -> list[ForecastReport | MetaculusQuestion]:
        objects: list[ForecastReport | MetaculusQuestion] = []
        for json in jsons:
            for i, object_type in enumerate(types):
                try:
                    obj = object_type.from_json(json)
                    objects.append(obj)
                    break
                except Exception as e:
                    if i == len(types) - 1:
                        raise e
                    continue
        if len(objects) != len(jsons):
            raise ValueError(
                f"Some objects were not loaded correctly. {len(objects)} objects loaded, {len(jsons)} jsons provided."
            )
        return objects

    @classmethod
    def get_readable_prediction(
        cls, prediction: PredictionTypes | BinaryPrediction
    ) -> str:
        if isinstance(prediction, NumericDistribution):
            return NumericReport.make_readable_prediction(prediction)
        elif isinstance(prediction, PredictedOptionList):
            return MultipleChoiceReport.make_readable_prediction(prediction)
        elif isinstance(prediction, float):
            return BinaryReport.make_readable_prediction(prediction)
        elif isinstance(prediction, BinaryPrediction):
            return BinaryReport.make_readable_prediction(
                prediction.prediction_in_decimal
            )
        elif isinstance(prediction, ConditionalPrediction):
            return ConditionalReport.make_readable_prediction(prediction)
        elif isinstance(prediction, PredictionAffirmed):
            return "affirm"
        else:
            raise ValueError("Unknown prediction type.")
