from pydantic import BaseModel

from forecasting_tools.forecasting.helpers.metaculus_api import MetaculusApi
from forecasting_tools.forecasting.questions_and_reports.binary_report import (
    BinaryReport,
)
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ForecastReport,
)
from forecasting_tools.forecasting.questions_and_reports.multiple_choice_report import (
    MultipleChoiceReport,
)
from forecasting_tools.forecasting.questions_and_reports.numeric_report import (
    NumericReport,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.util import file_manipulation


class TypeMapping(BaseModel):
    question_type: type[MetaculusQuestion]
    test_post_id: int
    report_type: type[ForecastReport]


class ReportOrganizer:
    __TYPE_MAPPING = [
        TypeMapping(
            question_type=BinaryQuestion,
            test_post_id=578,  # https://www.metaculus.com/questions/578/human-extinction-by-2100/
            report_type=BinaryReport,
        ),
        TypeMapping(
            question_type=NumericQuestion,
            test_post_id=14333,  # https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/
            report_type=NumericReport,
        ),
        # TypeMapping(
        #     question_type=DateQuestion,
        #     test_question_id=4110,  # https://www.metaculus.com/questions/4110/birthdate-of-oldest-living-human-in-2200/
        #     report_type=None,
        # ),
        TypeMapping(
            question_type=MultipleChoiceQuestion,
            test_post_id=22427,  # https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/
            report_type=MultipleChoiceReport,
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
                    raise ValueError(
                        f"No report type found for type {question_type}"
                    )
                return mapping.report_type
        raise ValueError(f"No report type found for type {question_type}")

    @classmethod
    def get_live_example_question_of_type(
        cls, question_type: type[MetaculusQuestion]
    ) -> MetaculusQuestion:
        assert issubclass(question_type, MetaculusQuestion)
        question_id = cls.get_example_post_id_for_question_type(question_type)
        question = MetaculusApi.get_question_by_post_id(question_id)
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
    def load_reports_from_file_path(
        cls, file_path: str
    ) -> list[ForecastReport]:
        jsons = file_manipulation.load_json_file(file_path)
        reports = []
        for json in jsons:
            report_types = cls.get_all_report_types()
            for i, report_type in enumerate(report_types):
                try:
                    report = report_type.from_json(json)
                    reports.append(report)
                    break
                except Exception as e:
                    if i == len(report_types) - 1:
                        raise e
                    continue
        if len(reports) != len(jsons):
            raise ValueError(
                f"Some reports were not loaded correctly. {len(reports)} reports loaded, {len(jsons)} jsons provided."
            )
        return reports

    @classmethod
    def save_reports_to_file_path(
        cls, reports: list[ForecastReport], file_path: str
    ) -> None:
        jsons = []
        for report in reports:
            jsons.append(report.to_json())
        file_manipulation.write_json_file(file_path, jsons)
