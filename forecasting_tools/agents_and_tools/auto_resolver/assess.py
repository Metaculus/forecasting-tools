from forecasting_tools.agents_and_tools.auto_resolver import AutoResolver
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from forecasting_tools.data_models.questions import QuestionBasicType, BinaryResolution, CanceledResolution, ResolutionType
from forecasting_tools.helpers.metaculus_client import MetaculusClient
from forecasting_tools import MetaculusQuestion, ApiFilter, BinaryQuestion
from dataclasses import dataclass, field


@dataclass
class QuestionAssessmentResult:
    """Detailed assessment result for a single question."""
    
    question_id: int
    question_title: str
    question_text: str
    question_url: str
    actual_resolution: ResolutionType
    predicted_resolution: Optional[ResolutionType]
    key_evidence: Optional[list[str]] = None
    outcome_category: Optional[str] = None


@dataclass
class ResolutionAssessmentReport:
    """
    Contained variables are arrays, named according to the convention: "{predicted}{true}".
    Values for predicted/true are as follows:
    - p: positive
    - n: negative
    - c: cancelled
    - x: not answered  
    """

    pp: list[int] = field(default_factory=list)
    pn: list[int] = field(default_factory=list)
    pc: list[int] = field(default_factory=list)
    
    np: list[int] = field(default_factory=list)
    nn: list[int] = field(default_factory=list)
    nc: list[int] = field(default_factory=list)

    cp: list[int] = field(default_factory=list)  
    cn: list[int] = field(default_factory=list)  
    cc: list[int] = field(default_factory=list)  

    xp: list[int] = field(default_factory=list)  
    xn: list[int] = field(default_factory=list)
    xc: list[int] = field(default_factory=list)
    
    nyr_p: list[int] = field(default_factory=list)
    nyr_n: list[int] = field(default_factory=list)
    nyr_c: list[int] = field(default_factory=list)
    nyr_nyr: list[int] = field(default_factory=list)
    nyr_x: list[int] = field(default_factory=list)
    
    question_results: dict[int, QuestionAssessmentResult] = field(default_factory=dict)

    @property
    def n_pp(self) -> int:
        return len(self.pp)

    @property
    def n_pn(self) -> int:
        return len(self.pn)

    @property
    def n_pc(self) -> int:
        return len(self.pc)

    @property
    def n_np(self) -> int:
        return len(self.np)

    @property
    def n_nn(self) -> int:
        return len(self.nn)

    @property
    def n_nc(self) -> int:
        return len(self.nc)

    @property
    def n_cp(self) -> int:
        return len(self.cp)

    @property
    def n_cn(self) -> int:
        return len(self.cn)

    @property
    def n_cc(self) -> int:
        return len(self.cc)

    @property
    def n_xp(self) -> int:
        return len(self.xp)

    @property
    def n_xn(self) -> int:
        return len(self.xn)

    @property
    def n_xc(self) -> int:
        return len(self.xc)

    @property
    def n_nyr_p(self) -> int:
        return len(self.nyr_p)

    @property
    def n_nyr_n(self) -> int:
        return len(self.nyr_n)

    @property
    def n_nyr_c(self) -> int:
        return len(self.nyr_c)

    @property
    def n_nyr_nyr(self) -> int:
        return len(self.nyr_nyr)

    @property
    def n_nyr_x(self) -> int:
        return len(self.nyr_x)
    
    def binary_results_table(self) -> str:
        """
        Returns a markdown table representation of the binary assessment report.

        Columns represent predicted resolutions, rows represent actual resolutions.
        Then "Not Answered" column contains cases where the resolver returned None
        or an unexpected value (logged as warnings for debugging).

        Returns:
            str: A markdown formatted confusion matrix table
        """
        corner_label = "Actual \\ Predicted"
        col_headers = ["Positive", "Negative", "Cancelled", "Not Yet Resolvable", "Not Answered"]
        row_labels = ["Positive", "Negative", "Cancelled", "Not Yet Resolvable"]
        # Rows ordered: actual Positive, actual Negative, actual Cancelled, actual Not Yet Resolvable
        # Columns ordered: predicted Positive, predicted Negative, predicted Cancelled, predicted Not Yet Resolvable, predicted Not Answered
        data = [
            [str(self.n_pp), str(self.n_np), str(self.n_cp), str(self.n_nyr_p), str(self.n_xp)],
            [str(self.n_pn), str(self.n_nn), str(self.n_cn), str(self.n_nyr_n), str(self.n_xn)],
            [str(self.n_pc), str(self.n_nc), str(self.n_cc), str(self.n_nyr_c), str(self.n_xc)],
            [str(self.n_nyr_p), str(self.n_nyr_n), str(self.n_nyr_c), str(self.n_nyr_nyr), str(self.n_nyr_x)],
        ]

        # Compute column widths dynamically
        col_widths = [len(corner_label)]
        for i, header in enumerate(col_headers):
            max_data_width = max(len(data[row][i]) for row in range(len(row_labels)))
            col_widths.append(max(len(header), max_data_width))

        # Update corner label width to account for row labels
        col_widths[0] = max(col_widths[0], max(len(label) for label in row_labels))

        def fmt_row(cells: list[str]) -> str:
            parts = []
            for i, cell in enumerate(cells):
                parts.append(f" {cell.ljust(col_widths[i])} ")
            return "|" + "|".join(parts) + "|"

        def fmt_separator() -> str:
            return "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"

        header_row = fmt_row([corner_label] + col_headers)
        separator = fmt_separator()
        body_rows = [
            fmt_row([row_labels[r]] + data[r])
            for r in range(len(row_labels))
        ]

        return "\n".join([header_row, separator] + body_rows)

    def detailed_report(self) -> str:
        """
        Returns a complete detailed markdown report including summary and per-question details.

        Returns:
            str: A complete markdown formatted assessment report
        """
        lines = []
        lines.append("# Auto Resolver Assessment Report\n")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Summary section
        lines.append("## Summary\n")
        lines.append(self.binary_results_table())
        lines.append("")

        # Results calculation
        total = (
            self.n_pp + self.n_pn + self.n_pc
            + self.n_np + self.n_nn + self.n_nc
            + self.n_cp + self.n_cn + self.n_cc
            + self.n_xp + self.n_xn + self.n_xc
            + self.n_nyr_p + self.n_nyr_n + self.n_nyr_c
            + self.n_nyr_nyr
        )
        correct = self.n_pp + self.n_nn + self.n_cc + self.n_nyr_nyr
        accuracy = (correct / total * 100) if total > 0 else 0

        lines.append(f"**Total Questions:** {total}")
        lines.append(f"**Correct Predictions:** {correct} ({accuracy:.1f}%)")
        lines.append("")

        # Detailed results section
        lines.append("## Detailed Results\n")
        lines.append("")

        # Sort question results by category, then by question ID
        sorted_results = sorted(
            self.question_results.values(),
            key=lambda x: (x.outcome_category or "", x.question_id)
        )

        for result in sorted_results:
            lines.append(f"### Question {result.question_id}\n")
            lines.append(f"**Title:** {result.question_title}")
            lines.append(f"**URL:** {result.question_url}")
            lines.append("")

            # Question contents
            lines.append(f"**Question Contents:**\n")
            lines.append("> " + result.question_text.replace("\n", "\n> "))
            lines.append("")

            # Resolution comparison table
            lines.append("| Output Resolution | Correct Resolution |")
            lines.append("|-------------------|--------------------|")
            actual_str = self._resolution_to_str(result.actual_resolution)
            predicted_str = self._resolution_to_str(result.predicted_resolution)
            lines.append(f"| {predicted_str} | {actual_str} |")
            lines.append("")

            # Key evidence section
            if result.key_evidence:
                lines.append("**Key Evidence:**")
                for evidence in result.key_evidence:
                    lines.append(f"- {evidence}")
                lines.append("")

            lines.append("---\n")

        return "\n".join(lines)

    def write_to_file(self, directory: str = "reports") -> Path:
        """
        Writes the detailed report to a markdown file in the specified directory.

        Args:
            directory: Directory path to write the report to (default: "reports")

        Returns:
            Path object pointing to the written file
        """
        from pathlib import Path

        # Create directory if it doesn't exist
        reports_dir = Path(directory)
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"assessment_report_{timestamp}.md"
        filepath = reports_dir / filename

        # Write report
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.detailed_report())

        logging.info(f"Assessment report written to {filepath}")
        return filepath

    @staticmethod
    def _resolution_to_str(resolution: Optional[ResolutionType]) -> str:
        """Convert a resolution to a human-readable string."""
        if resolution is None:
            return "NOT_YET_RESOLVABLE"
        elif isinstance(resolution, bool):
            return "TRUE" if resolution else "FALSE"
        elif isinstance(resolution, CanceledResolution):
            return str(resolution)
        else:
            return str(resolution)


    def __str__(self):
        return self.binary_results_table()

class ResolutionAssesser:
    """
    Utility for assessing how an auto resolver behaves on a set of already resolved questions.  

    Supports concurrent resolution of questions when assessing resolvers.
    """

    def __init__(self, resolver: AutoResolver, allowed_types: list[QuestionBasicType], questions: list[int | str] = [], tournaments: list[int | str] = [], max_concurrency: int = 3):
        """
        Initialize the resolution assessor.

        Args:
            resolver: AutoResolver to assess
            allowed_types: List of question types to include
            questions: Optional list of question IDs or URLs to assess
            tournaments: Optional list of tournament IDs to load
            max_concurrency: Maximum number of questions to resolve concurrently (default: 3)
        """
        self.client = MetaculusClient()
        self.questions: dict[int, MetaculusQuestion] = {}
        self.allowed_types = allowed_types
        self.resolver = resolver
        self.tournament_ids = tournaments
        self.question_ids = questions
        self._concurrency_limiter = asyncio.Semaphore(max_concurrency)


    def _insert_question(self, question: MetaculusQuestion):
        if question.actual_resolution_time is None:
            raise ValueError("Question is not yet resolved")
        if question.get_question_type() not in self.allowed_types:
            raise ValueError("Question is not of allowed type")
        if question.id_of_post is None:
            raise ValueError("Question does not have a post id")
        self.questions[question.id_of_post] = question

    def _load_question(self, question: int | str):
        loaded = None
        if type(question) is int:
            loaded = self.client.get_question_by_post_id(question)
        elif type(question) is str:
            loaded = self.client.get_question_by_url(question)
        else:
            return NotImplemented

        if loaded is None or not isinstance(loaded, MetaculusQuestion):
            raise ValueError("unable to find question")

        self._insert_question(loaded)

    def _load_questions(self, questions: list[int | str]):
        for question in questions:
            try:
                self._load_question(question)
            except ValueError as e:
                logging.warning(f"Skipping question {question}: {e}")
                continue
               
    async def _load_tournaments(self, tournament_ids: list[str | int]):
        filter = ApiFilter(
            allowed_tournaments=tournament_ids,
            allowed_statuses=["resolved"],
            allowed_types=self.allowed_types,
            group_question_mode="exclude",
            order_by="-published_time"
        )

        questions: list[MetaculusQuestion]  = await self.client.get_questions_matching_filter(filter)

        for question in questions:
            try:
                self._insert_question(question)
            except ValueError as e:
                question_id = getattr(question, 'id_of_post', 'unknown')
                logging.warning(f"Skipping question {question_id}: {e}")
                continue

    async def _load_tournament(self, tournament_id: str | int):
        await self._load_tournaments([tournament_id])

    async def _resolve_single_question(
        self, question: MetaculusQuestion, index: int
    ) -> tuple[int, Optional[ResolutionType], dict | None]:
        async with self._concurrency_limiter:
            resolution = await self.resolver.resolve_question(question)
            metadata = self.resolver.get_last_resolution_metadata()
            return (index, resolution, metadata)

    async def assess_resolver(self) -> ResolutionAssessmentReport:
        """
        Assess the resolver against the loaded questions.

        Uses concurrent execution (via asyncio.gather) to resolve multiple
        questions in parallel, controlled by a semaphore to limit concurrency.

        Returns:
            ResolutionAssessmentReport: Detailed report of resolver performance
        """
        report = ResolutionAssessmentReport()

        if self.tournament_ids:
            await self._load_tournaments(self.tournament_ids)
        if self.question_ids:
            self._load_questions(self.question_ids)

        question_items = list(self.questions.items())

        tasks = [
            self._resolve_single_question(question, idx)
            for idx, (question_id, question) in enumerate(question_items)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result, (question_id, question) in zip(results, question_items):
            if isinstance(result, Exception):
                logging.error(f"Exception resolving question {question_id}: {result}")
                continue

            index, test_resolution, metadata = result
            true_resolution = question.typed_resolution

            key_evidence = metadata.get("key_evidence") if metadata else None

            if isinstance(true_resolution, BinaryResolution):
                outcome_category = None

                match true_resolution, test_resolution:
                    # Positive actual resolution cases
                    case True, True:
                        report.pp.append(question_id)
                        outcome_category = "True Positive"
                    case True, False:
                        report.np.append(question_id)
                        outcome_category = "False Negative"
                    case True, None:
                        report.xp.append(question_id)
                        outcome_category = "Missed Positive"
                    case True, CanceledResolution():
                        report.cp.append(question_id)
                        outcome_category = "Positive Incorrectly Predicted as Cancelled"
                    # Negative actual resolution cases
                    case False, True:
                        report.pn.append(question_id)
                        outcome_category = "False Positive"
                    case False, False:
                        report.nn.append(question_id)
                        outcome_category = "True Negative"
                    case False, None:
                        report.xn.append(question_id)
                        outcome_category = "Missed Negative"
                    case False, CanceledResolution():
                        report.cn.append(question_id)
                        outcome_category = "Negative Incorrectly Predicted as Cancelled"
                    # Cancelled actual resolution cases
                    case CanceledResolution(), True:
                        report.pc.append(question_id)
                        outcome_category = "Cancelled Incorrectly Predicted as Positive"
                    case CanceledResolution(), False:
                        report.nc.append(question_id)
                        outcome_category = "Cancelled Incorrectly Predicted as Negative"
                    case CanceledResolution(), None:
                        report.xc.append(question_id)
                        outcome_category = "Cancelled Not Answered"
                    case CanceledResolution(), CanceledResolution():
                        report.cc.append(question_id)
                        outcome_category = "Correct Cancel"
                    # Catch-all for unmatched cases (edge cases, errors, NotImplemented, etc.)
                    case _:
                        logging.warning(
                            f"Question {question_id} had unmatched prediction: "
                            f"true_resolution={true_resolution}, test_resolution={test_resolution}"
                        )
                        if true_resolution == True:
                            report.xp.append(question_id)
                            outcome_category = "Unmatched - Positive"
                        elif true_resolution == False:
                            report.xn.append(question_id)
                            outcome_category = "Unmatched - Negative"
                        elif isinstance(true_resolution, CanceledResolution):
                            report.xc.append(question_id)
                            outcome_category = "Unmatched - Cancelled"
                        else:
                            continue

                # Create detailed result
                question_result = QuestionAssessmentResult(
                    question_id=question_id,
                    question_title=question.question_text[:100] if question.question_text else "No title",
                    question_text=question.question_text or "No text available",
                    question_url=question.page_url or f"https://metaculus.com/{question_id}",
                    actual_resolution=true_resolution,
                    predicted_resolution=test_resolution,
                    key_evidence=key_evidence,
                    outcome_category=outcome_category,
                )
                report.question_results[question_id] = question_result
        return report
