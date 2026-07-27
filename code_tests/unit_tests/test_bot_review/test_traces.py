from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from forecasting_tools.bot_review.outcomes import OutcomeTable, QuestionOutcome
from forecasting_tools.bot_review.traces import attach_traces, get_trace, reduce_comment
from forecasting_tools.data_models.comment import Comment

RUN_TIME = datetime(2026, 6, 30, 10, tzinfo=timezone.utc)

EXPLANATION = """
# SUMMARY
*Question*: Q1
*Final Prediction*: 20.0%
*Total Cost*: $0.0706 (estimated)
*Time Spent*: 0.28 minutes
*Bot Name*: TemplateBot

## Report 1 Summary
### Forecasts
*Forecaster 1*: 20.0%
*Forecaster 2 (gpt-5)*: 30.0%

# RESEARCH
## Report 1 Research
the research

# FORECASTS
## R1: Forecaster 1 Reasoning
the first rationale
## R1: Forecaster 2 Reasoning
the second rationale
"""


def make_comment(
    comment_id: int = 1,
    post_id: int = 1,
    created_at: datetime = RUN_TIME,
    text: str = EXPLANATION,
) -> Comment:
    return Comment(
        id=comment_id,
        on_post=post_id,
        author_id=7,
        author_username="bot",
        created_at=created_at,
        text=text,
        is_private=True,
    )


def make_question(post_id: int = 1, title: str = "Q1") -> QuestionOutcome:
    return QuestionOutcome(
        post_id=post_id,
        question_id=post_id + 100,
        title=title,
        url=f"https://www.metaculus.com/questions/{post_id}/",
        question_type="binary",
        status="resolved",
        resolution="yes",
        actual_resolve_time=None,
        spot_scoring_time=None,
        options=None,
        forecasted=True,
        forecast={"probability_yes": 0.2},
        forecast_time=None,
        scores=None,
        project_id=1,
        project_slug="testcup",
        project_name="TestCup",
    )


def make_table(questions) -> OutcomeTable:
    return OutcomeTable(
        generated_at=RUN_TIME, user_id=7, questions=questions, project_name="TestCup"
    )


class TestReduceComment:
    def test_reads_the_metadata_the_bot_writes_into_its_report(self):
        trace = reduce_comment(make_comment())
        assert trace.comment_id == 1
        assert trace.run_time == RUN_TIME
        assert trace.question_text == "Q1"
        assert trace.cost == 0.0706
        assert trace.minutes == 0.28
        assert not trace.truncated

    def test_reads_each_forecaster_prediction(self):
        trace = reduce_comment(make_comment())
        assert [f["key"] for f in trace.forecasters] == ["R1:F1", "R1:F2"]
        assert trace.forecasters[1]["prediction"] == "30.0%"

    def test_missing_metadata_is_none_not_an_error(self):
        trace = reduce_comment(make_comment(text="# SUMMARY\nnothing useful here"))
        assert trace.cost is None
        assert trace.minutes is None
        assert trace.question_text is None
        assert trace.forecasters == []

    def test_a_truncated_comment_is_flagged(self):
        text = "# SUMMARY\n\n---\n\n The comment size exceeded max size and has been truncated"
        assert reduce_comment(make_comment(text=text)).truncated


class TestAttachTraces:
    def client_returning(self, *comments) -> MagicMock:
        """A client whose comments are all private, as most bots post them."""
        client = MagicMock()
        client.get_own_comments.side_effect = lambda **kwargs: (
            list(comments) if kwargs.get("is_private") else []
        )
        return client

    def test_comments_are_read_in_bulk_not_once_per_post(self):
        questions = [make_question(1), make_question(2), make_question(3)]
        client = self.client_returning(make_comment())
        attach_traces(make_table(questions), client)
        assert client.get_own_comments.call_count == 2
        assert [
            c.kwargs["is_private"] for c in client.get_own_comments.call_args_list
        ] == [
            True,
            False,
        ]

    def test_public_comments_are_read_too(self):
        question = make_question(1)
        client = MagicMock()
        client.get_own_comments.side_effect = lambda **kwargs: (
            [] if kwargs.get("is_private") else [make_comment(comment_id=9)]
        )
        attach_traces(make_table([question]), client)
        assert [t.comment_id for t in question.traces] == [9]

    def test_comments_on_other_posts_are_ignored(self):
        question = make_question(1)
        client = self.client_returning(make_comment(post_id=1), make_comment(post_id=2))
        attach_traces(make_table([question]), client)
        assert len(question.traces) == 1

    def test_a_group_post_splits_its_comments_by_question_text(self):
        first = make_question(title="Q1")
        second = make_question(title="other")
        client = self.client_returning(
            make_comment(comment_id=1),
            make_comment(
                comment_id=2,
                text=EXPLANATION.replace("*Question*: Q1", "*Question*: other"),
            ),
        )
        attach_traces(make_table([first, second]), client)
        assert [t.comment_id for t in first.traces] == [1]
        assert [t.comment_id for t in second.traces] == [2]

    def test_a_single_question_post_keeps_every_run(self):
        question = make_question(1)
        client = self.client_returning(
            make_comment(comment_id=1), make_comment(comment_id=2)
        )
        attach_traces(make_table([question]), client)
        assert len(question.traces) == 2


class TestSelectingTheScoredRun:
    def question_with_runs(self, spot_offset_hours: float | None, *run_offsets: float):
        question = make_question(1).model_copy(
            update={
                "spot_scoring_time": (
                    None
                    if spot_offset_hours is None
                    else RUN_TIME + timedelta(hours=spot_offset_hours)
                )
            }
        )
        question.traces = [
            reduce_comment(
                make_comment(
                    comment_id=i, created_at=RUN_TIME + timedelta(hours=offset)
                )
            )
            for i, offset in enumerate(run_offsets)
        ]
        return question

    def test_the_latest_run_standing_at_spot_time_is_the_one_that_scored(self):
        question = self.question_with_runs(5, 0, 2, 9)
        assert question.trace is not None
        assert question.trace.comment_id == 1

    def test_a_run_after_spot_time_did_not_earn_the_score(self):
        question = self.question_with_runs(5, 9)
        assert question.traces
        assert question.trace is None

    def test_without_a_spot_time_the_latest_run_is_used(self):
        question = self.question_with_runs(None, 0, 9)
        assert question.trace is not None
        assert question.trace.comment_id == 1

    def test_no_runs_at_all(self):
        assert make_question(1).trace is None

    def test_traces_survive_a_json_round_trip(self):
        table = make_table([self.question_with_runs(5, 0)])
        reloaded = OutcomeTable.model_validate_json(table.model_dump_json())
        assert reloaded.questions[0].trace is not None
        assert reloaded.questions[0].trace.forecasters[0]["key"] == "R1:F1"


class TestGetTrace:
    def client_returning(self, *comments) -> MagicMock:
        """A client whose comments are all private, as most bots post them."""
        client = MagicMock()
        client.get_own_comments.side_effect = lambda **kwargs: (
            list(comments) if kwargs.get("is_private") else []
        )
        return client

    def test_returns_the_asked_for_section(self):
        client = self.client_returning(make_comment())
        assert "the research" in get_trace(1, "research", client=client)

    def test_returns_one_forecaster_rationale(self):
        client = self.client_returning(make_comment())
        text = get_trace(1, forecaster="R1:F2", client=client)
        assert "the second rationale" in text
        assert "the first rationale" not in text

    def test_reads_the_newest_comment_on_the_post(self):
        client = self.client_returning(
            make_comment(comment_id=1, text=EXPLANATION.replace("the research", "old")),
            make_comment(comment_id=2, created_at=RUN_TIME + timedelta(hours=1)),
        )
        assert "the research" in get_trace(1, "research", client=client)

    def test_reads_the_run_that_scored_when_given_its_comment_id(self):
        client = self.client_returning(
            make_comment(
                comment_id=1, text=EXPLANATION.replace("the research", "the scored run")
            ),
            make_comment(comment_id=2, created_at=RUN_TIME + timedelta(hours=1)),
        )
        assert "the scored run" in get_trace(1, "research", comment_id=1, client=client)

    def test_an_unknown_comment_id_returns_nothing(self):
        client = self.client_returning(make_comment(comment_id=1))
        assert get_trace(1, comment_id=99, client=client) == ""

    def test_a_post_with_no_comments(self):
        assert get_trace(1, client=self.client_returning()) == ""
