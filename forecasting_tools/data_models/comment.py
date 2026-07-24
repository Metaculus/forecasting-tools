from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class Comment(BaseModel):
    """A comment on a Metaculus post.

    A bot publishing a report posts the report's explanation as the comment
    text, so ``text`` parses with the same functions that read a saved
    ``ForecastReport``.

    ``on_post`` is a post id, not a question id. The two are separate sequences
    that overlap, so joining question-level data on ``on_post`` silently matches
    the wrong question rather than matching nothing.
    """

    id: int
    on_post: int
    author_id: int
    author_username: str
    created_at: datetime
    text: str
    is_private: bool

    @classmethod
    def from_metaculus_api_json(cls, comment_json: dict) -> Comment:
        author = comment_json["author"]
        return cls(
            id=comment_json["id"],
            on_post=comment_json["on_post"],
            author_id=author["id"],
            author_username=author["username"],
            created_at=comment_json["created_at"],
            text=comment_json["text"],
            is_private=comment_json["is_private"],
        )
