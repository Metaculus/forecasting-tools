from __future__ import annotations

from abc import ABC, abstractmethod


class QuestionResponder(ABC):
    NAME: str = NotImplemented
    DESCRIPTION_OF_WHEN_TO_USE: str = NotImplemented

    def __init__(self, question: str) -> None:
        if not question:
            raise ValueError("Question cannot be empty")
        if len(question) > 1000:
            raise ValueError("Question is too long")
        self.question = question

    def __init_subclass__(cls: type[QuestionResponder], **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if ABC not in cls.__bases__:
            if cls.DESCRIPTION_OF_WHEN_TO_USE is NotImplemented:
                raise NotImplementedError(
                    "You forgot to define DESCRIPTION_OF_WHEN_TO_USE"
                )
            if cls.NAME is NotImplemented:
                raise NotImplementedError("You forgot to define NAME")

    @abstractmethod
    async def respond_with_markdown(self) -> str:
        pass
