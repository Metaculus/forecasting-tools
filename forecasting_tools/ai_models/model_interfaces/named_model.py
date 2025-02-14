from __future__ import annotations

from abc import ABC


class NamedModel(ABC):
    MODEL_NAME: str

    def __init_subclass__(cls: type[NamedModel], **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if ABC not in cls.__bases__ and cls.MODEL_NAME is NotImplemented:
            raise NotImplementedError("You forgot to define MODEL_NAME")
