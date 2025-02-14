from __future__ import annotations

from abc import ABC


class PricedPerRequest(ABC):
    PRICE_PER_REQUEST: float = NotImplemented

    def __init_subclass__(cls: type[PricedPerRequest], **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if (
            ABC not in cls.__bases__
            and cls.PRICE_PER_REQUEST is NotImplemented
        ):
            raise NotImplementedError("You forgot to define PRICE_PER_REQUEST")
