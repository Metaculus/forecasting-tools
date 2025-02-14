from abc import ABC, abstractmethod


class TokensAreCalculatable(ABC):

    @abstractmethod
    def input_to_tokens(self, *args, **kwargs) -> int:
        pass
