import abc


class Textfyable(abc.ABC):
    @abc.abstractmethod
    def get_text(self) -> str: ...
