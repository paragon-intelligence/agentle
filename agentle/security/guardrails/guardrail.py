import abc

from rsb.models.base_model import BaseModel


class Guardrail(BaseModel, abc.ABC):
    @abc.abstractmethod
    def eval(self, inp: str) -> None: ...
