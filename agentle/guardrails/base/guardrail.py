import abc

from agentle.guardrails.guardrail_validation_error import GuardrailValidationError


from rsb.containers.result import Result


class Guardrail(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, input: str) -> Result[None, GuardrailValidationError]: ...
