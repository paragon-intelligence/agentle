from dataclasses import dataclass, field
from typing import override
from rsb.adapters.adapter import Adapter
from agentle.generations.providers.amazon.models.tool import Tool as BedrockTool
from pydantic import BaseModel

from agentle.generations.providers.amazon.models.tool_input_schema import (
    ToolInputSchema,
)
from agentle.generations.providers.amazon.models.tool_specification import (
    ToolSpecification,
)


@dataclass(frozen=True)
class ResponseSchemaToBedrockToolAdapter[T](Adapter[type[T], BedrockTool]):
    tool_description: str | None = field(default=None)

    @override
    def adapt(self, _f: type[T]) -> BedrockTool:
        name = _f.__name__
        if not issubclass(_f, BaseModel):
            raise ValueError(
                "Input must be a Pydantic model class. Only "
                + "pydantic models are supported for now"
            )

        input_schema = _f.model_json_schema()
        tool = BedrockTool(
            toolSpec=ToolSpecification(
                name=name,
                description=self.tool_description or "Structured Output",
                inputSchema=ToolInputSchema(json=input_schema),
            )
        )

        return tool
