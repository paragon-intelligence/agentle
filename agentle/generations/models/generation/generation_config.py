from agentle.generations.models.generation.trace_params import TraceParams
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class GenerationConfig(BaseModel):
    temperature: float | None = Field(default=None)
    max_output_tokens: int | None = Field(default=None)
    n: int = Field(default=1)
    top_p: float | None = Field(default=None)
    top_k: float | None = Field(default=None)
    google_generation_kwargs: dict[str, object] | None = Field(default=None)
    trace_params: TraceParams = Field(default_factory=lambda: TraceParams())
    timeout: float | None = Field(default=None)

    class Config:
        arbitrary_types_allowed = True
