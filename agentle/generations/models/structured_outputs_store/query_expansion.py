from rsb.decorators.value_objects import valueobject
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


@valueobject
class QueryExpansion(BaseModel):
    expanded_query: str | None = Field(
        default=None,
        description="The expanded query. Or null.",
    )
