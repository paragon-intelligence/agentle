import datetime

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class SchedulePlan(BaseModel):
    """Configuration for scheduling future calls"""

    earliest_at: datetime.datetime = Field(description="Earliest time to make the call")
    latest_at: datetime.datetime | None = Field(
        default=None, description="Latest time to attempt the call"
    )
    timezone: str = Field(default="UTC", description="Timezone for scheduling")
    retry_attempts: int = Field(
        default=3, ge=1, le=10, description="Number of retry attempts"
    )
    retry_interval_minutes: int = Field(
        default=15, ge=1, le=1440, description="Minutes between retries"
    )
