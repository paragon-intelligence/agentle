from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class SentenceSegment(BaseModel):
    """A segment of a transcribed audio file.

    Attributes:
        id: The unique identifier of the segment
        sentence: The transcribed text of the segment
        start: The start time of the segment in seconds
        end: The end time of the segment in seconds
        no_speech_prob: The probability that there is no speech in the segment

    Example:
        >>> SentenceSegment(id=0, sentence="Hello, world!", start=0.0, end=1.5, no_speech_prob=0.0)
    """

    id: int = Field(description="The unique identifier of the segment.")
    sentence: str = Field(description="The transcribed text of the segment.")
    start: float = Field(description="The start time of the segment in seconds.")
    end: float = Field(description="The end time of the segment in seconds.")
    no_speech_prob: float = Field(
        description="The probability that there is no speech in the segment."
    )
