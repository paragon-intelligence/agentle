from __future__ import annotations

from collections.abc import Sequence

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.stt.models.sentence_segment import SentenceSegment
from agentle.stt.models.subtitles import Subtitles


class AudioTranscription(BaseModel):
    """Result of audio-to-text transcription.

    Attributes:
        text: Full transcribed text
        segments: Timed text segments with confidence
        cost: API cost in USD
        duration: Audio length in seconds

    Example:
        >>> transcription = AudioTranscription(
        ...     text="Hello world",
        ...     segments=[...],
        ...     cost=0.02
        ... )
    """

    text: str = Field(title="Text", description="The transcribed text.")

    segments: Sequence[SentenceSegment] = Field(
        title="Parts", description="The transcribed text broken down into segments."
    )

    cost: float = Field(
        title="Cost", description="The cost incurred by the transcriptions request."
    )

    duration: float = Field(
        title="Duration",
        description="The duration of the audio file in seconds. May not be precise.",
    )

    subtitles: Subtitles = Field(
        title="Subtitles",
        description="The subtitles of the audio file.",
    )

    def merge(
        self, *audio_transcriptions: "AudioTranscription"
    ) -> "AudioTranscription":
        def segments_to_srt(segments: Sequence[SentenceSegment]) -> str:
            """
            Converts a sequence of SentenceSegment objects into SRT (SubRip Text) subtitle format.

            This function takes a sequence of `SentenceSegment` objects, each representing a segment of transcribed speech with start and end times,
            and formats them into a SRT subtitle string. SRT is a widely used subtitle format that includes an index, time range, and subtitle text for each segment.

            Args:
                segments (Sequence[SentenceSegment]): A sequence of SentenceSegment objects, each containing `sentence`, `start`, and `end` attributes.

            Returns:
                str: A string containing the segments formatted as SRT subtitles.

            Example:
                >>> from intellibricks.llms.types import SentenceSegment
                >>> segments_example = [
                ...     SentenceSegment(sentence="Hello world.", start=0.0, end=2.5),
                ...     SentenceSegment(sentence="This is a test subtitle.", start=3.0, end=6.7)
                ... ]
                >>> srt_content = segments_to_srt(segments_example)
                >>> print(srt_content)
                1
                00:00:00,000 --> 00:00:02,500
                Hello world.

                2
                00:00:03,000 --> 00:00:06,700
                This is a test subtitle.
            """

            def format_time(seconds: float) -> str:
                hours = int(seconds // 3600)
                remaining = seconds % 3600
                minutes = int(remaining // 60)
                remaining %= 60
                seconds_int = int(remaining)
                milliseconds = int((remaining - seconds_int) * 1000)
                return f"{hours:02}:{minutes:02}:{seconds_int:02},{milliseconds:03}"

            parts: list[str] = []
            for idx, segment in enumerate(segments, start=1):
                start_time = format_time(segment.start)
                end_time = format_time(segment.end)
                part = f"{idx}\n{start_time} --> {end_time}\n{segment.sentence}"
                parts.append(part)

            return "\n\n".join(parts)

        all_transcriptions = [self] + list(audio_transcriptions)
        merged_text = " ".join(t.text for t in all_transcriptions)
        merged_cost = sum(t.cost for t in all_transcriptions)
        merged_duration = sum(t.duration for t in all_transcriptions)
        merged_segments: list[SentenceSegment] = []
        for i, current_transcription in enumerate(all_transcriptions):
            current_offset = sum(t.duration for t in all_transcriptions[:i])
            for segment in current_transcription.segments:
                new_start = segment.start + current_offset
                new_end = segment.end + current_offset
                new_id = len(merged_segments)
                new_segment = SentenceSegment(
                    id=new_id,
                    sentence=segment.sentence,
                    start=new_start,
                    end=new_end,
                    no_speech_prob=segment.no_speech_prob,
                )
                merged_segments.append(new_segment)

        return AudioTranscription(
            text=merged_text,
            segments=merged_segments,
            cost=merged_cost,
            duration=merged_duration,
            subtitles=Subtitles(
                format="srt",
                subtitles=segments_to_srt(merged_segments),
            ),
        )
