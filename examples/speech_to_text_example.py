#!/usr/bin/env python
"""
Example of using the Gladia Speech-to-Text provider with agentle.

This example demonstrates how to transcribe an audio file using the Gladia
Speech-to-Text provider. Make sure to set your GLADIA_API_KEY environment
variable before running this example.

Usage:
    python speech_to_text_example.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from agentle.stt.models.transcription_config import TranscriptionConfig
from agentle.stt.providers.google.google_speech_to_text_provider import (
    GoogleSpeechToTextProvider,
)

load_dotenv()

# Add parent directory to path to import agentle
sys.path.append(str(Path(__file__).resolve().parent.parent))


# Initialize the provider
provider = GoogleSpeechToTextProvider()

# Path to the sample audio file
audio_file = Path(__file__).parent / "harvard.wav"

print(f"Transcribing {audio_file}...")

# Configure transcription parameters
config = TranscriptionConfig(
    timeout=60.0,  # Increase timeout for longer files
    context_prompt="This is a sample audio file containing speech",
)

try:
    # Run the async transcribe method in a synchronous context
    result = provider.transcribe(str(audio_file), config)

    # Display the transcription results
    print("\n=== Transcription Results ===")
    print(f"Full Text: {result.text}")
    print(f"Duration: {result.duration:.2f} seconds")

    print("\n=== Segments ===")
    for segment in result.segments:
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.sentence}")

    print("\n=== Subtitles (SRT) ===")
    print(result.subtitles.subtitles)

except Exception as e:
    print(f"Error during transcription: {e}")
