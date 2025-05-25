from typing import Literal


type ContextState = Literal[
    "initialized", "running", "paused", "completed", "failed", "cancelled"
]
