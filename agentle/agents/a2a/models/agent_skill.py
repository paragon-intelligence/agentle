"""
A2A Agent Skill Model

This module defines the AgentSkill class, which represents a specific capability that an agent
can perform. Skills are used to describe what agents can do and help in agent discovery and selection.
"""

import uuid
from collections.abc import Sequence

from rsb.models.base_model import BaseModel
from rsb.models.field import Field
from rsb.models.mimetype import MimeType


class AgentSkill(BaseModel):
    """
    Represents a specific capability that an agent can perform.

    Agent skills are used to describe what agents can do and help in agent discovery
    and selection. Each skill has a unique identifier, a name, and a description.
    Additional metadata such as tags, examples, and supported interaction modes
    can also be included.

    Attributes:
        id: Unique identifier for the agent's skill
        name: Human readable name of the skill
        description: Description of what the skill does
        tags: Optional tags categorizing the skill capability
        examples: Optional example scenarios illustrating the skill
        inputModes: Optional supported input interaction modes
        outputModes: Optional supported output interaction modes

    Example:
        ```python
        from agentle.agents.a2a.models.agent_skill import AgentSkill
        from rsb.models.mimetype import MimeType

        # Create a simple skill
        translation_skill = AgentSkill(
            name="Language Translation",
            description="Translates text between different languages",
            tags=["language", "translation"]
        )

        # Create a more detailed skill with input/output modes
        code_generation_skill = AgentSkill(
            name="Code Generation",
            description="Generates code based on natural language descriptions",
            tags=["programming", "code", "generation"],
            examples=[
                "Write a Python function to calculate Fibonacci numbers",
                "Create a JavaScript function to sort an array of objects by a property"
            ],
            inputModes=[MimeType.TEXT_PLAIN],
            outputModes=[MimeType.APPLICATION_JAVASCRIPT, MimeType.TEXT_X_PYTHON]
        )
        ```
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """
    Unique identifier for the agent's skill
    """

    name: str
    """
    Human readable name of the skill
    """

    description: str
    """
    Description of the skill - will be used by the client or a human
    as a hint to understand what the skill does.
    """

    tags: Sequence[str] = Field(default_factory=list)
    """
    Set of tagwords describing classes of capabilities for this specific skill
    """

    examples: Sequence[str] | None = Field(default=None)
    """
    Set of example scenarios that the skill can perform.
    """

    inputModes: Sequence[MimeType] | None = Field(default=None)
    """
    Set of interaction modes that the skill supports for input.
    """

    outputModes: Sequence[MimeType] | None = Field(default=None)
    """
    Set of interaction modes that the skill supports for output.
    """
