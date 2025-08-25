from __future__ import annotations

import json
from collections.abc import (
    AsyncGenerator,
    Generator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from contextlib import asynccontextmanager, contextmanager
from textwrap import dedent
from typing import TYPE_CHECKING, Any, cast

from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.private_attr import PrivateAttr

from agentle.agents.agent import Agent
from agentle.agents.agent_run_output import AgentRunOutput
from agentle.agents.context import Context
from agentle.agents.templates.data_collection.collected_data import CollectedData
from agentle.agents.templates.data_collection.field_spec import FieldSpec
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.providers.base.generation_provider_type import (
    GenerationProviderType,
)
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.tools.tool import Tool
from agentle.generations.tools.tool_execution_result import ToolExecutionResult

if TYPE_CHECKING:
    from agentle.agents.agent_input import AgentInput
    from agentle.generations.models.generation.trace_params import TraceParams


class ProgressiveProfilingAgent(BaseModel):
    """A stateless agent specialized in progressive data collection using structured outputs"""

    field_specs: Sequence[FieldSpec]
    generation_provider: GenerationProviderType
    model: str | None = None
    max_attempts_per_field: int = 3

    # Private attributes
    _agent: Agent[CollectedData] | None = PrivateAttr(default=None)

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    @property
    def agent(self) -> Agent[CollectedData]:
        if self._agent is None:
            raise ValueError("ERROR: Agent is not set.")
        return self._agent

    def model_post_init(self, context: Any) -> None:
        """Initialize the internal agent with instructions for structured output"""
        # Build field descriptions for instructions
        field_descriptions = self._build_field_descriptions()

        # Create specialized instructions
        instructions = dedent(f"""\
        You are a friendly data collection specialist focused on progressive profiling.
        Your goal is to collect all required information from the user in a natural, conversational way.

        ## Fields to Collect:
        {field_descriptions}

        ## Guidelines:
        1. Be conversational and friendly while collecting information
        2. Ask for one or a few related fields at a time, not all at once
        3. Extract and validate any information the user provides
        4. If a user provides multiple pieces of information at once, extract all of them
        5. Be flexible - users might provide information in any order
        6. Handle corrections gracefully if users want to update previously provided information
        7. Provide a conversational response while collecting data

        ## Response Format:
        You MUST always return a CollectedData object with:
        - fields: A dictionary containing ALL collected data (both previous and new)
        - pending_fields: List of field names still needed (only required fields)
        - completed: Whether all required fields have been collected

        ## Current State:
        The current state of collected data will be provided in the conversation.
        You should:
        1. Acknowledge what has already been collected
        2. Extract any new information from the user's message
        3. Merge new data with existing data in your response
        4. Update the pending_fields list accordingly
        5. Set completed=true only when all required fields are collected

        ## Validation Rules:
        - string: Any text value
        - integer: Must be a valid whole number
        - float: Must be a valid decimal number
        - boolean: Accept "yes", "no", "true", "false", "1", "0"
        - email: Must contain @ and a domain
        - date: Accept common date formats

        Always provide a natural, conversational response while ensuring the structured data is complete and accurate.
        """)

        # Create the internal agent with structured output only
        self._agent = Agent(
            name="Progressive Profiling Agent",
            description="An agent that progressively collects user information through conversation",
            generation_provider=self.generation_provider,
            model=self.model or self.generation_provider.default_model,
            instructions=instructions,
            response_schema=CollectedData,
        )

    def run(
        self,
        input: AgentInput | Any,
        *,
        current_state: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        trace_params: TraceParams | None = None,
    ) -> AgentRunOutput[CollectedData]:
        """
        Run the progressive profiling agent

        Args:
            input: The user input (can be string, Context, sequence of messages, etc.)
            current_state: Current collected data state (if None, starts fresh)
            timeout: Optional timeout for the operation
            trace_params: Optional trace parameters

        Returns:
            AgentRunOutput[CollectedData] with the updated state
        """
        # Build the enhanced input with state context
        enhanced_input = self._enhance_input_with_state(input, current_state or {})

        # Run the internal agent
        result = self.agent.run(
            enhanced_input, timeout=timeout, trace_params=trace_params
        )

        # Post-process the result to ensure data validity
        if result.parsed:
            result.parsed = self._validate_collected_data(result.parsed)

        return result

    async def run_async(
        self,
        input: AgentInput | Any,
        *,
        current_state: Mapping[str, Any] | None = None,
        trace_params: TraceParams | None = None,
    ) -> AgentRunOutput[CollectedData]:
        """
        Run the progressive profiling agent asynchronously

        Args:
            input: The user input (can be string, Context, sequence of messages, etc.)
            current_state: Current collected data state (if None, starts fresh)
            trace_params: Optional trace parameters

        Returns:
            AgentRunOutput[CollectedData] with the updated state
        """
        # Build the enhanced input with state context
        enhanced_input = self._enhance_input_with_state(input, current_state or {})

        # Run the internal agent
        result = await self.agent.run_async(enhanced_input, trace_params=trace_params)

        # Post-process the result to ensure data validity
        if result.parsed:
            result.parsed = self._validate_collected_data(result.parsed)

        return result

    @contextmanager
    def start_mcp_servers(self) -> Generator[None, None, None]:
        """Start MCP servers for the internal agent"""
        with self.agent.start_mcp_servers():
            yield

    @asynccontextmanager
    async def start_mcp_servers_async(self) -> AsyncGenerator[None, None]:
        """Start MCP servers asynchronously for the internal agent"""
        async with self.agent.start_mcp_servers_async():
            yield

    def _enhance_input_with_state(
        self, input: AgentInput | Any, current_state: Mapping[str, Any]
    ) -> AgentInput | Any:
        """
        Enhance the input with current state information

        This method prepends state context to the input while preserving
        the original input type when possible.
        """
        # Calculate pending fields
        pending_fields = self._get_pending_fields(current_state)

        # Build state summary
        state_summary = {
            "collected_data": dict(current_state),
            "pending_required_fields": pending_fields,
            "total_fields": len(self.field_specs),
            "required_fields": len([fs for fs in self.field_specs if fs.required]),
            "optional_fields": len([fs for fs in self.field_specs if not fs.required]),
        }

        # Create state context message
        state_context = dedent(f"""\
        ## Current Collection State:
        {json.dumps(state_summary, indent=2)}
        
        Please analyze the user input, extract any relevant field values, 
        and return the complete CollectedData object with all collected fields 
        (both previous and new).
        """)

        # Check if input is a Context object
        if hasattr(input, "message_history"):  # Duck typing for Context
            # Add state message to the context's message history
            state_message = UserMessage(parts=[TextPart(text=state_context)])
            try:
                # Clone context and prepend message
                new_messages = [state_message] + list(
                    cast(Context, input).message_history
                )
                cast(Context, input).message_history = new_messages
                return input
            except Exception:
                # If we can't modify the context, convert to messages list
                return [state_message] + list(cast(Context, input).message_history)

        # Check if input is a sequence (list or tuple)
        if isinstance(input, (list, tuple)) and input:
            first_item = input[0]

            # Check if it's a sequence of messages
            if isinstance(
                first_item, (UserMessage, AssistantMessage, DeveloperMessage)
            ):
                # Return a list of messages with state message prepended
                state_message = UserMessage(parts=[TextPart(text=state_context)])
                return [state_message] + list(input)

            # Check if it's a sequence of parts (TextPart, FilePart, etc.)
            elif isinstance(
                first_item,
                (
                    TextPart,
                    FilePart,
                    Tool,
                    ToolExecutionSuggestion,
                    ToolExecutionResult,
                ),
            ):
                # Create a UserMessage with state part and input parts
                state_part = TextPart(text=state_context)
                li = list(
                    cast(
                        Sequence[
                            TextPart
                            | FilePart
                            | Tool[Any]
                            | ToolExecutionSuggestion
                            | ToolExecutionResult
                        ],
                        input,
                    )
                )
                return UserMessage(parts=[state_part] + li)

        # Check if input is a single message
        if isinstance(input, (UserMessage, AssistantMessage, DeveloperMessage)):
            # Return a list with state message and the input message
            state_message = UserMessage(parts=[TextPart(text=state_context)])
            return [state_message, input]

        # Check if input is a single part (TextPart or FilePart)
        if isinstance(input, (TextPart, FilePart)):
            # Create a UserMessage with both parts
            state_part = TextPart(text=state_context)
            return UserMessage(parts=[state_part, input])

        # For simple string inputs
        if isinstance(input, str):
            return f"{state_context}\n\n## User Input:\n{input}"

        # For all other types (including non-iterable ones), convert to string
        return f"{state_context}\n\n## User Input:\n{str(input)}"

    def _validate_collected_data(self, data: CollectedData) -> CollectedData:
        """
        Validate and clean the collected data

        Ensures all field values are properly typed and validated
        """
        validated_fields: MutableMapping[str, Any] = {}

        for field_name, value in data.fields.items():
            # Find the field spec
            field_spec = next(
                (fs for fs in self.field_specs if fs.name == field_name), None
            )

            if field_spec:
                try:
                    # Convert and validate the value
                    converted_value = self._convert_value(value, field_spec.type)
                    validated_fields[field_name] = converted_value
                except ValueError:
                    # Skip invalid values
                    pass

        # Update the data with validated fields
        data.fields = validated_fields

        # Recalculate pending fields and completion status
        data.pending_fields = self._get_pending_fields(validated_fields)
        data.completed = self._check_completion(validated_fields)

        return data

    def _build_field_descriptions(self) -> str:
        """Build a formatted description of all fields to collect"""
        descriptions: MutableSequence[str] = []

        for spec in self.field_specs:
            desc = f"- **{spec.name}** ({spec.type})"
            if spec.required:
                desc += " [REQUIRED]"
            desc += f": {spec.description}"

            if spec.examples:
                desc += f"\n  Examples: {', '.join(spec.examples)}"

            if spec.validation:
                desc += f"\n  Validation: {spec.validation}"

            descriptions.append(desc)

        return "\n".join(descriptions)

    def _convert_value(self, value: Any, field_type: str) -> Any:
        """Convert a value to the specified type"""
        if field_type == "string":
            return str(value)
        elif field_type == "integer":
            return int(value)
        elif field_type == "float":
            return float(value)
        elif field_type == "boolean":
            if isinstance(value, str):
                return value.lower() in ("true", "yes", "1", "on")
            return bool(value)
        elif field_type == "email":
            # Basic email validation
            email_str = str(value).strip().lower()
            if "@" not in email_str or "." not in email_str.split("@")[1]:
                raise ValueError("Invalid email format")
            return email_str
        elif field_type == "date":
            # Could use dateutil.parser here for more robust parsing
            return str(value)  # Simplified for now
        else:
            return value

    def _check_completion(self, collected_data: Mapping[str, Any]) -> bool:
        """Check if all required fields have been collected"""
        for spec in self.field_specs:
            if spec.required and spec.name not in collected_data:
                return False
        return True

    def _get_pending_fields(
        self, collected_data: Mapping[str, Any]
    ) -> MutableSequence[str]:
        """Get list of pending required fields"""
        pending: MutableSequence[str] = []
        for spec in self.field_specs:
            if spec.required and spec.name not in collected_data:
                pending.append(spec.name)
        return pending

    def get_field_status(self, collected_data: Mapping[str, Any]) -> str:
        """
        Get a human-readable status of the collection progress

        Args:
            collected_data: The current state of collected data

        Returns:
            A formatted string showing collection status
        """
        collected: MutableSequence[str] = []
        pending: MutableSequence[str] = []

        for spec in self.field_specs:
            if spec.name in collected_data:
                collected.append(f"✓ {spec.name}: {collected_data[spec.name]}")
            elif spec.required:
                pending.append(f"○ {spec.name} ({spec.type}) - {spec.description}")
            else:
                pending.append(
                    f"○ {spec.name} ({spec.type}) [optional] - {spec.description}"
                )

        result = "## Collection Status:\n\n"

        if collected:
            result += "### Collected:\n" + "\n".join(collected) + "\n\n"

        if pending:
            result += "### Still Needed:\n" + "\n".join(pending)
        else:
            result += "### All required fields have been collected! ✓"

        return result


if __name__ == "__main__":
    from agentle.generations.providers.google.google_generation_provider import (
        GoogleGenerationProvider,
    )

    # Define the fields to collect
    user_profile_fields = [
        FieldSpec(
            name="full_name",
            type="string",
            description="User's full name",
            examples=["John Doe", "Jane Smith"],
        ),
        FieldSpec(
            name="email",
            type="email",
            description="User's email address",
            validation="Must be a valid email format",
        ),
        FieldSpec(
            name="age",
            type="integer",
            description="User's age",
            validation="Must be between 13 and 120",
        ),
        FieldSpec(
            name="interests",
            type="string",
            description="User's interests or hobbies",
            required=False,
            examples=["reading", "sports", "cooking"],
        ),
    ]

    # Create the progressive profiling agent
    profiler = ProgressiveProfilingAgent(
        field_specs=user_profile_fields,
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
    )

    # State is managed externally
    current_state: Mapping[str, Any] = {}

    # Start collecting data
    response = profiler.run(
        "Hi! I'd like to sign up for your service.", current_state=current_state
    )
    print(response)

    # # Update state from response
    # if response.parsed:
    #     current_state = response.parsed.fields
    #     print(f"Current state: {current_state}")

    # # Continue the conversation with updated state
    # response = profiler.run("My name is John Doe", current_state=current_state)
    # print(response.text)

    # # Check progress
    # if response.parsed:
    #     current_state = response.parsed.fields
    #     print(f"Collected: {response.parsed.fields}")
    #     print(f"Still needed: {response.parsed.pending_fields}")
    #     print(f"Complete: {response.parsed.completed}")

    # # Continue until all fields are collected
    # while response.parsed and not response.parsed.completed:
    #     user_input = input("You: ")
    #     response = profiler.run(user_input, current_state=current_state)
    #     print(f"Agent: {response.text}")

    #     if response.parsed:
    #         current_state = response.parsed.fields

    # # Final collected data
    # print(f"Profile complete! Collected data: {current_state}")
