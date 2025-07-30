==========
Agent API
==========

This page documents the API for the ``Agent`` class, which is the core building block of the Agentle framework.
This page documentation and others in this module are a WIP. If you want to check the internal API, for now, I recommend going directly to the code.

Agent Class
----------

.. code-block:: python

    class Agent:
        def __init__(
            self,
            name: str,
            generation_provider: GenerationProvider,
            model: str,
            description: Optional[str] = None,
            instructions: Optional[str] = None,
            tools: Optional[List[Callable]] = None,
            response_schema: Optional[Type[BaseModel]] = None,
            response_format: Optional[ResponseFormat] = None,
            static_knowledge: Optional[List[Union[str, StaticKnowledge]]] = None,
            document_parser: Optional[DocumentParser] = None,
            validation_behavior: ValidationBehavior = ValidationBehavior.STRICT,
            skills: Optional[List[AgentSkill]] = None,
        ):
            """
            Initialize an agent.
            
            Args:
                name: A unique name for the agent
                generation_provider: The provider that handles generation
                model: The specific model to use
                description: Optional description of the agent's purpose
                instructions: Detailed instructions that guide the agent's behavior
                tools: Optional list of tools/functions the agent can use
                response_schema: Optional Pydantic model for structured outputs
                response_format: Optional format for structured outputs
                static_knowledge: Optional knowledge sources to enhance capabilities
                document_parser: Optional custom parser for knowledge documents
                validation_behavior: Behavior when validation fails for structured outputs
                skills: Optional list of skills for agent-to-agent communication
            """
            # Implementation details...

Core Methods
-----------

run()
~~~~~

.. code-block:: python

    def run(
        self,
        input_data: Any,
        trace_params: Optional[Dict[str, Any]] = None
    ) -> AgentRunResult:
        """
        Run the agent with the given input.
        
        Args:
            input_data: The input to the agent (string, dataframe, image, etc.)
            trace_params: Optional parameters for tracing the generation
            
        Returns:
            An AgentRunResult containing the agent's response
        """
        # Implementation details...

The ``run()`` method accepts various input types, including:

- **Strings**: Simple text queries
- **DataFrames**: Pandas DataFrames (converted to markdown tables)
- **Images**: PIL Images or image files (for multimodal models)
- **Dictionaries/JSON**: Converted to JSON strings
- **NumPy arrays**: Formatted for readability
- **Dates/Times**: Formatted as ISO strings
- **File paths**: Content is extracted and processed
- **Pydantic models**: Serialized to JSON
- **File-like objects**: Content is extracted and processed
- **Custom message structures**: UserMessage, AssistantMessage, etc.

reset()
~~~~~~

.. code-block:: python

    def reset(self) -> None:
        """
        Reset the agent's conversation history.
        """
        # Implementation details...

This method clears the agent's conversation history, starting a fresh conversation.

Other Methods
-----------

prepare_context()
~~~~~~~~~~~~~~~

.. code-block:: python

    def prepare_context(
        self,
        input_data: Any
    ) -> Context:
        """
        Prepare a context object from input data.
        
        Args:
            input_data: The input to convert to a context
            
        Returns:
            A Context object ready for generation
        """
        # Implementation details...

This method converts input data to a Context object which contains the conversation history and other metadata.

with_tools()
~~~~~~~~~~

.. code-block:: python

    def with_tools(
        self,
        tools: List[Callable]
    ) -> Agent:
        """
        Create a new agent with additional tools.
        
        Args:
            tools: List of tools to add to the agent
            
        Returns:
            A new Agent instance with the added tools
        """
        # Implementation details...

This method creates a new agent with the same configuration plus additional tools.

with_static_knowledge()
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def with_static_knowledge(
        self,
        static_knowledge: List[Union[str, StaticKnowledge]]
    ) -> Agent:
        """
        Create a new agent with additional static knowledge.
        
        Args:
            static_knowledge: List of knowledge sources to add
            
        Returns:
            A new Agent instance with the added knowledge
        """
        # Implementation details...

This method creates a new agent with the same configuration plus additional static knowledge.

AgentRunResult
-------------

The ``run()`` method returns an ``AgentRunResult`` object with the following properties:

.. code-block:: python

    class AgentRunResult:
        """Result of running an agent."""
        
        @property
        def text(self) -> str:
            """Get the text response from the agent."""
            # Implementation details...
            
        @property
        def parsed(self) -> Optional[BaseModel]:
            """Get the structured output if a response_schema was provided."""
            # Implementation details...
            
        @property
        def message(self) -> Message:
            """Get the complete message object."""
            # Implementation details...
            
        @property
        def raw(self) -> Any:
            """Get the raw response from the provider."""
            # Implementation details...

Example Usage
------------

Basic Usage
~~~~~~~~~~

.. code-block:: python

    from agentle.agents.agent import Agent
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    # Create a basic agent
    agent = Agent(
        name="Basic Agent",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You are a helpful assistant."
    )

    # Run the agent
    result = agent.run("What is the capital of France?")
    print(result.text)  # "The capital of France is Paris."

With Tools
~~~~~~~~~

.. code-block:: python

    def get_weather(location: str) -> str:
        """Get weather for a location."""
        # Implementation details...
        return f"Simulated weather data for {location}"

    # Create an agent with a tool
    agent_with_tool = Agent(
        name="Weather Agent",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You provide weather information.",
        tools=[get_weather]
    )

    result = agent_with_tool.run("What's the weather in Tokyo?")

With Structured Output
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pydantic import BaseModel
    from typing import List

    class WeatherForecast(BaseModel):
        location: str
        temperature: float
        conditions: str
        forecast: List[str]

    # Create an agent with structured output
    structured_agent = Agent(
        name="Structured Weather Agent",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You provide weather forecasts.",
        response_schema=WeatherForecast
    )

    result = structured_agent.run("What's the weather in Tokyo?")
    forecast = result.parsed
    print(f"Temperature: {forecast.temperature}°C")

With Static Knowledge
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from agentle.agents.knowledge.static_knowledge import StaticKnowledge

    # Create an agent with static knowledge
    knowledgeable_agent = Agent(
        name="Knowledgeable Agent",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You are an expert on company policies.",
        static_knowledge=[
            StaticKnowledge(content="docs/company_policy.pdf", cache=3600),
            "The company was founded in 2020."
        ]
    )

    result = knowledgeable_agent.run("What is our vacation policy?")

Advanced Configuration
-------------------

Validation Behavior
~~~~~~~~~~~~~~~~~

You can control how the agent handles validation errors for structured outputs:

.. code-block:: python

    from agentle.agents.validation_behavior import ValidationBehavior

    # Create an agent with custom validation behavior
    agent = Agent(
        name="Validating Agent",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You provide structured data.",
        response_schema=YourResponseSchema,
        validation_behavior=ValidationBehavior.WARN  # Options: STRICT, WARN, IGNORE
    )

Agent Skills
~~~~~~~~~~

For use with agent teams, you can define agent skills:

.. code-block:: python

    from agentle.agents.a2a.models.agent_skill import AgentSkill

    # Create an agent with skills
    agent = Agent(
        name="Skilled Agent",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You have specialized skills.",
        skills=[
            AgentSkill(name="code-generation", description="Write code in various languages"),
            AgentSkill(name="debugging", description="Find and fix bugs in code")
        ]
    )