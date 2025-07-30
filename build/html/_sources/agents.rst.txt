=======
Agents
=======

The core building block of Agentle is the ``Agent`` class. This page explains how to create and customize agents for different use cases.

Basic Agent Creation
-------------------

Here's how to create a basic agent:

.. code-block:: python

    from agentle.agents.agent import Agent
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    # Create a general-purpose agent
    agent = Agent(
        name="Basic Agent",
        description="A helpful assistant for general purposes.",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You are a helpful assistant who provides accurate information."
    )

Agent Parameters
--------------

The ``Agent`` class accepts the following key parameters:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Parameter
     - Description
   * - ``name``
     - A unique name for the agent
   * - ``description``
     - Optional description of the agent's purpose and capabilities
   * - ``generation_provider``
     - The provider that handles generation (e.g., GoogleGenerationProvider)
   * - ``model``
     - The specific model to use (e.g., "gemini-2.5-flash")
   * - ``instructions``
     - Detailed instructions that guide the agent's behavior
   * - ``tools``
     - Optional list of tools/functions the agent can use
   * - ``response_schema``
     - Optional Pydantic model for structured outputs
   * - ``static_knowledge``
     - Optional knowledge sources to enhance the agent's capabilities
   * - ``document_parser``
     - Optional custom parser for knowledge documents

Using ModelKind for Provider Independence
----------------------------------------

Agentle provides a powerful abstraction called ``ModelKind`` that decouples your code from specific provider model names. Instead of using provider-specific model identifiers, you can use standardized capability categories:

.. code-block:: python

    from agentle.agents.agent import Agent
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    # Using provider-specific model name
    agent1 = Agent(
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",  # Only works with Google
        instructions="You are a helpful assistant."
    )

    # Using ModelKind for provider-agnostic code
    agent2 = Agent(
        generation_provider=GoogleGenerationProvider(),
        model="category_standard",  # Works with any provider
        instructions="You are a helpful assistant."
    )

Available ModelKind Categories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - ModelKind
     - Description
   * - ``category_nano``
     - Smallest, fastest, most cost-effective models (e.g., GPT-4.1 nano)
   * - ``category_mini``
     - Small but capable models (e.g., GPT-4.1 mini, Claude Haiku)
   * - ``category_standard``
     - Mid-range, balanced performance (e.g., Claude Sonnet, Gemini Flash)
   * - ``category_pro``
     - High performance models (e.g., Gemini Pro, GPT-4 Turbo)
   * - ``category_flagship``
     - Best available models from each provider (e.g., Claude Opus, GPT-4.5)
   * - ``category_reasoning``
     - Specialized for complex reasoning tasks
   * - ``category_vision``
     - Optimized for multimodal capabilities with image/video processing
   * - ``category_coding``
     - Specialized for programming tasks
   * - ``category_instruct``
     - Fine-tuned for instruction following

Benefits of ModelKind
~~~~~~~~~~~~~~~~~~~

Using ModelKind provides several important benefits:

1. **Provider Independence**: Your code works with any AI provider without modification
2. **Future-Proof**: When providers release new models, only the internal mapping tables need to be updated
3. **Capability-Based Selection**: Select models based on capabilities rather than provider-specific names
4. **Simplified Failover**: When using ``FailoverGenerationProvider``, each provider automatically maps to its equivalent model

.. code-block:: python

    # Create a failover provider with multiple underlying providers
    from agentle.generations.providers.failover.failover_generation_provider import FailoverGenerationProvider
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider
    from agentle.generations.providers.openai.openai import OpenaiGenerationProvider

    failover = FailoverGenerationProvider(
        generation_providers=[
            GoogleGenerationProvider(),
            OpenaiGenerationProvider(api_key="your-openai-key")
        ]
    )

    # Using a specific model would fail with providers that don't support it
    # agent = Agent(generation_provider=failover, model="gpt-4o")  # Will fail for Google

    # Using ModelKind ensures compatibility across all providers
    agent = Agent(
        generation_provider=failover,
        model="category_pro",  # Mapped to appropriate model by each provider
        instructions="You are a helpful assistant."
    )

How ModelKind Works
~~~~~~~~~~~~~~~~~

Behind the scenes, Agentle uses a decorator that:

1. Intercepts calls to the provider's ``generate_async`` method
2. Checks if the model parameter is a ModelKind value
3. Calls the provider's ``map_model_kind_to_provider_model`` method to get the provider-specific model name
4. Substitutes this mapped value before the actual provider method is called

Each provider implements its own mapping function to translate ModelKind values to the most appropriate model for that provider.

Creating Specialized Agents
--------------------------

You can create agents specialized for particular domains by customizing the instructions and other parameters:

.. code-block:: python

    # Create a travel agent
    travel_agent = Agent(
        name="Travel Guide",
        description="A helpful travel guide that answers questions about destinations.",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="""You are a knowledgeable travel guide who helps users plan trips.
        You provide information about destinations, offer travel tips, suggest itineraries,
        and answer questions about local customs, attractions, and practical travel matters."""
    )

    # Create a coding assistant
    coding_agent = Agent(
        name="Coding Assistant",
        description="An expert in writing and debugging code across multiple languages.",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="""You are a coding expert who helps with programming tasks.
        You can write code, debug issues, explain concepts, and provide best practices
        across languages like Python, JavaScript, Java, C++, and others."""
    )

Running Agents
-------------

The primary way to interact with agents is through the ``run`` method:

.. code-block:: python

    # Simple string input
    result = agent.run("What is the capital of France?")
    print(result.text)

    # With a custom message
    from agentle.generations.models.messages.user_message import UserMessage
    from agentle.generations.models.message_parts.text import TextPart

    message = UserMessage(parts=[TextPart(text="Tell me about Paris")])
    result = agent.run(message)
    print(result.text)


Agent Response Structure
----------------------

When you call ``agent.run()``, you get back a response object with these key properties:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Property
     - Description
   * - ``text``
     - The text response from the agent
   * - ``parsed``
     - The structured output (if a response_schema was provided)
   * - ``generation``
     - The complete generation object with the agent's response

Advanced Agent Configuration
--------------------------

For more advanced use cases, you can:

* Add tools to enable function calling capabilities
* Incorporate static knowledge from documents or URLs
* Define structured output schemas with Pydantic
* Combine agents into pipelines or teams
* Deploy agents as APIs or UIs

These topics are covered in detail in their respective documentation sections.