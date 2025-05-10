=======
Agents
=======

The core building block of Agentle is the ``Agent`` class. This page explains how to create and customize agents for different use cases.

Basic Agent Creation
-------------------

Here's how to create a basic agent:

.. code-block:: python

    from agentle.agents.agent import Agent
    from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider

    # Create a general-purpose agent
    agent = Agent(
        name="Basic Agent",
        description="A helpful assistant for general purposes.",
        generation_provider=GoogleGenaiGenerationProvider(),
        model="gemini-2.0-flash",
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
     - The provider that handles generation (e.g., GoogleGenaiGenerationProvider)
   * - ``model``
     - The specific model to use (e.g., "gemini-2.0-flash")
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

Creating Specialized Agents
--------------------------

You can create agents specialized for particular domains by customizing the instructions and other parameters:

.. code-block:: python

    # Create a travel agent
    travel_agent = Agent(
        name="Travel Guide",
        description="A helpful travel guide that answers questions about destinations.",
        generation_provider=GoogleGenaiGenerationProvider(),
        model="gemini-2.0-flash",
        instructions="""You are a knowledgeable travel guide who helps users plan trips.
        You provide information about destinations, offer travel tips, suggest itineraries,
        and answer questions about local customs, attractions, and practical travel matters."""
    )

    # Create a coding assistant
    coding_agent = Agent(
        name="Coding Assistant",
        description="An expert in writing and debugging code across multiple languages.",
        generation_provider=GoogleGenaiGenerationProvider(),
        model="gemini-2.0-flash",
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