.. agentle documentation master file, created by
   sphinx-quickstart on Tue Apr 29 15:21:37 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Agentle Documentation
=====================

**Agentle**: A powerful yet elegant framework for building the next generation of AI agents.

Agentle makes it effortless to create, compose, and deploy intelligent AI agents - from simple task-focused agents to complex multi-agent systems. Built with developer productivity and type safety in mind, Agentle provides a clean, intuitive API for transforming cutting-edge AI capabilities into production-ready applications.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api/index
   examples
   contributing

Installation
-----------

.. code-block:: bash

   pip install agentle

Quick Start
----------

.. code-block:: python

   from agentle.agents.agent import Agent
   from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider

   # Create a simple agent
   agent = Agent(
       name="Quick Start Agent",
       generation_provider=GoogleGenaiGenerationProvider(),
       model="gemini-2.0-flash",
       instructions="You are a helpful assistant who provides concise, accurate information."
   )

   # Run the agent
   response = agent.run("What are the three laws of robotics?")

   # Print the response
   print(response.text)

Key Features
-----------

- **Simple Agent Creation** - Build powerful AI agents with minimal code
- **Composable Architecture** - Create sequential pipelines or dynamic teams of specialized agents
- **Tool Integration** - Seamlessly connect agents to external tools and functions
- **Structured Outputs** - Get strongly-typed responses with Pydantic integration
- **Ready for Production** - Deploy as APIs (BlackSheep), UIs (Streamlit), or embedded in apps
- **Built-in Observability** - Automatic tracing via Langfuse with extensible interfaces
- **Agent-to-Agent (A2A)** - Support for Google's standardized A2A protocol
- **Prompt Management** - Flexible system for organizing and managing prompts
- **Knowledge Integration** - Seamlessly incorporate static knowledge from various sources

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

