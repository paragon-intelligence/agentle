======================
Knowledge Integration
======================

Agentle provides a powerful knowledge integration system that allows agents to leverage information from various sources when generating responses. This feature is particularly useful for building specialized agents that need domain-specific knowledge beyond their pre-trained capabilities.

Basic Knowledge Integration
--------------------------

Here's a simple example of integrating knowledge with an agent:

.. code-block:: python

    from agentle.agents.agent import Agent
    from agentle.agents.knowledge.static_knowledge import StaticKnowledge
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    # Create an agent with static knowledge
    travel_expert = Agent(
        name="Japan Travel Expert",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You are a Japan travel expert who provides detailed information about Japanese destinations.",
        # Provide static knowledge from multiple sources
        static_knowledge=[
            # Include knowledge from local documents
            StaticKnowledge(content="data/japan_travel_guide.pdf", cache=3600),  # Cache for 1 hour
            # Include knowledge from websites
            StaticKnowledge(content="https://www.japan-guide.com/", cache="infinite"),  # Cache indefinitely
            # Include direct text knowledge
            "Tokyo is the capital of Japan and one of the most populous cities in the world."
        ]
    )

    # The agent will incorporate the knowledge when answering
    response = travel_expert.run("What should I know about visiting Tokyo in cherry blossom season?")
    print(response.text)

Knowledge Source Types
--------------------

The framework supports multiple knowledge source types through the ``StaticKnowledge`` class:

* **Documents**: PDF, DOCX, TXT, PPTX, and other document formats
* **URLs**: Web pages and online resources
* **Raw Text**: Direct text snippets

You can provide knowledge in several ways:

1. **As strings**: The framework will automatically convert them to ``StaticKnowledge`` objects

   .. code-block:: python

       agent = Agent(
           # ... other agent settings ...
           static_knowledge=[
               # URLs as strings
               "https://example.com/research-paper.html",
               # Local file paths as strings
               "data/report.pdf",
               "references/definitions.txt",
           ]
       )

2. **As StaticKnowledge objects**: For maximum control, especially with caching

   .. code-block:: python

       from agentle.agents.knowledge.static_knowledge import StaticKnowledge

       agent = Agent(
           # ... other agent settings ...
           static_knowledge=[
               # Document with 1 hour cache
               StaticKnowledge(content="data/report.pdf", cache=3600),
               # URL with infinite cache
               StaticKnowledge(content="https://example.com/api-docs", cache="infinite"),
               # Raw text with no cache
               StaticKnowledge(content="This is raw knowledge text", cache=None),
           ]
       )

Caching Behavior
--------------

The caching system works as follows:

1. If ``cache`` is not specified or is ``None``, content is parsed fresh each time
2. If ``cache`` is an integer, the content is cached for that many seconds (requires aiocache)
3. If ``cache`` is the string "infinite", the content is cached indefinitely until the process ends (requires aiocache)

To enable caching, you'll need to install the optional aiocache package:

.. code-block:: bash

    pip install aiocache

Caching is particularly useful for large documents or URLs that are expensive to parse repeatedly.

How Knowledge Integration Works
-----------------------------

When you provide static knowledge to an agent:

1. The agent uses appropriate document parsers to extract content from each knowledge source
2. If caching is enabled and the aiocache package is installed, parsed content is cached for the specified duration
3. The parsed content is organized into a structured knowledge base format
4. This knowledge base is appended to the agent's instructions
5. When the agent responds to queries, it can leverage this knowledge base

Custom Document Parsers
---------------------

For specialized knowledge extraction needs, you can provide a custom document parser to the agent:

.. code-block:: python

    from agentle.agents.agent import Agent
    from agentle.agents.knowledge.static_knowledge import StaticKnowledge
    from agentle.parsing.parsers.file_parser import FileParser

    # Create a custom document parser with specialized settings
    custom_parser = FileParser(
        strategy="high",  # Use high-detail parsing
        visual_description_agent=your_custom_vision_agent  # Customize image analysis
    )

    # Create an agent with the custom parser
    research_agent = Agent(
        # ... other agent settings ...
        static_knowledge=[
            StaticKnowledge(content="research_papers/paper.pdf", cache=3600),
            # ... other knowledge sources ...
        ],
        document_parser=custom_parser
    )

You can also create completely custom document parsers by implementing the ``DocumentParser`` abstract base class:

.. code-block:: python

    from typing import override
    from pathlib import Path
    from agentle.parsing.document_parser import DocumentParser
    from agentle.parsing.parsed_document import ParsedFile
    from agentle.parsing.section_content import SectionContent

    # Create a custom parser
    class CustomParser(DocumentParser):
        """Custom document parser implementation"""
        
        @override
        async def parse_async(self, document_path: str) -> ParsedFile:
            # Implement your custom parsing logic here
            path = Path(document_path)
            
            # For this example, we'll just use a placeholder
            parsed_content = f"Content from {path.name} would be parsed with custom logic"
            
            # Return in the standard ParsedFile format
            return ParsedFile(
                name=path.name,
                sections=[
                    SectionContent(
                        number=1,
                        text=parsed_content,
                        md=parsed_content
                    )
                ]
            )

    # Use the custom parser with an agent
    agent = Agent(
        name="Document Expert",
        # ... other agent settings ...
        static_knowledge=[
            StaticKnowledge(content="documents/report.pdf", cache="infinite")
        ],
        # Pass your custom parser to the agent
        document_parser=CustomParser()
    )

Practical Example: Legal Assistant
--------------------------------

Here's a comprehensive example showing how to create a legal assistant with domain-specific knowledge:

.. code-block:: python

    from agentle.agents.agent import Agent
    from agentle.agents.knowledge.static_knowledge import StaticKnowledge
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider
    from agentle.parsing.factories.file_parser_default_factory import file_parser_default_factory

    # Create a legal assistant with domain-specific knowledge
    legal_assistant = Agent(
        name="Legal Assistant",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You are a legal assistant specialized in contract law. Help users understand legal concepts and review contracts.",
        
        # Provide multiple knowledge sources with different caching strategies
        static_knowledge=[
            # Local document sources with caching
            StaticKnowledge(content="legal_docs/contract_templates.pdf", cache=3600),  # Cache for 1 hour
            StaticKnowledge(content="legal_docs/legal_definitions.docx", cache="infinite"),  # Cache indefinitely
            
            # Online resources with caching
            StaticKnowledge(content="https://www.law.cornell.edu/wex/contract", cache=86400),  # Cache for 1 day
            
            # Direct knowledge snippets (no need for caching)
            "Force majeure clauses excuse a party from performance when extraordinary events prevent fulfillment of obligations."
        ],
        
        # Optional: Use a custom document parser for specialized parsing needs
        document_parser=file_parser_default_factory(strategy="high")
    )

    # The agent will leverage all provided knowledge when responding
    response = legal_assistant.run("What should I look for in a non-disclosure agreement?")
    print(response.text)

Best Practices
------------

1. **Caching Strategy**: Use appropriate caching based on how frequently the source changes
2. **Knowledge Organization**: Organize knowledge into related documents rather than one large document
3. **Quality over Quantity**: Provide high-quality, relevant knowledge rather than overwhelming the agent
4. **Test Different Sources**: Experiment with different knowledge sources to find the best combination
5. **Update Regularly**: Keep knowledge sources updated, especially for domains that change frequently