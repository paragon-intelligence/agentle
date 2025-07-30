==================
Prompt Management
==================

Agentle provides a flexible system for managing prompts, allowing you to organize, version, and reuse prompt templates across your application. This page explains how to use the prompt management system.

Basic Prompt Management
----------------------

At the core of Agentle's prompt management is the ``Prompt`` class and various prompt providers:

.. code-block:: python

    from agentle.prompts.models.prompt import Prompt
    from agentle.prompts.prompt_providers.fs_prompt_provider import FSPromptProvider

    # Create a prompt provider that loads prompts from files
    prompt_provider = FSPromptProvider(base_path="./prompts")

    # Load a prompt
    weather_prompt = prompt_provider.provide("weather_template")

    # Compile the prompt with variables
    compiled_prompt = weather_prompt.compile(
        location="Tokyo",
        units="celsius",
        days=5
    )

    # Use the prompt with an agent
    agent.run(compiled_prompt)

Creating Prompt Templates
-----------------------

Prompt templates are text files with variable placeholders that can be filled in at runtime. By default, Agentle uses a simple ``{{variable}}`` syntax for variables:

.. code-block:: text
    :caption: ./prompts/weather_template.md

    Please provide a detailed {{days}}-day weather forecast for {{location}} in {{units}} degrees.
    Include information about:
    
    1. Temperature highs and lows
    2. Precipitation chances
    3. Wind conditions
    4. Any special weather alerts
    
    Make the forecast concise but informative.

You can organize your prompts in a directory structure:

.. code::

    prompts/
    ├── travel/
    │   ├── itinerary.txt
    │   └── city_guide.txt
    ├── coding/
    │   ├── python_explanation.txt
    │   └── code_review.txt
    └── weather_template.txt

Prompt Providers
--------------

Agentle supports multiple prompt providers for different storage mechanisms:

File System Provider
~~~~~~~~~~~~~~~~~

Store prompts as files on disk:

.. code-block:: python

    from agentle.prompts.prompt_providers.fs_prompt_provider import FSPromptProvider

    # Load prompts from a directory
    fs_provider = FSPromptProvider(base_path="./prompts")

    # Load a prompt (will look for ./prompts/weather_template.txt)
    prompt = fs_provider.provide("weather_template")
    
    # Load from a subdirectory (will look for ./prompts/travel/itinerary.txt)
    itinerary_prompt = fs_provider.provide("travel/itinerary")

In-Memory Provider
~~~~~~~~~~~~~~~

Define prompts directly in your code:

.. code-block:: python

    from agentle.prompts.prompt_providers.memory_prompt_provider import MemoryPromptProvider
    from agentle.prompts.models.prompt import Prompt

    # Create prompts
    greeting_prompt = Prompt(template="Hello {{name}}! How can I help you today?")
    farewell_prompt = Prompt(template="Goodbye {{name}}. Have a great day!")

    # Create an in-memory provider with these prompts
    memory_provider = MemoryPromptProvider({
        "greeting": greeting_prompt,
        "farewell": farewell_prompt
    })

    # Load a prompt
    prompt = memory_provider.provide("greeting")

    # Compile with variables
    compiled = prompt.compile(name="Alice")
    print(compiled)  # "Hello Alice! How can I help you today?"

Database Provider
~~~~~~~~~~~~~~

Store and retrieve prompts from various database systems, both SQL and NoSQL:

.. code-block:: python

    from agentle.prompts.prompt_providers.db_prompt_provider import DBPromptProvider
    
    # SQLite example
    import sqlite3
    
    # Connect to a database
    conn = sqlite3.connect("prompts.db")
    
    # Create a table for prompts (if it doesn't exist)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            name TEXT PRIMARY KEY,
            template TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert some prompts
    conn.execute(
        "INSERT OR REPLACE INTO prompts (name, template) VALUES (?, ?)",
        ("greeting", "Hello {{name}}! How can I help you today?")
    )
    conn.commit()
    
    # Create a database prompt provider
    db_provider = DBPromptProvider(
        connection=conn,
        query="SELECT template FROM prompts WHERE name = ?",
        param_conversion=lambda name: (name,)
    )
    
    # Load a prompt from the database
    prompt = db_provider.provide("greeting")

The DBPromptProvider supports a wide range of database systems:

**Relational Databases:**

.. code-block:: python

    # PostgreSQL (with asyncpg)
    import asyncpg
    
    async def get_postgres_prompt():
        conn = await asyncpg.connect(
            user="postgres", password="password",
            database="mydb", host="localhost"
        )
        
        provider = DBPromptProvider(
            connection=conn,
            query="SELECT content FROM prompt_templates WHERE name = $1"
        )
        
        return provider.provide("welcome_email")
    
    # MySQL/MariaDB
    import mysql.connector
    
    conn = mysql.connector.connect(
        host="localhost", user="user",
        password="password", database="mydb"
    )
    
    mysql_provider = DBPromptProvider(
        connection=conn,
        query="SELECT template FROM prompts WHERE name = %s"
    )

**NoSQL Databases:**

.. code-block:: python

    # MongoDB
    from pymongo import MongoClient
    
    client = MongoClient("mongodb://localhost:27017/")
    
    mongo_provider = DBPromptProvider(
        connection=client,
        query="prompt_db.templates",  # Format: "database.collection"
        param_conversion=lambda prompt_id: {"name": prompt_id}
    )
    
    # Redis
    import redis
    
    r = redis.Redis(host="localhost", port=6379, db=0)
    
    # Add a prompt to Redis
    r.set("prompt:greeting", "Hello {{name}}! Welcome to our service.")
    
    redis_provider = DBPromptProvider(
        connection=r,
        query="",  # Query is ignored for Redis
        param_conversion=lambda prompt_id: f"prompt:{prompt_id}"
    )
    
    # Couchbase
    from couchbase.cluster import Cluster
    from couchbase.auth import PasswordAuthenticator
    
    auth = PasswordAuthenticator("username", "password")
    cluster = Cluster("couchbase://localhost", authenticator=auth)
    
    couchbase_provider = DBPromptProvider(
        connection=cluster,
        query="prompt_bucket.prompts",  # Format: "bucket.collection"
        param_conversion=lambda prompt_id: prompt_id
    )

The provider automatically detects the database type and uses the appropriate query mechanism. It works with both synchronous and asynchronous database clients, making it adaptable to various application architectures.

Using Prompts with Agents
-----------------------

Prompts can be used with agents in several ways:

As Direct Input
~~~~~~~~~~~~~

.. code-block:: python

    # Load and compile a prompt
    support_prompt = prompt_provider.provide("customer_support")
    compiled_prompt = support_prompt.compile(
        customer_name="John",
        issue_type="billing",
        account_id="ACC12345"
    )
    
    # Use as direct input to an agent
    response = agent.run(compiled_prompt)

In Agent Instructions
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Load and compile a prompt for agent instructions
    instruction_prompt = prompt_provider.provide("coding_assistant_instructions")
    compiled_instructions = instruction_prompt.compile(
        language="Python",
        framework="FastAPI",
        coding_style="PEP8"
    )
    
    # Create an agent with the compiled instructions
    agent = Agent(
        name="Coding Assistant",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions=compiled_instructions
    )


Advanced Prompt Templates
-----------------------

You can create more complex templates with conditions, loops, and other logic:

.. code-block:: text
    :caption: ./prompts/report_template.txt

    # {{report_type}} Report
    
    {{#if customer_name}}
    Prepared for: {{customer_name}}
    {{/if}}
    
    ## Summary
    {{summary}}
    
    ## Details
    {{#each data_points}}
    - {{this.name}}: {{this.value}}
    {{/each}}
    
    {{#if include_recommendations}}
    ## Recommendations
    {{recommendations}}
    {{/if}}
    
    Generated on {{date}}

This more advanced template uses Handlebars-like syntax for conditional sections and loops.

Creating Custom Prompt Providers
-----------------------------

You can create custom prompt providers by implementing the ``PromptProvider`` interface:

.. code-block:: python

    from typing import override
    from agentle.prompts.prompt_provider import PromptProvider
    from agentle.prompts.models.prompt import Prompt
    
    class CustomPromptProvider(PromptProvider):
        """Custom prompt provider implementation"""
        
        def __init__(self, api_key: str):
            self.api_key = api_key
            # Initialize your custom prompt storage/service
            
        @override
        def provide(self, name: str) -> Prompt:
            """Provide a prompt by name"""
            # Implement your custom logic to retrieve prompts
            # For example, fetching from an API or cloud storage
            
            template = self._fetch_from_api(name)
            return Prompt(template=template)
            
        def _fetch_from_api(self, name: str) -> str:
            """Fetch a prompt template from an external API"""
            # Implementation...
            return template
    
    # Use the custom provider
    custom_provider = CustomPromptProvider(api_key="your-api-key")
    prompt = custom_provider.provide("welcome_message")

Best Practices
------------

1. **Organize by Domain**: Group prompts by domain/functionality
2. **Version Control**: Keep prompts in version control alongside code
3. **Test Prompts**: Verify prompt effectiveness with test cases
4. **Documentation**: Document expected variables and use cases
5. **Reuse**: Build a library of reusable prompt components
6. **Monitor Performance**: Track which prompts perform best and iterate