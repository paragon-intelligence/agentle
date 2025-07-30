======
Inputs
======

Agentle agents can process a wide variety of input types out-of-the-box, making it simple to work with different data formats without complex conversions.

Basic Input Types
---------------

Here are the most common input types you can pass to the ``run`` method:

.. code-block:: python

    from agentle.agents.agent import Agent
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    # Create a basic agent
    agent = Agent(
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You are a versatile assistant that can analyze different types of data."
    )

    # String input (simplest case)
    agent.run("What is the capital of Japan?")

    # Pandas DataFrame
    import pandas as pd
    df = pd.DataFrame({
        "Country": ["Japan", "France", "USA"],
        "Capital": ["Tokyo", "Paris", "Washington DC"],
        "Population": [126.3, 67.8, 331.9]
    })
    agent.run(df)  # Automatically converts to markdown table

    # Image input (for multimodal models)
    from PIL import Image
    img = Image.open("chart.png")
    agent.run(img)  # Automatically handles image format

    # Dictionary/JSON
    user_data = {
        "name": "Alice",
        "interests": ["AI", "Python", "Data Science"],
        "experience_years": 5
    }
    agent.run(user_data)  # Automatically formats as JSON

Advanced Input Types
------------------

Agentle also supports more specialized input formats:

.. code-block:: python

    # NumPy array
    import numpy as np
    data = np.array([[1, 2, 3], [4, 5, 6]])
    agent.run(data)  # Automatically formats array

    # Date and time
    from datetime import datetime
    agent.run(datetime.now())  # Formatted as ISO string

    # File path
    from pathlib import Path
    agent.run(Path("report.txt"))  # Reads and processes file content

    # Pydantic model
    from pydantic import BaseModel
    
    class UserProfile(BaseModel):
        name: str
        age: int
        interests: list[str]

    profile = UserProfile(name="Bob", age=28, interests=["AI", "Robotics"])
    agent.run(profile)  # Automatically formats model as JSON

    # File-like objects
    from io import StringIO, BytesIO
    text_io = StringIO("This is some text data from a stream")
    agent.run(text_io)  # Reads content from StringIO

Custom Message Structures
-----------------------

For more control, you can create custom message structures:

.. code-block:: python

    from agentle.generations.models.messages.user_message import UserMessage
    from agentle.generations.models.messages.assistant_message import AssistantMessage
    from agentle.generations.models.messages.developer_message import DeveloperMessage
    from agentle.generations.models.message_parts.text import TextPart

    # Create a conversation with multiple message types
    messages = [
        # System instructions (not visible to the user)
        DeveloperMessage(parts=[
            TextPart(text="You are a helpful travel assistant that speaks in a friendly tone.")
        ]),
        
        # User's initial message
        UserMessage(parts=[
            TextPart(text="I'm planning a trip to Japan in April.")
        ]),
        
        # Previous assistant response in the conversation
        AssistantMessage(parts=[
            TextPart(text="That's a wonderful time to visit Japan! Cherry blossoms should be in bloom.")
        ]),
        
        # User's follow-up question
        UserMessage(parts=[
            TextPart(text="What cities should I visit for the best cherry blossom viewing?")
        ])
    ]

    # Pass the complete conversation to the agent
    result = agent.run(messages)

Multi-Part Messages
-----------------

Each message can contain multiple parts of different types, enabling rich multimodal interactions:

.. code-block:: python

    from agentle.generations.models.messages.user_message import UserMessage
    from agentle.generations.models.message_parts.text import TextPart
    from agentle.generations.models.message_parts.file import FilePart
    from agentle.generations.tools.tool import Tool

    # Define a simple weather tool
    def get_weather(location: str) -> str:
        """Get weather for a location"""
        return f"Simulated weather data for {location}"

    # Create a message with different part types
    message = UserMessage(
        parts=[
            # Text part for regular text input
            TextPart(text="Can you analyze this image and data?"),
            
            # File part for image analysis (multimodal models)
            FilePart(
                data=open("vacation_photo.jpg", "rb").read(),
                mime_type="image/jpeg"
            ),
        ]
    )

    # Run the agent with the multi-part message
    result = agent.run(message)

Context Object
------------

For maximum control, you can create a Context object to manage complete conversations:

.. code-block:: python

    from agentle.agents.context import Context
    from agentle.generations.models.messages.user_message import UserMessage
    from agentle.generations.models.messages.developer_message import DeveloperMessage
    from agentle.generations.models.message_parts.text import TextPart
    from agentle.agents.step import Step

    # Create a custom context with specific messages
    context = Context(
        messages=[
            DeveloperMessage(parts=[
                TextPart(text="You are a travel planning assistant with expertise in budgeting.")
            ]),
            UserMessage(parts=[
                TextPart(text="I want to plan a 7-day trip to Europe with a $3000 budget.")
            ])
        ],
        # Optionally track conversation steps
        steps=[
            Step(type="user_input", content="Initial travel budget query")
        ]
    )

    # Run the agent with the custom context
    result = agent.run(context)

Automatic Type Conversion
-----------------------

Agentle automatically handles type conversion for most input types:

1. DataFrames are converted to markdown tables
2. Images are encoded appropriately for multimodal models
3. JSON/dictionaries are formatted as JSON strings
4. NumPy arrays are formatted to be readable
5. File paths are read and content is extracted
6. Pydantic models are serialized to JSON
7. File-like objects have their content extracted

This automatic conversion simplifies working with different data types and allows you to focus on your application logic rather than data formatting.