==========
Tools API
==========

This page documents the API for working with tools (function calling) in Agentle.

Tool Classes
-----------

Agentle provides several classes for working with tools:

Tool
~~~~

The ``Tool`` class is the core class for tool integration:

.. code-block:: python

    class Tool:
        """
        Represents a tool that can be used by an agent.
        """
        
        @staticmethod
        def from_callable(
            callable: Callable
        ) -> "Tool":
            """
            Create a Tool from a callable (function or method).
            
            Args:
                callable: The function or method to convert to a tool
                
            Returns:
                A Tool instance
            """
            # Implementation details...


FunctionDefinition
~~~~~~~~~~~~~~~~

Defines the signature of a function:

.. code-block:: python

    class FunctionDefinition:
        """
        Represents the definition of a function that can be called.
        """
        
        def __init__(
            self,
            name: str,
            description: str,
            parameters: Dict[str, Any],
            required_params: List[str]
        ):
            """
            Initialize a function definition.
            
            Args:
                name: The name of the function
                description: A description of what the function does
                parameters: Parameter definitions with types and descriptions
                required_params: List of required parameter names
            """
            # Implementation details...

FunctionCall
~~~~~~~~~~

Represents a call to a function:

.. code-block:: python

    class FunctionCall:
        """
        Represents a function call from the model.
        """
        
        def __init__(
            self,
            name: str,
            arguments: Dict[str, Any]
        ):
            """
            Initialize a function call.
            
            Args:
                name: The name of the function to call
                arguments: The arguments to pass to the function
            """
            # Implementation details...

Creating Tools
------------

From Functions
~~~~~~~~~~~~

The simplest way to create a tool is from a Python function:

.. code-block:: python

    def get_weather(location: str) -> str:
        """
        Get the current weather for a location.
        
        Args:
            location: The city or location to get weather for
            
        Returns:
            A string describing the weather
        """
        weather_data = {
            "New York": "Sunny, 75°F",
            "London": "Rainy, 60°F",
            "Tokyo": "Cloudy, 65°F",
            "Sydney": "Clear, 80°F",
        }
        return weather_data.get(location, f"Weather data not available for {location}")

    # Create a tool from the function
    weather_tool = Tool.from_callable(get_weather)

From Methods
~~~~~~~~~~

You can also create tools from class methods:

.. code-block:: python

    class Calculator:
        def add(self, a: float, b: float) -> float:
            """Add two numbers together.
            
            Args:
                a: First number
                b: Second number
                
            Returns:
                The sum of a and b
            """
            return a + b
    
        def subtract(self, a: float, b: float) -> float:
            """Subtract b from a.
            
            Args:
                a: First number
                b: Second number
                
            Returns:
                The result of a - b
            """
            return a - b
    
    calculator = Calculator()
    
    # Create tools from instance methods
    add_tool = Tool.from_callable(calculator.add)
    subtract_tool = Tool.from_callable(calculator.subtract)

Using Tools with Agents
---------------------

Passing Tools to Agents
~~~~~~~~~~~~~~~~~~~~~

The most common way to use tools is to pass them directly to an agent:

.. code-block:: python

    from agentle.agents.agent import Agent
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    # Create an agent with tools
    agent = Agent(
        name="Weather Assistant",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You are a helpful assistant that can answer questions about the weather.",
        tools=[get_weather]  # Pass the function as a tool
    )

    # The agent will automatically use the tool when appropriate
    response = agent.run("What's the weather like in Tokyo?")
    print(response.text)

Adding Tools to Existing Agents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can add tools to an existing agent using the ``with_tools()`` method:

.. code-block:: python

    # Create a basic agent
    agent = Agent(
        name="Assistant",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You are a helpful assistant."
    )

    # Add tools to the agent
    agent_with_tools = agent.with_tools([get_weather, calculator.add, calculator.subtract])

    # Use the agent with tools
    response = agent_with_tools.run("What's the weather in London?")

Tool Docstring and Signature
--------------------------

Agentle extracts information about tools from their docstrings and type hints:

Docstring Format
~~~~~~~~~~~~~

For best results, use Google-style docstrings:

.. code-block:: python

    def calculate_mortgage(
        principal: float,
        interest_rate: float,
        years: int
    ) -> dict:
        """
        Calculate monthly mortgage payments.
        
        Args:
            principal: The loan amount in dollars
            interest_rate: Annual interest rate (as a percentage, e.g., 5.5 for 5.5%)
            years: Loan term in years
            
        Returns:
            A dictionary containing monthly payment, total interest, and total cost
        """
        monthly_rate = interest_rate / 100 / 12
        num_payments = years * 12
        
        # Calculate monthly payment
        monthly_payment = principal * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)
        
        # Calculate total interest and total cost
        total_cost = monthly_payment * num_payments
        total_interest = total_cost - principal
        
        return {
            "monthly_payment": round(monthly_payment, 2),
            "total_interest": round(total_interest, 2),
            "total_cost": round(total_cost, 2)
        }

Type Hints
~~~~~~~~

Always use type hints to help Agentle understand the input and output types:

.. code-block:: python

    def search_database(
        query: str,
        limit: int = 10,
        sort_by: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the database with the given query.
        
        Args:
            query: The search query string
            limit: Maximum number of results to return
            sort_by: Field to sort results by
            filters: Optional filters to apply
            
        Returns:
            A list of matching records
        """
        # Implementation details...
        return results

Working with Tool Results
-----------------------

When an agent uses a tool, the following happens:

1. The agent determines which tool to call based on the user query
2. The agent generates the arguments to pass to the tool
3. Agentle executes the tool with the provided arguments
4. The tool result is returned to the agent
5. The agent integrates the tool result into its response

You can access tool execution details through the ``step`` property of the response:

.. code-block:: python

    # Run the agent
    response = agent.run("What's the weather in Tokyo?")
    
    # Get tool execution details
    for step in response.steps:
        if step.type == "tool_execution":
            print(f"Tool: {step.tool_name}")
            print(f"Arguments: {step.arguments}")
            print(f"Result: {step.result}")

Advanced Tool Usage
-----------------

Custom Tool Execution
~~~~~~~~~~~~~~~~~~

For advanced use cases, you can implement custom tool execution logic:

.. code-block:: python

    from agentle.generations.tools.tool_executor import ToolExecutor
    from typing import override, Dict, Any, Optional

    class CustomToolExecutor(ToolExecutor):
        """Custom tool executor with additional capabilities."""
        
        def __init__(self, rate_limit_per_minute: int = 60):
            self.rate_limit_per_minute = rate_limit_per_minute
            self.last_execution_time = {}
            
        @override
        def execute(
            self,
            function_name: str,
            function_to_call: Callable,
            arguments: Dict[str, Any]
        ) -> Any:
            """
            Execute a function with rate limiting.
            
            Args:
                function_name: Name of the function to call
                function_to_call: The actual function to call
                arguments: Arguments to pass to the function
                
            Returns:
                The result of the function call
            """
            # Implement rate limiting
            now = time.time()
            if function_name in self.last_execution_time:
                elapsed = now - self.last_execution_time[function_name].get(0, 0)
                min_interval = 60 / self.rate_limit_per_minute
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
            
            # Execute the function
            result = function_to_call(**arguments)
            
            # Update execution time
            self.last_execution_time[function_name] = now
            
            return result
    
    # Use the custom executor with an agent
    agent = Agent(
        name="Rate Limited Agent",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You use tools with rate limiting.",
        tools=[get_weather, search_database],
        tool_executor=CustomToolExecutor(rate_limit_per_minute=30)
    )

Tool Serialization
~~~~~~~~~~~~~~~

Tools can be serialized for storage or transmission:

.. code-block:: python

    from agentle.generations.tools.tool_serializer import ToolSerializer
    
    # Create a tool serializer
    serializer = ToolSerializer()
    
    # Serialize a tool to JSON
    tool_json = serializer.serialize(weather_tool)
    
    # Deserialize a tool from JSON
    deserialized_tool = serializer.deserialize(tool_json)

Best Practices
------------

1. **Clear Docstrings**: Provide clear, detailed docstrings that explain what the function does
2. **Type Hints**: Always use type hints for parameters and return values
3. **Error Handling**: Ensure your tools handle errors gracefully
4. **Idempotence**: When possible, make your tools idempotent (same input always produces same output)
5. **Security**: Be mindful of security implications, especially for tools that access sensitive resources
6. **Performance**: Keep tool execution time reasonable (preferably under 5 seconds)
7. **Statelessness**: When possible, design tools to be stateless for easier testing and debugging