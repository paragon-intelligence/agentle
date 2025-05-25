"""
BlackSheep API Deployment Examples

This example demonstrates how to deploy Agentle agents, teams, and pipelines
as production-ready REST APIs using the BlackSheep ASGI framework.

Features demonstrated:
- Single agent API deployment
- Agent team API with intelligent orchestration
- Agent pipeline API with sequential processing
- Human-in-the-Loop (HITL) workflows with suspension/resumption
- Automatic OpenAPI documentation generation
- Production-ready error handling

Run this example:
    python examples/blacksheep_api_deployment.py

Then visit:
    http://localhost:8000/docs - Interactive API documentation
    http://localhost:8000/openapi - OpenAPI specification
"""

import asyncio
import os
from typing import Any, List

from agentle.agents.a2a.models.agent_skill import AgentSkill
from agentle.agents.agent import Agent
from agentle.agents.agent_pipeline import AgentPipeline
from agentle.agents.agent_team import AgentTeam
from agentle.agents.asgi.blacksheep.agent_to_blacksheep_application_adapter import (
    AgentToBlackSheepApplicationAdapter,
)
from agentle.agents.errors.tool_suspension_error import ToolSuspensionError
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)
from pydantic import BaseModel


# =============================================================================
# Example 1: Single Agent API
# =============================================================================


def create_code_assistant() -> Agent:
    """Create a code assistant agent with tool calling capabilities."""

    def search_documentation(language: str, topic: str) -> str:
        """Search programming documentation for a specific topic."""
        docs = {
            "python": {
                "async": "Python async/await allows non-blocking code execution...",
                "classes": "Python classes are blueprints for creating objects...",
                "decorators": "Decorators modify or enhance function behavior...",
            },
            "javascript": {
                "promises": "Promises represent eventual completion of async operations...",
                "closures": "Closures give access to outer function scope...",
                "prototypes": "Prototypes enable object inheritance in JavaScript...",
            },
        }

        lang_docs = docs.get(language.lower(), {})
        return lang_docs.get(
            topic.lower(), f"No documentation found for {topic} in {language}"
        )

    def format_code(code: str, language: str = "python") -> str:
        """Format and validate code syntax."""
        # In a real implementation, this would use a code formatter
        formatted = f"```{language}\n{code.strip()}\n```"
        return f"Formatted {language} code:\n{formatted}"

    return Agent(
        name="Code Assistant",
        description="An AI assistant specialized in helping with programming tasks",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash",
        instructions="""You are a helpful programming assistant that can:
        - Answer questions about programming languages and concepts
        - Help debug code and explain programming concepts
        - Search documentation and provide code examples
        - Format and validate code syntax
        
        Always provide clear, practical examples and explanations.""",
        tools=[search_documentation, format_code],
    )


# =============================================================================
# Example 2: Agent Team API with Intelligent Orchestration
# =============================================================================


def create_development_team() -> AgentTeam:
    """Create a team of specialized development agents."""

    provider = GoogleGenerationProvider()

    # Backend Developer Agent
    backend_agent = Agent(
        name="Backend Developer",
        description="Specialized in server-side development, APIs, and databases",
        generation_provider=provider,
        model="gemini-2.0-flash",
        instructions="""You are a backend development expert focused on:
        - Server-side architecture and design patterns
        - RESTful API development and best practices
        - Database design and optimization
        - Security and authentication
        - Performance optimization and scalability""",
        skills=[
            AgentSkill(
                name="api-design", description="Design RESTful APIs and microservices"
            ),
            AgentSkill(
                name="database-design", description="Design and optimize databases"
            ),
            AgentSkill(
                name="security", description="Implement security best practices"
            ),
        ],
    )

    # Frontend Developer Agent
    frontend_agent = Agent(
        name="Frontend Developer",
        description="Specialized in user interfaces and client-side development",
        generation_provider=provider,
        model="gemini-2.0-flash",
        instructions="""You are a frontend development expert focused on:
        - Modern JavaScript frameworks (React, Vue, Angular)
        - Responsive design and CSS best practices
        - User experience and accessibility
        - Performance optimization
        - State management and component architecture""",
        skills=[
            AgentSkill(
                name="ui-design", description="Create responsive user interfaces"
            ),
            AgentSkill(
                name="javascript", description="Modern JavaScript and frameworks"
            ),
            AgentSkill(
                name="accessibility", description="Web accessibility best practices"
            ),
        ],
    )

    # DevOps Engineer Agent
    devops_agent = Agent(
        name="DevOps Engineer",
        description="Specialized in deployment, infrastructure, and automation",
        generation_provider=provider,
        model="gemini-2.0-flash",
        instructions="""You are a DevOps expert focused on:
        - CI/CD pipeline design and implementation
        - Container orchestration with Docker and Kubernetes
        - Cloud infrastructure and automation
        - Monitoring and logging solutions
        - Infrastructure as Code (IaC)""",
        skills=[
            AgentSkill(name="deployment", description="Deploy and manage applications"),
            AgentSkill(
                name="automation", description="Automate infrastructure and processes"
            ),
            AgentSkill(name="monitoring", description="Set up monitoring and alerting"),
        ],
    )

    return AgentTeam(
        agents=[backend_agent, frontend_agent, devops_agent],
        orchestrator_provider=provider,
        orchestrator_model="gemini-2.0-flash",
    )


# =============================================================================
# Example 3: Agent Pipeline API with Sequential Processing
# =============================================================================


def create_content_pipeline() -> AgentPipeline:
    """Create a content creation pipeline with sequential processing."""

    provider = GoogleGenerationProvider()

    # Research Agent (Stage 1)
    research_agent = Agent(
        name="Research Specialist",
        generation_provider=provider,
        model="gemini-2.0-flash",
        instructions="""You are a research specialist focused on gathering accurate information.
        Your role is to:
        - Research topics thoroughly using available knowledge
        - Fact-check information and cite sources when possible
        - Identify key points and relevant details
        - Organize findings in a structured format
        - Prioritize accuracy and credibility over speed""",
    )

    # Content Writer Agent (Stage 2)
    content_writer = Agent(
        name="Content Writer",
        generation_provider=provider,
        model="gemini-2.0-flash",
        instructions="""You are a skilled content writer who creates engaging content.
        Your role is to:
        - Transform research into compelling, readable content
        - Maintain a consistent tone and style
        - Structure content with clear headings and flow
        - Ensure content is engaging and informative
        - Adapt writing style to the target audience""",
    )

    # Editor Agent (Stage 3)
    editor_agent = Agent(
        name="Content Editor",
        generation_provider=provider,
        model="gemini-2.0-flash",
        instructions="""You are a meticulous editor focused on quality and clarity.
        Your role is to:
        - Review content for grammar, spelling, and style
        - Improve clarity and readability
        - Ensure consistent formatting and structure
        - Fact-check claims and verify accuracy
        - Polish the final content for publication""",
    )

    return AgentPipeline(
        agents=[research_agent, content_writer, editor_agent],
        debug_mode=True,  # Enable detailed logging for demonstration
    )


# =============================================================================
# Example 4: HITL-Enabled Financial Agent
# =============================================================================


def create_financial_agent() -> Agent:
    """Create a financial agent with Human-in-the-Loop approval workflows."""

    def transfer_funds(from_account: str, to_account: str, amount: float) -> str:
        """Transfer funds between accounts - requires approval for large amounts."""
        if amount > 10000:
            raise ToolSuspensionError(
                reason=f"Transfer of ${amount:,.2f} requires human approval",
                approval_data={
                    "from_account": from_account,
                    "to_account": to_account,
                    "amount": amount,
                    "risk_level": "high" if amount > 50000 else "medium",
                },
                timeout_seconds=3600,  # 1 hour timeout
            )
        return (
            f"✅ Transfer completed: ${amount:,.2f} from {from_account} to {to_account}"
        )

    def send_wire_transfer(recipient: str, amount: float, swift_code: str) -> str:
        """Send international wire transfer - always requires approval."""
        raise ToolSuspensionError(
            reason=f"International wire transfer of ${amount:,.2f} requires compliance approval",
            approval_data={
                "recipient": recipient,
                "amount": amount,
                "swift_code": swift_code,
                "transfer_type": "international_wire",
                "risk_level": "high",
            },
            timeout_seconds=7200,  # 2 hours timeout
        )

    def check_account_balance(account_number: str) -> str:
        """Check account balance - no approval required."""
        # Simulate account balances
        balances = {
            "ACC-001": 25000.50,
            "ACC-002": 150000.75,
            "ACC-003": 5000.00,
        }
        balance = balances.get(account_number, 0.0)
        return f"Account {account_number} balance: ${balance:,.2f}"

    return Agent(
        name="Financial Assistant",
        description="AI assistant for financial operations with human oversight",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash",
        instructions="""You are a financial assistant that helps with banking operations.
        
        You can:
        - Check account balances (no approval needed)
        - Transfer funds domestically (approval needed for amounts > $10,000)
        - Send international wire transfers (always requires approval)
        
        Always confirm transaction details before executing operations.
        For suspended operations, explain that human approval is required and provide the resumption process.""",
        tools=[transfer_funds, send_wire_transfer, check_account_balance],
    )


# =============================================================================
# Example 5: Structured Output Agent
# =============================================================================


class WeatherForecast(BaseModel):
    """Structured weather forecast response."""

    location: str
    current_temperature: float
    conditions: str
    humidity: int
    wind_speed: float
    forecast: List[str]
    alerts: List[str] = []


def create_weather_agent() -> Agent[WeatherForecast]:
    """Create a weather agent that returns structured forecasts."""

    def get_weather_data(location: str) -> dict[str, Any]:
        """Get weather data for a location."""
        # Simulate weather API response
        weather_db = {
            "new york": {
                "temp": 72.5,
                "conditions": "Partly cloudy",
                "humidity": 65,
                "wind": 8.2,
                "forecast": [
                    "Tomorrow: Sunny, 75°F",
                    "Day 2: Rainy, 68°F",
                    "Day 3: Cloudy, 70°F",
                ],
                "alerts": ["Air quality moderate"],
            },
            "london": {
                "temp": 15.0,
                "conditions": "Rainy",
                "humidity": 85,
                "wind": 12.1,
                "forecast": [
                    "Tomorrow: Cloudy, 16°C",
                    "Day 2: Sunny, 18°C",
                    "Day 3: Rainy, 14°C",
                ],
                "alerts": [],
            },
            "tokyo": {
                "temp": 28.0,
                "conditions": "Sunny",
                "humidity": 70,
                "wind": 5.5,
                "forecast": [
                    "Tomorrow: Partly cloudy, 26°C",
                    "Day 2: Rainy, 22°C",
                    "Day 3: Sunny, 29°C",
                ],
                "alerts": ["High UV index warning"],
            },
        }

        return weather_db.get(
            location.lower(),
            {
                "temp": 20.0,
                "conditions": "Unknown",
                "humidity": 50,
                "wind": 0.0,
                "forecast": ["No forecast available"],
                "alerts": ["Location not found"],
            },
        )

    return Agent(
        name="Weather Forecaster",
        description="AI weather assistant that provides structured forecasts",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash",
        instructions="""You are a weather forecasting assistant that provides accurate, structured weather information.
        
        Use the weather data tool to get current conditions and forecasts.
        Always provide complete information including temperature, conditions, humidity, wind speed, and multi-day forecasts.
        Include any weather alerts or warnings in your response.""",
        tools=[get_weather_data],
        response_schema=WeatherForecast,
    )


# =============================================================================
# Deployment Functions
# =============================================================================


def deploy_single_agent():
    """Deploy a single agent as a REST API."""
    print("🤖 Deploying Code Assistant API...")

    agent = create_code_assistant()
    app = AgentToBlackSheepApplicationAdapter().adapt(agent)

    print("📚 API Documentation available at:")
    print("   http://localhost:8000/docs")
    print("🔗 Endpoints:")
    print("   POST /api/v1/agents/code_assistant/run")
    print("   POST /api/v1/agents/code_assistant/run/resume")

    return app


def deploy_agent_team():
    """Deploy an agent team as a REST API."""
    print("👥 Deploying Development Team API...")

    team = create_development_team()
    app = AgentToBlackSheepApplicationAdapter().adapt(team)

    print("📚 API Documentation available at:")
    print("   http://localhost:8000/docs")
    print("🔗 Endpoints:")
    print("   POST /api/v1/team/run")
    print("   POST /api/v1/team/resume")

    return app


def deploy_agent_pipeline():
    """Deploy an agent pipeline as a REST API."""
    print("🔗 Deploying Content Creation Pipeline API...")

    pipeline = create_content_pipeline()
    app = AgentToBlackSheepApplicationAdapter().adapt(pipeline)

    print("📚 API Documentation available at:")
    print("   http://localhost:8000/docs")
    print("🔗 Endpoints:")
    print("   POST /api/v1/pipeline/run")
    print("   POST /api/v1/pipeline/resume")

    return app


def deploy_hitl_agent():
    """Deploy a HITL-enabled financial agent."""
    print("🏦 Deploying Financial Assistant API with HITL...")

    agent = create_financial_agent()
    app = AgentToBlackSheepApplicationAdapter().adapt(agent)

    print("📚 API Documentation available at:")
    print("   http://localhost:8000/docs")
    print("🔗 Endpoints:")
    print("   POST /api/v1/agents/financial_assistant/run")
    print("   POST /api/v1/agents/financial_assistant/run/resume")
    print("💡 Try operations that require approval (transfers > $10,000)")

    return app


def deploy_structured_output_agent():
    """Deploy an agent with structured outputs."""
    print("🌤️ Deploying Weather Forecaster API with Structured Outputs...")

    agent = create_weather_agent()
    app = AgentToBlackSheepApplicationAdapter().adapt(agent)

    print("📚 API Documentation available at:")
    print("   http://localhost:8000/docs")
    print("🔗 Endpoints:")
    print("   POST /api/v1/agents/weather_forecaster/run")
    print("📊 Returns structured WeatherForecast objects")

    return app


# =============================================================================
# Example Usage and Testing
# =============================================================================


async def test_apis():
    """Test the deployed APIs programmatically."""
    print("\n🧪 Testing API Examples...")

    # Test single agent
    print("\n1. Testing Code Assistant...")
    code_agent = create_code_assistant()
    response = await code_agent.run_async(
        "Explain Python decorators and show an example"
    )
    print(f"Response: {response.text[:100]}...")

    # Test agent team
    print("\n2. Testing Development Team...")
    dev_team = create_development_team()
    response = await dev_team.run_async(
        "How do I implement user authentication in a web application?"
    )
    print(f"Team Response: {response.text[:100]}...")

    # Test pipeline
    print("\n3. Testing Content Pipeline...")
    content_pipeline = create_content_pipeline()
    response = await content_pipeline.run_async(
        "Create an article about the benefits of renewable energy"
    )
    print(f"Pipeline Response: {response.text[:100]}...")

    # Test structured output
    print("\n4. Testing Weather Agent...")
    weather_agent = create_weather_agent()
    response = await weather_agent.run_async("What's the weather like in Tokyo?")
    if response.parsed:
        forecast = response.parsed
        print(
            f"Structured Response: {forecast.location}, {forecast.current_temperature}°C, {forecast.conditions}"
        )

    # Test HITL agent (will suspend)
    print("\n5. Testing Financial Agent (HITL)...")
    financial_agent = create_financial_agent()
    try:
        response = await financial_agent.run_async(
            "Transfer $25,000 from ACC-001 to ACC-002"
        )
        if response.is_suspended:
            print(f"✋ Execution suspended: {response.suspension_reason}")
            print(f"🎫 Resumption token: {response.resumption_token}")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Expected suspension: {e}")


def main():
    """Main function to demonstrate different deployment options."""
    print("🚀 BlackSheep API Deployment Examples")
    print("=" * 50)

    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  Warning: GOOGLE_API_KEY not set. Set it to run the examples:")
        print("   export GOOGLE_API_KEY='your-api-key-here'")
        print()

    print("Choose a deployment example:")
    print("1. Single Agent API (Code Assistant)")
    print("2. Agent Team API (Development Team)")
    print("3. Agent Pipeline API (Content Creation)")
    print("4. HITL Agent API (Financial Assistant)")
    print("5. Structured Output API (Weather Forecaster)")
    print("6. Test all APIs programmatically")

    choice = input("\nEnter your choice (1-6): ").strip()

    if choice == "1":
        app = deploy_single_agent()
    elif choice == "2":
        app = deploy_agent_team()
    elif choice == "3":
        app = deploy_agent_pipeline()
    elif choice == "4":
        app = deploy_hitl_agent()
    elif choice == "5":
        app = deploy_structured_output_agent()
    elif choice == "6":
        asyncio.run(test_apis())
        return
    else:
        print("Invalid choice. Defaulting to single agent API.")
        app = deploy_single_agent()

    # Run the server
    print("\n🌐 Starting server on http://localhost:8000")
    print("Press Ctrl+C to stop the server")

    try:
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=8000)
    except ImportError:
        print("❌ uvicorn not installed. Install it with: pip install uvicorn")
    except KeyboardInterrupt:
        print("\n👋 Server stopped")


if __name__ == "__main__":
    main()
