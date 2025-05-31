# WhatsApp Integration Guide

This guide explains how to use Agentle's WhatsApp integration to build production-ready WhatsApp bots with Evolution API.

## Overview

Agentle's WhatsApp integration provides:

- **🚀 Easy Setup**: Simple configuration with Evolution API
- **📦 Session Management**: Production-ready session storage with Redis support
- **🔧 Flexible Architecture**: Pluggable providers for different WhatsApp APIs
- **📊 Built-in Observability**: Logging, error handling, and monitoring
- **🌐 Instant APIs**: Convert bots to REST APIs with automatic documentation
- **🛡️ Production Ready**: Error handling, retries, and graceful degradation

## Quick Start

### 1. Install Dependencies

```bash
# Basic installation
pip install agentle

# For Redis session storage (production)
pip install agentle[redis]

# For complete WhatsApp support
pip install agentle[whatsapp]
```

### 2. Set Up Evolution API

1. Install and run Evolution API server
2. Create an instance in Evolution API
3. Get your API key and instance name

### 3. Basic Bot Example

```python
import asyncio
from agentle.agents.agent import Agent
from agentle.agents.whatsapp.whatsapp_bot import WhatsAppBot
from agentle.agents.whatsapp.providers.evolution.evolution_api_provider import EvolutionAPIProvider
from agentle.agents.whatsapp.providers.evolution.evolution_api_config import EvolutionAPIConfig
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

# Create agent
agent = Agent(
    name="My WhatsApp Bot",
    generation_provider=GoogleGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You are a helpful assistant."
)

# Configure Evolution API
config = EvolutionAPIConfig(
    base_url="http://localhost:8080",
    instance_name="my-bot",
    api_key="your-api-key"
)

# Create provider and bot
provider = EvolutionAPIProvider(config)
bot = WhatsAppBot(agent, provider)

# Run as async service
async def main():
    await bot.start()
    # Keep running...
    
asyncio.run(main())
```

## Session Management

### In-Memory Sessions (Development)

```python
from agentle.session.in_memory_session_store import InMemorySessionStore
from agentle.session.session_manager import SessionManager

# Create session store
session_store = InMemorySessionStore[WhatsAppSession](
    cleanup_interval_seconds=300  # Cleanup every 5 minutes
)

# Create session manager
session_manager = SessionManager[WhatsAppSession](
    session_store=session_store,
    default_ttl_seconds=1800,  # 30 minutes
    enable_events=True
)

# Use with provider
provider = EvolutionAPIProvider(
    config=evolution_config,
    session_manager=session_manager
)
```

### Redis Sessions (Production)

```python
from agentle.session.redis_session_store import RedisSessionStore

# Create Redis session store
session_store = RedisSessionStore[WhatsAppSession](
    redis_url="redis://localhost:6379/0",
    key_prefix="whatsapp:sessions:",
    default_ttl_seconds=3600,  # 1 hour
    session_class=WhatsAppSession
)

# Create session manager
session_manager = SessionManager[WhatsAppSession](
    session_store=session_store,
    default_ttl_seconds=3600,
    enable_events=True
)

# Use with provider
provider = EvolutionAPIProvider(
    config=evolution_config,
    session_manager=session_manager
)
```

### Session Events

```python
# Add event handlers
async def on_session_created(session_id: str, session_data: Any) -> None:
    print(f"New session: {session_id}")

async def on_session_deleted(session_id: str, session_data: Any) -> None:
    print(f"Session deleted: {session_id}")

session_manager.add_event_handler("session_created", on_session_created)
session_manager.add_event_handler("session_deleted", on_session_deleted)
```

## Bot Configuration

```python
from agentle.agents.whatsapp.models.whatsapp_bot_config import WhatsAppBotConfig

bot_config = WhatsAppBotConfig(
    typing_indicator=True,
    typing_duration=3,
    auto_read_messages=True,
    session_timeout_minutes=30,
    max_message_length=4000,
    welcome_message="Hello! How can I help you?",
    error_message="Sorry, something went wrong. Please try again.",
)

bot = WhatsAppBot(agent, provider, config=bot_config)
```

## Adding Tools

```python
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    # Your weather API implementation
    return f"Weather in {location}: Sunny, 25°C"

def book_appointment(date: str, time: str) -> str:
    """Book an appointment."""
    # Your booking system implementation
    return f"Appointment booked for {date} at {time}"

# Create agent with tools
agent = Agent(
    name="Assistant Bot",
    generation_provider=GoogleGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="""You are a helpful assistant that can:
    - Provide weather information
    - Book appointments
    Always be friendly and helpful.""",
    tools=[get_weather, book_appointment]
)
```

## REST API Deployment

### Simple API Server

```python
# Convert bot to BlackSheep application
app = bot.to_blacksheep_app(
    webhook_path="/webhook/whatsapp",
    show_error_details=False  # Set to True for development
)

# Run with uvicorn
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Production API with Custom Configuration

```python
from blacksheep import Application
from blacksheep.server.openapi.v3 import OpenAPIHandler
from openapidocs.v3 import Info

# Create custom OpenAPI documentation
docs = OpenAPIHandler(
    ui_path="/api/docs",
    info=Info(
        title="My WhatsApp Bot API",
        version="1.0.0",
        description="Production WhatsApp bot with AI capabilities"
    )
)

# Convert bot to application
app = bot.to_blacksheep_app(
    webhook_path="/webhooks/whatsapp",
    docs=docs,
    show_error_details=False
)

# Add custom middleware, CORS, etc.
# app.use_cors()
# app.use_authentication()

# Deploy
uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Environment Configuration

### Environment Variables

```bash
# Evolution API Configuration
EVOLUTION_API_URL=http://localhost:8080
EVOLUTION_INSTANCE_NAME=my-bot
EVOLUTION_API_KEY=your-api-key

# Session Storage
REDIS_URL=redis://localhost:6379/0

# Bot Configuration
BOT_MODE=production  # simple, development, production
WEBHOOK_URL=https://your-domain.com/webhook/whatsapp
PORT=8000
DEBUG=false

# AI Provider
GOOGLE_API_KEY=your-google-api-key
```

### Configuration Class

```python
import os
from dataclasses import dataclass

@dataclass
class WhatsAppBotSettings:
    # Evolution API
    evolution_url: str = os.getenv("EVOLUTION_API_URL", "http://localhost:8080")
    evolution_instance: str = os.getenv("EVOLUTION_INSTANCE_NAME", "my-bot")
    evolution_api_key: str = os.getenv("EVOLUTION_API_KEY", "")
    
    # Session storage
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    session_ttl: int = int(os.getenv("SESSION_TTL", "3600"))
    
    # Bot behavior
    webhook_url: str = os.getenv("WEBHOOK_URL", "")
    typing_duration: int = int(os.getenv("TYPING_DURATION", "3"))
    auto_read: bool = os.getenv("AUTO_READ_MESSAGES", "true").lower() == "true"
    
    # Server
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

# Use configuration
settings = WhatsAppBotSettings()

evolution_config = EvolutionAPIConfig(
    base_url=settings.evolution_url,
    instance_name=settings.evolution_instance,
    api_key=settings.evolution_api_key
)
```

## Error Handling and Monitoring

### Custom Error Handling

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add custom webhook handlers
async def on_webhook_error(payload, error):
    logging.error(f"Webhook error: {error}")
    # Send to monitoring system
    
bot.add_webhook_handler(on_webhook_error)
```

### Health Checks

```python
# Add health check endpoint
@app.route("/health")
async def health_check():
    stats = provider.get_stats()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "stats": stats
    }
```

### Monitoring Stats

```python
# Get provider statistics
provider_stats = provider.get_stats()
print(f"Instance: {provider_stats['instance_name']}")
print(f"Active sessions: {provider_stats['session_stats']['store_stats']['total_sessions']}")

# Get session manager statistics
session_stats = session_manager.get_stats()
print(f"Session events enabled: {session_stats['events_enabled']}")
print(f"Default TTL: {session_stats['default_ttl_seconds']}s")
```

## Advanced Features

### Message Filtering

```python
class CustomWhatsAppBot(WhatsAppBot):
    async def handle_message(self, message: WhatsAppMessage) -> None:
        # Filter spam or unwanted messages
        if self.is_spam(message):
            return
        
        # Add custom preprocessing
        processed_message = self.preprocess_message(message)
        
        # Call parent handler
        await super().handle_message(processed_message)
    
    def is_spam(self, message: WhatsAppMessage) -> bool:
        # Your spam detection logic
        return False
    
    def preprocess_message(self, message: WhatsAppMessage) -> WhatsAppMessage:
        # Your preprocessing logic
        return message
```

### Multi-Instance Support

```python
# Create multiple bot instances
bots = {}
instances = ["bot1", "bot2", "bot3"]

for instance_name in instances:
    config = EvolutionAPIConfig(
        base_url="http://localhost:8080",
        instance_name=instance_name,
        api_key=f"key-{instance_name}"
    )
    
    provider = EvolutionAPIProvider(config)
    bot = WhatsAppBot(agent, provider)
    bots[instance_name] = bot

# Start all bots
for bot in bots.values():
    await bot.start()
```

### Custom Message Types

```python
from agentle.agents.whatsapp.models.whatsapp_location_message import WhatsAppLocationMessage

async def handle_location_message(self, message: WhatsAppLocationMessage):
    """Handle location messages specially."""
    location_info = f"Received location: {message.latitude}, {message.longitude}"
    if message.name:
        location_info += f" ({message.name})"
    
    # Process location with agent
    response = await self.agent.run_async(f"User shared location: {location_info}")
    await self.provider.send_text_message(message.from_number, response.text)
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "whatsapp_bot_example.py", "production"]
```

### Docker Compose with Redis

```yaml
version: '3.8'

services:
  whatsapp-bot:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - EVOLUTION_API_URL=http://evolution:8080
      - BOT_MODE=production
    depends_on:
      - redis
      - evolution
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    
  evolution:
    image: atendai/evolution-api:latest
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/evolution
    depends_on:
      - postgres
    
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=evolution
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  redis_data:
  postgres_data:
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: whatsapp-bot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: whatsapp-bot
  template:
    metadata:
      labels:
        app: whatsapp-bot
    spec:
      containers:
      - name: whatsapp-bot
        image: your-registry/whatsapp-bot:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        - name: EVOLUTION_API_URL
          value: "http://evolution-service:8080"
        - name: BOT_MODE
          value: "production"
        resources:
          limits:
            memory: "512Mi"
            cpu: "500m"
          requests:
            memory: "256Mi"
            cpu: "250m"
---
apiVersion: v1
kind: Service
metadata:
  name: whatsapp-bot-service
spec:
  selector:
    app: whatsapp-bot
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Best Practices

### 1. Security

- Use environment variables for sensitive configuration
- Implement webhook verification
- Rate limit API endpoints
- Use HTTPS in production
- Rotate API keys regularly

### 2. Performance

- Use Redis for session storage in production
- Implement proper caching strategies
- Monitor memory usage with session cleanup
- Use connection pooling for databases
- Implement graceful shutdowns

### 3. Reliability

- Implement retry logic for failed operations
- Use circuit breakers for external services
- Monitor webhook delivery
- Set up proper logging and alerting
- Implement health checks

### 4. Scalability

- Use horizontal scaling with load balancers
- Implement sticky sessions if needed
- Use distributed session storage (Redis)
- Monitor and tune session TTLs
- Implement proper resource limits

## Troubleshooting

### Common Issues

1. **Connection Errors**
   ```bash
   # Check Evolution API status
   curl http://localhost:8080/instance/fetchInstances
   
   # Check Redis connection
   redis-cli ping
   ```

2. **Session Issues**
   ```python
   # Check session statistics
   stats = session_manager.get_stats()
   print(f"Active sessions: {stats}")
   
   # List active sessions
   sessions = await session_manager.list_sessions()
   print(f"Session IDs: {sessions}")
   ```

3. **Webhook Problems**
   ```python
   # Verify webhook URL
   webhook_url = provider.get_webhook_url()
   print(f"Current webhook: {webhook_url}")
   
   # Re-register webhook
   await provider.set_webhook_url("https://your-domain.com/webhook")
   ```

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable agent debugging
agent = Agent(
    # ... other params ...
    debug=True
)

# Create development server with error details
app = bot.to_blacksheep_app(show_error_details=True)
```

### Monitoring

```python
# Add monitoring endpoints
@app.route("/metrics")
async def metrics():
    return {
        "provider_stats": provider.get_stats(),
        "session_stats": session_manager.get_stats(),
        "timestamp": datetime.now().isoformat()
    }

@app.route("/sessions")
async def list_sessions():
    sessions = await session_manager.list_sessions(include_metadata=True)
    return {"sessions": sessions}
```

## Migration Guide

### From Basic WhatsApp Integration

If you're migrating from a basic WhatsApp integration:

1. **Update imports**:
   ```python
   # Old
   from agentle.whatsapp import WhatsAppBot
   
   # New
   from agentle.agents.whatsapp.whatsapp_bot import WhatsAppBot
   from agentle.agents.whatsapp.providers.evolution.evolution_api_provider import EvolutionAPIProvider
   ```

2. **Add session management**:
   ```python
   # Add session manager
   session_manager = SessionManager[WhatsAppSession](
       session_store=InMemorySessionStore[WhatsAppSession]()
   )
   
   provider = EvolutionAPIProvider(config, session_manager=session_manager)
   ```

3. **Update configuration**:
   ```python
   # Use structured configuration
   bot_config = WhatsAppBotConfig(
       typing_indicator=True,
       auto_read_messages=True,
       # ... other settings
   )
   ```

This comprehensive integration provides a solid foundation for building production-ready WhatsApp bots with Agentle! 