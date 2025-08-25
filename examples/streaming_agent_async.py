import asyncio
from agentle.agents.agent import Agent

from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()


class Response(BaseModel):
    reasoning: str
    response: str


agent = Agent()

print("Streaming...")
print("=" * 50)


async def main():
    async for chunk in await agent.run_async("quem foi George Floyd?", stream=True):
        print(chunk.text)


if __name__ == "__main__":
    asyncio.run(main())
