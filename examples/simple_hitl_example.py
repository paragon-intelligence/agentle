"""
Simple Human-in-the-Loop (HITL) Example

This example demonstrates the enhanced Tool callbacks in Agentle that enable
Human-in-the-Loop workflows. It shows how to use before_call and after_call
callbacks for approval requests and audit logging.
"""

import asyncio
from datetime import datetime

from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)
from agentle.generations.tools.tool import Tool


# Simple approval system for demonstration
approval_requests = []
audit_log = []


async def request_approval(context=None, **kwargs):
    """Request human approval before executing a sensitive tool."""
    print(f"🔔 APPROVAL REQUIRED")
    print(f"   Tool arguments: {kwargs}")
    print(f"   Context ID: {context.context_id if context else 'None'}")

    # Log the approval request
    approval_requests.append(
        {
            "timestamp": datetime.now(),
            "arguments": kwargs,
            "context_id": context.context_id if context else None,
        }
    )

    # Simulate approval process (in production, this would be async)
    print("   Simulating human approval...")
    await asyncio.sleep(1)  # Simulate human response time
    print("   ✅ APPROVED by human operator")

    return True


def log_execution(context=None, result=None, **kwargs):
    """Log tool execution for audit trail."""
    print(f"📝 AUDIT LOG")
    print(f"   Tool executed with args: {kwargs}")
    print(f"   Result: {str(result)[:100]}...")
    print(f"   Timestamp: {datetime.now()}")
    print(f"   Context ID: {context.context_id if context else 'None'}")

    # Add to audit log
    audit_log.append(
        {
            "timestamp": datetime.now(),
            "arguments": kwargs,
            "result": str(result),
            "context_id": context.context_id if context else None,
        }
    )


# Define tools that require approval
def transfer_money(amount: float, to_account: str) -> str:
    """Transfer money - requires human approval."""
    return f"💰 Transferred ${amount:,.2f} to account {to_account}"


def send_email(to: str, subject: str) -> str:
    """Send email - requires human approval."""
    return f"📧 Email sent to {to} with subject '{subject}'"


def check_balance(account: str) -> str:
    """Check account balance - no approval required."""
    return f"💳 Account {account} balance: $10,000.00"


async def main():
    """Demonstrate HITL functionality."""
    print("🤝 Human-in-the-Loop Demo")
    print("=" * 40)

    # Create tools with different approval requirements
    transfer_tool = Tool.from_callable(
        transfer_money,
        before_call=request_approval,  # Requires approval
        after_call=log_execution,  # Always log
    )

    email_tool = Tool.from_callable(
        send_email,
        before_call=request_approval,  # Requires approval
        after_call=log_execution,  # Always log
    )

    balance_tool = Tool.from_callable(
        check_balance,
        # No before_call = no approval required
        after_call=log_execution,  # But still log for audit
    )

    # Create agent with HITL-enabled tools
    agent = Agent(
        name="Financial Assistant",
        generation_provider=GoogleGenaiGenerationProvider(),
        model="gemini-2.0-flash",
        instructions="""You are a financial assistant. Some operations require human approval:
        - Money transfers: REQUIRE approval
        - Email sending: REQUIRE approval  
        - Balance checks: NO approval needed
        
        Always explain what approvals are needed.""",
        tools=[transfer_tool, email_tool, balance_tool],
    )

    # Test 1: Balance check (no approval)
    print("\n📊 Test 1: Balance Check (No Approval)")
    print("-" * 30)
    response1 = await agent.run_async("Check the balance for account A123")
    print(f"Agent response: {response1.text}")

    # Test 2: Money transfer (requires approval)
    print("\n💸 Test 2: Money Transfer (Requires Approval)")
    print("-" * 30)
    response2 = await agent.run_async("Transfer $5000 to account B456")
    print(f"Agent response: {response2.text}")

    # Test 3: Email (requires approval)
    print("\n📧 Test 3: Send Email (Requires Approval)")
    print("-" * 30)
    response3 = await agent.run_async(
        "Send an email to admin@company.com with subject 'Transfer Complete'"
    )
    print(f"Agent response: {response3.text}")

    # Show summary
    print("\n📋 Summary")
    print("-" * 30)
    print(f"Approval requests: {len(approval_requests)}")
    print(f"Audit log entries: {len(audit_log)}")

    print("\n🎉 HITL Demo completed!")
    print("\nKey features demonstrated:")
    print("✅ before_call callbacks for approval workflows")
    print("✅ after_call callbacks for audit logging")
    print("✅ Context passing to callbacks")
    print("✅ Selective approval requirements")


if __name__ == "__main__":
    print("Starting Simple HITL Demo...")
    asyncio.run(main())
