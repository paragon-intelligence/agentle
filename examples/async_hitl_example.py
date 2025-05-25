"""
Asynchronous Human-in-the-Loop (HITL) Example

This example demonstrates true asynchronous HITL workflows where:
1. Agent execution can be suspended for external approval
2. The process doesn't block waiting for approval
3. Execution can be resumed hours or days later
4. Multiple suspended executions can be managed concurrently

This solves the real-world problem where waiting for human approval
would otherwise block the entire application process.
"""

import asyncio
from datetime import datetime

from agentle.agents.agent import Agent
from agentle.agents.errors.tool_suspension_error import ToolSuspensionError
from agentle.agents.suspension_manager import (
    SuspensionManager,
    InMemorySuspensionStore,
    set_default_suspension_manager,
)
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)
from agentle.generations.tools.tool import Tool


# Global suspension manager for this example
suspension_manager = SuspensionManager(InMemorySuspensionStore())
set_default_suspension_manager(suspension_manager)

# Global storage for suspended tokens (in production, use proper storage)
suspended_tokens = []


def wire_transfer(amount: float, to_account: str, memo: str = "") -> str:
    """
    Execute a wire transfer - requires human approval for large amounts.

    This tool demonstrates how to suspend execution for human approval
    without blocking the entire process.
    """
    # Check if approval is needed
    if amount > 1000:
        # Suspend execution and wait for approval
        raise ToolSuspensionError(
            reason=f"Wire transfer of ${amount:,.2f} requires human approval",
            approval_data={
                "operation": "wire_transfer",
                "amount": amount,
                "to_account": to_account,
                "memo": memo,
                "risk_level": "high" if amount > 10000 else "medium",
                "requested_at": datetime.now().isoformat(),
            },
            timeout_seconds=86400,  # 24 hours
        )

    # Execute the transfer if no approval needed
    return f"✅ Wire transfer completed: ${amount:,.2f} to {to_account}. Memo: {memo}"


def send_marketing_email(to_list: str, subject: str, campaign_id: str) -> str:
    """Send marketing email - requires approval for large campaigns."""
    # Simulate checking campaign size
    recipient_count = len(to_list.split(","))

    if recipient_count > 100:
        raise ToolSuspensionError(
            reason=f"Marketing campaign to {recipient_count} recipients requires approval",
            approval_data={
                "operation": "marketing_email",
                "recipient_count": recipient_count,
                "subject": subject,
                "campaign_id": campaign_id,
                "risk_level": "high" if recipient_count > 1000 else "medium",
            },
            timeout_seconds=172800,  # 48 hours
        )

    return f"📧 Marketing email sent to {recipient_count} recipients: {subject}"


def check_account_balance(account: str) -> str:
    """Check account balance - no approval required."""
    return f"💳 Account {account} balance: $25,000.00"


async def create_financial_agent() -> Agent:
    """Create a financial agent with suspension-capable tools."""

    # Create tools that can suspend execution
    transfer_tool = Tool.from_callable(wire_transfer)
    email_tool = Tool.from_callable(send_marketing_email)
    balance_tool = Tool.from_callable(check_account_balance)

    return Agent(
        name="Async Financial Assistant",
        generation_provider=GoogleGenaiGenerationProvider(),
        model="gemini-2.0-flash",
        instructions="""You are a financial assistant that can:
        1. Execute wire transfers (requires approval for amounts > $1000)
        2. Send marketing emails (requires approval for > 100 recipients)
        3. Check account balances (no approval needed)
        
        When operations require approval, the system will pause and notify the appropriate personnel.""",
        tools=[transfer_tool, email_tool, balance_tool],
    )


async def simulate_user_requests():
    """Simulate multiple user requests that may require approval."""
    agent = await create_financial_agent()

    print("🏦 Async HITL Demo - Simulating User Requests")
    print("=" * 60)

    # Request 1: Small transfer (no approval needed)
    print("\n📊 Request 1: Small transfer (no approval needed)")
    print("-" * 40)
    try:
        result1 = await agent.run_async(
            "Transfer $500 to account ACC-001 with memo 'Office supplies'"
        )
        if result1.is_suspended:
            print(f"❌ Unexpected suspension: {result1.suspension_reason}")
        else:
            print(f"✅ Completed: {result1.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Request 2: Large transfer (requires approval)
    print("\n💸 Request 2: Large transfer (requires approval)")
    print("-" * 40)
    try:
        result2 = await agent.run_async(
            "Transfer $15000 to account ACC-002 with memo 'Equipment purchase'"
        )
        if result2.is_suspended:
            print(f"⏸️  SUSPENDED: {result2.suspension_reason}")
            print(f"📋 Resumption token: {result2.resumption_token}")
            print("💡 Process continues without blocking!")

            # Store the token for later resumption
            suspended_tokens.append(result2.resumption_token)
        else:
            print(f"✅ Completed: {result2.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Request 3: Large email campaign (requires approval)
    print("\n📧 Request 3: Large email campaign (requires approval)")
    print("-" * 40)
    try:
        # Simulate a large recipient list
        large_recipient_list = ",".join([f"user{i}@company.com" for i in range(500)])
        result3 = await agent.run_async(
            f"Send marketing email to {large_recipient_list} with subject 'New Product Launch' for campaign CAMP-2024-001"
        )
        if result3.is_suspended:
            print(f"⏸️  SUSPENDED: {result3.suspension_reason}")
            print(f"📋 Resumption token: {result3.resumption_token}")
            print("💡 Process continues without blocking!")

            # Store the token for later resumption
            suspended_tokens.append(result3.resumption_token)
        else:
            print(f"✅ Completed: {result3.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Request 4: Balance check (no approval needed)
    print("\n💳 Request 4: Balance check (no approval needed)")
    print("-" * 40)
    try:
        result4 = await agent.run_async("Check the balance for account ACC-001")
        if result4.is_suspended:
            print(f"❌ Unexpected suspension: {result4.suspension_reason}")
        else:
            print(f"✅ Completed: {result4.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n📊 Summary: Processed 4 requests")
    print("   - 2 completed immediately")
    print("   - 2 suspended for approval")
    print("   - Process never blocked!")


async def simulate_approval_interface():
    """Simulate the human approval interface (web UI, mobile app, etc.)."""
    print("\n👤 Human Approval Interface")
    print("=" * 60)

    # Get pending approvals
    pending = await suspension_manager.get_pending_approvals()
    print(f"📋 Found {len(pending)} pending approvals")

    for i, approval in enumerate(pending, 1):
        print(f"\n🔍 Approval Request #{i}")
        print(f"   Token: {approval['token'][:8]}...")
        print(f"   Reason: {approval['reason']}")
        print(f"   Created: {approval['created_at']}")
        print(f"   User: {approval.get('user_id', 'Unknown')}")

        approval_data = approval.get("approval_data", {})
        if approval_data:
            print(f"   Operation: {approval_data.get('operation', 'Unknown')}")
            if "amount" in approval_data:
                print(f"   Amount: ${approval_data['amount']:,.2f}")
            if "recipient_count" in approval_data:
                print(f"   Recipients: {approval_data['recipient_count']}")
            print(f"   Risk Level: {approval_data.get('risk_level', 'Unknown')}")

        # Simulate human decision (in real app, this comes from UI)
        print("   👤 Human Decision: APPROVE")

        # Approve the request
        success = await suspension_manager.approve_request(
            token=approval["token"],
            approved=True,
            approver_id="manager_001",
            approval_data={"approved_at": datetime.now().isoformat()},
        )

        if success:
            print("   ✅ Approved successfully")
        else:
            print("   ❌ Failed to approve")


async def simulate_resumption():
    """Simulate resuming suspended executions after approval."""
    print("\n🔄 Resuming Suspended Executions")
    print("=" * 60)

    agent = await create_financial_agent()

    # Get pending approvals that are now approved
    pending = await suspension_manager.get_pending_approvals()

    if not pending:
        print("📋 No pending approvals found - they may have been processed")
        return

    for approval in pending:
        token = approval["token"]
        print(f"\n🔄 Resuming execution: {token[:8]}...")

        try:
            # Resume the execution
            result = await agent.resume_async(
                resumption_token=token,
                approval_data={"approved": True, "approver": "manager_001"},
            )

            if result.is_suspended:
                print(f"   ⏸️  Still suspended: {result.suspension_reason}")
            else:
                print(f"   ✅ Completed: {result.text}")

        except Exception as e:
            print(f"   ❌ Error resuming: {e}")


async def demonstrate_concurrent_operations():
    """Demonstrate that multiple operations can be suspended concurrently."""
    print("\n🔀 Concurrent Operations Demo")
    print("=" * 60)

    agent = await create_financial_agent()

    # Start multiple operations concurrently
    tasks = [
        agent.run_async("Transfer $5000 to ACC-100"),
        agent.run_async("Transfer $8000 to ACC-200"),
        agent.run_async("Send email to 200 recipients about quarterly results"),
        agent.run_async("Check balance for ACC-300"),
    ]

    print("🚀 Starting 4 concurrent operations...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    completed = 0
    suspended = 0
    errors = 0

    for i, result in enumerate(results, 1):
        if isinstance(result, Exception):
            print(f"   Operation {i}: ❌ Error - {result}")
            errors += 1
        elif hasattr(result, "is_suspended") and result.is_suspended:
            suspension_reason = getattr(result, "suspension_reason", "Unknown reason")
            print(f"   Operation {i}: ⏸️  Suspended - {suspension_reason}")
            suspended += 1
        else:
            print(f"   Operation {i}: ✅ Completed")
            completed += 1

    print(
        f"\n📊 Results: {completed} completed, {suspended} suspended, {errors} errors"
    )
    print("💡 All operations processed concurrently without blocking!")


async def main():
    """Run the complete async HITL demonstration."""
    print("🚀 Starting Asynchronous Human-in-the-Loop Demo")
    print("=" * 80)
    print("This demo shows how agents can suspend execution for approval")
    print("without blocking the entire process - crucial for production systems!")
    print("=" * 80)

    try:
        # Step 1: Simulate user requests (some will suspend)
        await simulate_user_requests()

        # Step 2: Show concurrent operations
        await demonstrate_concurrent_operations()

        # Step 3: Simulate human approval interface
        await simulate_approval_interface()

        # Step 4: Resume suspended executions
        await simulate_resumption()

        print("\n🎉 Async HITL Demo completed successfully!")
        print("\n🔑 Key Benefits Demonstrated:")
        print("✅ Non-blocking suspension for human approval")
        print("✅ Concurrent handling of multiple requests")
        print("✅ Persistent suspension state management")
        print("✅ Flexible approval workflows")
        print("✅ Production-ready asynchronous patterns")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("Starting Asynchronous HITL Demo...")
    asyncio.run(main())
