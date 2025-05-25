"""
Complex HITL with Pipelines and Teams Example

This example demonstrates how Human-in-the-Loop (HITL) workflows work with:
1. Agent Pipelines - Sequential execution with suspension/resumption
2. Agent Teams - Dynamic orchestration with suspension/resumption
3. Nested scenarios - Teams within pipelines, complex approval workflows

Key scenarios covered:
- Pipeline suspension at different steps
- Team suspension during orchestration
- Multiple suspension points in complex workflows
- State preservation and resumption logic
"""

# type: ignore

import asyncio
from datetime import datetime

from agentle.agents.agent import Agent
from agentle.agents.agent_pipeline import AgentPipeline
from agentle.agents.agent_team import AgentTeam
from agentle.agents.errors.tool_suspension_error import ToolSuspensionError
from agentle.agents.suspension_manager import (
    SuspensionManager,
    SQLiteSuspensionStore,
)
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)
from agentle.generations.tools.tool import Tool


# Global suspension manager for this example
suspension_manager = SuspensionManager(SQLiteSuspensionStore("complex_hitl_demo.db"))

# Global storage for suspended tokens
suspended_executions: list[dict[str, str]] = []


def financial_analysis_tool(company: str, analysis_type: str) -> str:
    """
    Financial analysis tool that requires approval for sensitive analyses.
    """
    sensitive_types = ["insider_trading", "merger_analysis", "bankruptcy_prediction"]

    if analysis_type in sensitive_types:
        raise ToolSuspensionError(
            reason=f"Financial analysis of type '{analysis_type}' for {company} requires compliance approval",
            approval_data={
                "operation": "financial_analysis",
                "company": company,
                "analysis_type": analysis_type,
                "risk_level": "high",
                "compliance_required": True,
                "requested_at": datetime.now().isoformat(),
            },
            timeout_seconds=7200,  # 2 hours
        )

    return f"✅ Financial analysis completed: {analysis_type} for {company}"


def data_access_tool(database: str, query_type: str) -> str:
    """
    Data access tool that requires approval for sensitive databases.
    """
    sensitive_databases = ["customer_pii", "financial_records", "employee_data"]

    if database in sensitive_databases:
        raise ToolSuspensionError(
            reason=f"Access to {database} database requires data governance approval",
            approval_data={
                "operation": "data_access",
                "database": database,
                "query_type": query_type,
                "risk_level": "high",
                "governance_required": True,
                "requested_at": datetime.now().isoformat(),
            },
            timeout_seconds=3600,  # 1 hour
        )

    return f"✅ Data access completed: {query_type} on {database}"


def report_generation_tool(report_type: str, recipients: str) -> str:
    """
    Report generation tool that requires approval for external distribution.
    """
    external_recipients = [
        "external_auditors",
        "regulatory_bodies",
        "public_disclosure",
    ]

    if recipients in external_recipients:
        raise ToolSuspensionError(
            reason=f"Report distribution to {recipients} requires legal approval",
            approval_data={
                "operation": "report_distribution",
                "report_type": report_type,
                "recipients": recipients,
                "risk_level": "critical",
                "legal_review_required": True,
                "requested_at": datetime.now().isoformat(),
            },
            timeout_seconds=14400,  # 4 hours
        )

    return f"✅ Report generated and distributed: {report_type} to {recipients}"


async def create_financial_pipeline() -> AgentPipeline:
    """Create a financial analysis pipeline with HITL-enabled tools."""

    # Data Collection Agent
    data_agent = Agent(
        name="Data Collection Agent",
        generation_provider=GoogleGenaiGenerationProvider(),
        model="gemini-2.0-flash",
        instructions="""You are a data collection specialist. Your job is to gather the necessary 
        financial data for analysis. Use the data_access_tool to retrieve information from various databases.""",
        tools=[Tool.from_callable(data_access_tool)],
        suspension_manager=suspension_manager,
    )

    # Analysis Agent
    analysis_agent = Agent(
        name="Financial Analysis Agent",
        generation_provider=GoogleGenaiGenerationProvider(),
        model="gemini-2.0-flash",
        instructions="""You are a financial analyst. Analyze the data provided and perform 
        comprehensive financial analysis using the financial_analysis_tool.""",
        tools=[Tool.from_callable(financial_analysis_tool)],
        suspension_manager=suspension_manager,
    )

    # Reporting Agent
    reporting_agent = Agent(
        name="Report Generation Agent",
        generation_provider=GoogleGenaiGenerationProvider(),
        model="gemini-2.0-flash",
        instructions="""You are a report specialist. Create comprehensive reports and distribute 
        them to appropriate stakeholders using the report_generation_tool.""",
        tools=[Tool.from_callable(report_generation_tool)],
        suspension_manager=suspension_manager,
    )

    return AgentPipeline(
        agents=[data_agent, analysis_agent, reporting_agent], debug_mode=True
    )


async def create_compliance_team() -> AgentTeam:
    """Create a compliance team with HITL-enabled agents."""

    provider = GoogleGenaiGenerationProvider()

    # Risk Assessment Agent
    risk_agent = Agent(
        name="Risk Assessment Agent",
        description="Specialized in assessing financial and operational risks",
        generation_provider=provider,
        model="gemini-2.0-flash",
        instructions="You assess risks and perform due diligence using financial analysis tools.",
        tools=[Tool.from_callable(financial_analysis_tool)],
        suspension_manager=suspension_manager,
    )

    # Data Governance Agent
    governance_agent = Agent(
        name="Data Governance Agent",
        description="Specialized in data access and privacy compliance",
        generation_provider=provider,
        model="gemini-2.0-flash",
        instructions="You handle data access requests and ensure compliance with data governance policies.",
        tools=[Tool.from_callable(data_access_tool)],
        suspension_manager=suspension_manager,
    )

    # Legal Compliance Agent
    legal_agent = Agent(
        name="Legal Compliance Agent",
        description="Specialized in legal review and regulatory compliance",
        generation_provider=provider,
        model="gemini-2.0-flash",
        instructions="You handle legal reviews and ensure regulatory compliance for reports and disclosures.",
        tools=[Tool.from_callable(report_generation_tool)],
        suspension_manager=suspension_manager,
    )

    return AgentTeam(
        agents=[risk_agent, governance_agent, legal_agent],
        orchestrator_provider=provider,
        orchestrator_model="gemini-2.0-flash",
    )


async def demonstrate_pipeline_suspension():
    """Demonstrate suspension and resumption in an Agent Pipeline."""
    print("\n🔄 Pipeline Suspension Demo")
    print("=" * 60)

    pipeline = await create_financial_pipeline()

    # Start a pipeline that will suspend at the analysis step
    print("📊 Starting financial analysis pipeline...")
    result = await pipeline.run_async(
        "Analyze TechCorp for potential merger_analysis and generate a report for external_auditors"
    )

    if result.is_suspended and result.resumption_token:
        print(f"⏸️  Pipeline suspended: {result.suspension_reason}")
        print(f"📋 Resumption token: {result.resumption_token}")
        suspended_executions.append(
            {
                "type": "pipeline",
                "token": result.resumption_token,  # type: ignore
                "reason": result.suspension_reason or "Unknown reason",
            }
        )

        # Simulate approval process
        print("👤 Simulating compliance approval...")
        await suspension_manager.approve_request(
            token=result.resumption_token,  # type: ignore
            approved=True,
            approver_id="compliance_officer_001",
            approval_data={
                "approved_via": "compliance_dashboard",
                "notes": "Approved for merger analysis",
            },
        )

        # Resume the pipeline
        print("🔄 Resuming pipeline execution...")
        resumed_result = await pipeline.resume_async(result.resumption_token)  # type: ignore

        if resumed_result.is_suspended:
            print(f"⏸️  Pipeline suspended again: {resumed_result.suspension_reason}")
            print(f"📋 New resumption token: {resumed_result.resumption_token}")
            suspended_executions.append(
                {
                    "type": "pipeline_continued",
                    "token": resumed_result.resumption_token,
                    "reason": resumed_result.suspension_reason,
                }
            )

            # Approve the second suspension (report distribution)
            print("👤 Simulating legal approval...")
            await suspension_manager.approve_request(
                token=resumed_result.resumption_token,
                approved=True,
                approver_id="legal_counsel_001",
                approval_data={
                    "approved_via": "legal_review_system",
                    "notes": "Approved for external distribution",
                },
            )

            # Resume again
            print("🔄 Resuming pipeline execution (final step)...")
            final_result = await pipeline.resume_async(resumed_result.resumption_token)
            print(f"✅ Pipeline completed: {final_result.text}")
        else:
            print(
                f"✅ Pipeline completed after first resumption: {resumed_result.text}"
            )
    else:
        print(f"✅ Pipeline completed without suspension: {result.text}")


async def demonstrate_team_suspension():
    """Demonstrate suspension and resumption in an Agent Team."""
    print("\n👥 Team Suspension Demo")
    print("=" * 60)

    team = await create_compliance_team()

    # Start a team task that will require multiple approvals
    print("🏢 Starting compliance team task...")
    result = await team.run_async(
        "Perform insider_trading analysis on FinanceCorpand access customer_pii database for compliance review"
    )

    if result.is_suspended:
        print(f"⏸️  Team suspended: {result.suspension_reason}")
        print(f"📋 Resumption token: {result.resumption_token}")
        suspended_executions.append(
            {
                "type": "team",
                "token": result.resumption_token,
                "reason": result.suspension_reason,
            }
        )

        # Simulate approval
        print("👤 Simulating team approval...")
        await suspension_manager.approve_request(
            token=result.resumption_token,
            approved=True,
            approver_id="team_supervisor_001",
            approval_data={
                "approved_via": "team_dashboard",
                "notes": "Approved for compliance analysis",
            },
        )

        # Resume the team
        print("🔄 Resuming team execution...")
        resumed_result = await team.resume_async(result.resumption_token)

        if resumed_result.is_suspended:
            print(f"⏸️  Team suspended again: {resumed_result.suspension_reason}")
            # Handle additional suspensions...
            await suspension_manager.approve_request(
                token=resumed_result.resumption_token,
                approved=True,
                approver_id="data_governance_lead_001",
            )
            final_result = await team.resume_async(resumed_result.resumption_token)
            print(f"✅ Team task completed: {final_result.text}")
        else:
            print(
                f"✅ Team task completed after first resumption: {resumed_result.text}"
            )
    else:
        print(f"✅ Team task completed without suspension: {result.text}")


async def demonstrate_nested_scenario():
    """Demonstrate a complex nested scenario with teams within pipelines."""
    print("\n🔀 Nested Scenario Demo (Team within Pipeline)")
    print("=" * 60)

    # Create a pipeline where one step uses a team
    provider = GoogleGenaiGenerationProvider()

    # Initial data prep agent
    prep_agent = Agent(
        name="Data Preparation Agent",
        generation_provider=provider,
        model="gemini-2.0-flash",
        instructions="You prepare data for complex analysis workflows.",
        suspension_manager=suspension_manager,
    )

    # Compliance team (as created above)
    compliance_team = await create_compliance_team()

    # Final reporting agent
    final_agent = Agent(
        name="Final Report Agent",
        generation_provider=provider,
        model="gemini-2.0-flash",
        instructions="You create final comprehensive reports.",
        tools=[Tool.from_callable(report_generation_tool)],
        suspension_manager=suspension_manager,
    )

    print("🏗️  Starting nested workflow...")
    print("   Step 1: Data preparation")
    prep_result = await prep_agent.run_async(
        "Prepare data for comprehensive compliance analysis of MegaCorp"
    )

    print("   Step 2: Compliance team analysis")
    team_result = await compliance_team.run_async(prep_result.text)

    if team_result.is_suspended:
        print(f"   ⏸️  Nested team suspended: {team_result.suspension_reason}")
        suspended_executions.append(
            {
                "type": "nested_team",
                "token": team_result.resumption_token,
                "reason": team_result.suspension_reason,
            }
        )

        # Approve and resume
        await suspension_manager.approve_request(
            token=team_result.resumption_token,
            approved=True,
            approver_id="nested_approver_001",
        )
        team_result = await compliance_team.resume_async(team_result.resumption_token)

    print("   Step 3: Final reporting")
    final_result = await final_agent.run_async(
        f"Create final report based on: {team_result.text}"
    )

    if final_result.is_suspended:
        print(f"   ⏸️  Final step suspended: {final_result.suspension_reason}")
        await suspension_manager.approve_request(
            token=final_result.resumption_token,
            approved=True,
            approver_id="final_approver_001",
        )
        final_result = await final_agent.resume_async(final_result.resumption_token)

    print(f"✅ Nested workflow completed: {final_result.text}")


async def show_suspension_summary():
    """Show a summary of all suspensions that occurred."""
    print("\n📊 Suspension Summary")
    print("=" * 60)

    pending_approvals = await suspension_manager.get_pending_approvals()

    print(f"📋 Total suspensions handled: {len(suspended_executions)}")
    print(f"📋 Currently pending approvals: {len(pending_approvals)}")

    for i, suspension in enumerate(suspended_executions, 1):
        print(f"   {i}. {suspension['type']}: {suspension['reason']}")

    if pending_approvals:
        print("\n⏳ Pending approvals:")
        for approval in pending_approvals:
            print(f"   - {approval['reason']} (Token: {approval['token'][:8]}...)")


async def main():
    """Run the complete complex HITL demonstration."""
    print("🚀 Complex HITL with Pipelines and Teams Demo")
    print("=" * 80)
    print("This demo shows how HITL works in complex scenarios:")
    print("• Agent Pipelines with multiple suspension points")
    print("• Agent Teams with dynamic orchestration and suspensions")
    print("• Nested workflows with teams within pipelines")
    print("• State preservation and resumption across complex workflows")
    print("=" * 80)

    try:
        # Demonstrate different scenarios
        await demonstrate_pipeline_suspension()
        await demonstrate_team_suspension()
        await demonstrate_nested_scenario()
        await show_suspension_summary()

        print("\n🎉 Complex HITL Demo completed successfully!")
        print("\n🔑 Key Insights:")
        print("✅ Pipelines can suspend at any step and resume from that exact point")
        print(
            "✅ Teams can suspend during orchestration and maintain conversation state"
        )
        print(
            "✅ Nested workflows (teams within pipelines) handle suspensions gracefully"
        )
        print("✅ State preservation works across complex multi-agent scenarios")
        print("✅ Multiple suspension points in a single workflow are fully supported")
        print(
            "✅ Each suspension type (pipeline/team/nested) maintains its own context"
        )

        print("\n💡 Production Benefits:")
        print(
            "🏢 Enterprise workflows can pause for compliance without losing progress"
        )
        print("🔄 Complex multi-step processes remain resumable across days/weeks")
        print("👥 Team-based workflows maintain orchestration state during suspensions")
        print("🔀 Nested scenarios (common in enterprise) are fully supported")
        print("📊 Complete audit trail of all suspensions and approvals")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("Starting Complex HITL Demo...")
    asyncio.run(main())
