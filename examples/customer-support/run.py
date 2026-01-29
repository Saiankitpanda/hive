import asyncio
import os
import sys
from pathlib import Path

# Add the repository root to the python path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(repo_root / "core"))

from framework.runner.runner import AgentRunner


async def main():
    agent_path = Path(__file__).parent

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("CEREBRAS_API_KEY"):
        print("⚠️  No API key found. Please set OPENAI_API_KEY or CEREBRAS_API_KEY.")
        print("   Example: export OPENAI_API_KEY=sk-...")
        return

    print(f"🚀 Loading Customer Support Agent from {agent_path}...")

    # Initialize runner
    runner = AgentRunner.load(agent_path)

    # Sample ticket
    ticket = {
        "user_id": "u123",
        "subject": "500 Error on API",
        "body": "I am getting a 500 error when calling the /v1/generate endpoint. My API key starts with sk-123.",
    }

    print("\n📩 Incoming Ticket:")
    print(f"   Subject: {ticket['subject']}")
    print(f"   Body: {ticket['body']}")
    print("\n🤖 Agent starting execution...")

    # Run the agent
    result = await runner.run(input_data={"ticket": ticket})

    # Handle Human-in-the-Loop (HITL)
    if result.paused_at:
        print(f"\n⏸  Execution paused at node: {result.paused_at}")

        # In a real app, you would fetch this from the DB/UI
        memory = result.output
        draft = memory.get("draft_body", "(No draft found)")

        print("\n📝 Draft Response for Review:")
        print("-" * 40)
        print(draft)
        print("-" * 40)

        print("\n👤 Human Review Required.")
        choice = input("   Approve this draft? (y/n/edit): ").strip().lower()

        feedback = ""
        approved = False

        if choice == "y":
            approved = True
            print("   ✅ Approved.")
        elif choice == "edit":
            feedback = input("   Enter feedback: ")
            print(
                "   (Note: In a full implementation, this would loop back to refine. For now we proceed with 'false' approval to terminal.)"
            )
            # For simplicity in this demo, we just pass the feedback
        else:
            print("   ❌ Rejected.")

        # Resume execution
        print("\n▶️  Resuming execution...")
        result = await runner.run(
            session_state=result.session_state,
            input_data={"approved": approved, "feedback": feedback},
        )

    # Final result
    if result.success:
        print("\n✅ Execution completed successfully!")
        print("   Output State:")
        for k, v in result.output.items():
            if k == "ticket":
                continue  # Skip echoing the huge input
            print(f"   • {k}: {str(v)[:100]}...")
    else:
        print(f"\n❌ Execution failed: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
