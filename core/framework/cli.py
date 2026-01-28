"""
Command-line interface for Goal Agent.

Usage:
    python -m core run exports/my-agent --input '{"key": "value"}'
    python -m core info exports/my-agent
    python -m core validate exports/my-agent
    python -m core list exports/
    python -m core dispatch exports/ --input '{"key": "value"}'
    python -m core shell exports/my-agent

Testing commands:
    python -m core test-run <agent_path> --goal <goal_id>
    python -m core test-debug <goal_id> <test_id>
    python -m core test-list <goal_id>
    python -m core test-stats <goal_id>
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Goal Agent - Build and run goal-driven agents")
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Anthropic model to use",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Register runner commands (run, info, validate, list, dispatch, shell)
    from framework.runner.cli import register_commands

    register_commands(subparsers)

    # Register testing commands (test-run, test-debug, test-list, test-stats)
    from framework.testing.cli import register_testing_commands

    register_testing_commands(subparsers)

    # Register dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Run the agent dashboard")
    dashboard_parser.add_argument("agent_path", help="Path to agent export")
    dashboard_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    dashboard_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    dashboard_parser.add_argument("--input", default="{}", help="Initial input JSON (optional)")

    def run_dashboard_command(args):
        from pathlib import Path
        from framework.runner.runner import AgentRunner
        from framework.dashboard import run_dashboard

        agent_path = Path(args.agent_path)

        # Load agent
        print(f"Loading agent from {agent_path}...")
        runner = AgentRunner.load(agent_path, model=args.model)

        # Enable dashboard mode (forces AgentRuntime/EventBus usage)
        if hasattr(runner, "enable_dashboard_mode"):
            runner.enable_dashboard_mode()

        # Run dashboard
        run_dashboard(runner, host=args.host, port=args.port)
        return 0

    dashboard_parser.set_defaults(func=run_dashboard_command)

    args = parser.parse_args()

    if hasattr(args, "func"):
        sys.exit(args.func(args))


if __name__ == "__main__":
    main()
