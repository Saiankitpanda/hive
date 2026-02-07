"""
Command-line interface for Hive Agent Framework.

Usage:
    python -m framework run exports/my-agent --input '{"key": "value"}'
    python -m framework info exports/my-agent
    python -m framework validate exports/my-agent
    python -m framework list exports/
    python -m framework dispatch exports/ --input '{"key": "value"}'
    python -m framework shell exports/my-agent

Dashboard:
    python -m framework dashboard exports/my-agent
    python -m framework dashboard exports/my-agent --port 3000
    python -m framework dashboard exports/my-agent --host 0.0.0.0 --port 8000

Testing commands:
    python -m framework test-run <agent_path> --goal <goal_id>
    python -m framework test-debug <goal_id> <test_id>
    python -m framework test-list <goal_id>
    python -m framework test-stats <goal_id>

For more info: https://github.com/adenhq/hive
"""

import argparse
import sys

__version__ = "1.1.0"


def main():
    parser = argparse.ArgumentParser(
        description="Hive Agent Framework - Build and run goal-driven AI agents",
        epilog="Documentation: https://docs.adenhq.com | GitHub: https://github.com/adenhq/hive",
    )
    parser.add_argument(
        "--version", "-v", action="version", version=f"Hive Agent Framework v{__version__}"
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="LLM model to use (default: claude-haiku-4-5-20251001)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Register runner commands (run, info, validate, list, dispatch, shell)
    from framework.runner.cli import register_commands

    register_commands(subparsers)

    # Register testing commands (test-run, test-debug, test-list, test-stats)
    from framework.testing.cli import register_testing_commands

    register_testing_commands(subparsers)

    # Register dashboard command
    dashboard_parser = subparsers.add_parser(
        "dashboard", help="Launch the real-time agent dashboard with graph visualization"
    )
    dashboard_parser.add_argument("agent_path", help="Path to agent export directory")
    dashboard_parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)"
    )
    dashboard_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind (default: 8000)"
    )
    dashboard_parser.add_argument("--input", default="{}", help="Initial input JSON (optional)")

    def run_dashboard_command(args):
        from pathlib import Path
        from framework.runner.runner import AgentRunner
        from framework.dashboard import run_dashboard

        agent_path = Path(args.agent_path)

        # Load agent
        print(f"🐝 Loading agent from {agent_path}...")
        runner = AgentRunner.load(agent_path, model=args.model)

        # Enable dashboard mode (forces AgentRuntime/EventBus usage)
        if hasattr(runner, "enable_dashboard_mode"):
            runner.enable_dashboard_mode()

        print(f"🚀 Starting dashboard at http://{args.host}:{args.port}")
        print("   Press Ctrl+C to stop")

        # Run dashboard
        run_dashboard(runner, host=args.host, port=args.port)
        return 0

    dashboard_parser.set_defaults(func=run_dashboard_command)

    args = parser.parse_args()

    if hasattr(args, "func"):
        sys.exit(args.func(args))


if __name__ == "__main__":
    main()
