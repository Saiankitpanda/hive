"""
Server module for running the dashboard.
"""

import threading
import uvicorn
import logging
from .app import create_app

logger = logging.getLogger(__name__)

def run_dashboard(agent_runner, host="127.0.0.1", port=8000):
    """
    Run the dashboard server.

    Args:
        agent_runner: The AgentRunner instance to visualize/control.
        host: Host to bind to.
        port: Port to bind to.
    """
    app = create_app(agent_runner)

    logger.info(f"Starting dashboard at http://{host}:{port}")

    # We run uvicorn directly. In a real CLI usage, this might block,
    # so we might want to run it in a separate thread if the CLI needs to do other things,
    # but usually the dashboard IS the main process when launched via `core dashboard`.
    uvicorn.run(app, host=host, port=port, log_level="info")
