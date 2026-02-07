# Dashboard Guide

The Hive Dashboard provides real-time visualization and monitoring of your AI agents.

## Quick Start

```bash
# Run the dashboard
PYTHONPATH=core:exports python -m framework dashboard exports/your_agent --port 8000
```

Open http://localhost:8000 in your browser.

## Features

### 📊 Agent Graph Visualization
- Interactive Mermaid diagram of your agent's node structure
- Visual representation of nodes and edges
- Different shapes for node types (router, human_input, etc.)

### 📝 Real-Time Logs
- Live WebSocket streaming of execution events
- Color-coded log levels (success, error, info)
- Expandable event data details
- Clear logs functionality

### 📈 Execution Statistics
- **Runs**: Total execution count
- **Success Rate**: Percentage of successful runs
- **Avg Time**: Average execution duration

### 🎨 Theme Toggle
- Dark mode (default)
- Light mode
- Persisted in localStorage

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+R` | Run agent |
| `Ctrl+T` | Toggle theme |
| `Ctrl+L` | Clear logs |
| `?` | Show shortcuts help |
| `Esc` | Close dialogs |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard HTML |
| `/api/graph` | GET | Agent graph structure |
| `/api/run` | POST | Trigger agent execution |
| `/ws` | WebSocket | Real-time events |

## WebSocket Events

The dashboard receives these event types:

- `execution_started` - Agent run initiated
- `execution_completed` - Run finished successfully
- `execution_failed` - Run failed with error
- `node_start` - Node execution began
- `node_complete` - Node execution finished
- `state_changed` - Agent state updated

## Troubleshooting

### Dashboard won't start
```bash
# Ensure FastAPI is installed
pip install fastapi uvicorn jinja2

# Check agent path exists
ls exports/your_agent/
```

### WebSocket disconnects
- Check firewall settings
- Dashboard auto-reconnects after 2 seconds

### Graph not rendering
- Ensure agent.json has valid nodes/edges
- Check browser console for Mermaid errors

## Configuration

```bash
# Custom host and port
python -m framework dashboard exports/agent --host 0.0.0.0 --port 3000

# With initial input
python -m framework dashboard exports/agent --input '{"key": "value"}'
```
