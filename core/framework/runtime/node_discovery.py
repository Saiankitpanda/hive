"""
Node Discovery Tool - Find and interact with other agents in the graph.

Allows agents to:
- Discover other available agents/nodes
- Query agent capabilities
- Request collaboration
- Delegate tasks to specialized agents

Part of the Core Agent Tools from ROADMAP.
"""

from dataclasses import dataclass, field
from typing import Any
from datetime import datetime


@dataclass
class AgentInfo:
    """Information about a discovered agent."""

    id: str
    name: str
    description: str
    capabilities: list[str]
    status: str  # "available", "busy", "offline"
    version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "status": self.status,
            "version": self.version,
            "metadata": self.metadata,
        }


@dataclass
class DiscoveryResult:
    """Result of a discovery query."""

    agents: list[AgentInfo]
    query: str
    timestamp: float
    total_found: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "agents": [a.to_dict() for a in self.agents],
            "query": self.query,
            "timestamp": self.timestamp,
            "total_found": self.total_found,
        }


class NodeDiscoveryService:
    """
    Service for discovering and querying agents in a graph.

    Usage:
        discovery = NodeDiscoveryService()

        # Register an agent
        discovery.register(AgentInfo(
            id="research-agent",
            name="Research Agent",
            description="Searches and summarizes information",
            capabilities=["web_search", "summarization"],
            status="available"
        ))

        # Find agents by capability
        results = discovery.find_by_capability("web_search")

        # Find by name
        agent = discovery.find_by_id("research-agent")

        # Get all available agents
        available = discovery.get_available()
    """

    def __init__(self):
        self._agents: dict[str, AgentInfo] = {}
        self._capability_index: dict[str, list[str]] = {}

    def register(self, agent: AgentInfo) -> None:
        """Register an agent in the discovery service."""
        self._agents[agent.id] = agent

        # Index capabilities
        for cap in agent.capabilities:
            if cap not in self._capability_index:
                self._capability_index[cap] = []
            if agent.id not in self._capability_index[cap]:
                self._capability_index[cap].append(agent.id)

    def unregister(self, agent_id: str) -> bool:
        """Unregister an agent."""
        if agent_id not in self._agents:
            return False

        agent = self._agents.pop(agent_id)

        # Remove from capability index
        for cap in agent.capabilities:
            if cap in self._capability_index:
                self._capability_index[cap] = [
                    a for a in self._capability_index[cap] if a != agent_id
                ]

        return True

    def update_status(self, agent_id: str, status: str) -> bool:
        """Update agent status."""
        if agent_id in self._agents:
            self._agents[agent_id].status = status
            return True
        return False

    def find_by_id(self, agent_id: str) -> AgentInfo | None:
        """Find agent by ID."""
        return self._agents.get(agent_id)

    def find_by_name(self, name: str) -> list[AgentInfo]:
        """Find agents by name (partial match)."""
        name_lower = name.lower()
        return [a for a in self._agents.values() if name_lower in a.name.lower()]

    def find_by_capability(self, capability: str) -> DiscoveryResult:
        """Find agents with a specific capability."""
        agent_ids = self._capability_index.get(capability, [])
        agents = [self._agents[aid] for aid in agent_ids if aid in self._agents]

        return DiscoveryResult(
            agents=agents,
            query=f"capability:{capability}",
            timestamp=datetime.now().timestamp(),
            total_found=len(agents),
        )

    def find_by_capabilities(self, capabilities: list[str]) -> DiscoveryResult:
        """Find agents with ALL specified capabilities."""
        matching = []

        for agent in self._agents.values():
            if all(cap in agent.capabilities for cap in capabilities):
                matching.append(agent)

        return DiscoveryResult(
            agents=matching,
            query=f"capabilities:{','.join(capabilities)}",
            timestamp=datetime.now().timestamp(),
            total_found=len(matching),
        )

    def get_available(self) -> list[AgentInfo]:
        """Get all available agents."""
        return [a for a in self._agents.values() if a.status == "available"]

    def get_all(self) -> list[AgentInfo]:
        """Get all registered agents."""
        return list(self._agents.values())

    def get_capabilities(self) -> list[str]:
        """Get all registered capabilities."""
        return list(self._capability_index.keys())

    def search(self, query: str) -> DiscoveryResult:
        """Search agents by query string."""
        query_lower = query.lower()
        matching = []

        for agent in self._agents.values():
            # Match name, description, or capabilities
            if (
                query_lower in agent.name.lower()
                or query_lower in agent.description.lower()
                or any(query_lower in cap.lower() for cap in agent.capabilities)
            ):
                matching.append(agent)

        return DiscoveryResult(
            agents=matching,
            query=query,
            timestamp=datetime.now().timestamp(),
            total_found=len(matching),
        )


# Tool definition for use with LLM providers
NODE_DISCOVERY_TOOL = {
    "name": "discover_agents",
    "description": "Find other agents in the system that can help with specific tasks",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find agents (by name, capability, or description)",
            },
            "capability": {
                "type": "string",
                "description": "Specific capability to search for (e.g., 'web_search', 'code_generation')",
            },
        },
    },
}


def create_discovery_tool_executor(service: NodeDiscoveryService):
    """Create a tool executor function for the discovery service."""

    def executor(tool_input: dict[str, Any]) -> str:
        query = tool_input.get("query")
        capability = tool_input.get("capability")

        if capability:
            result = service.find_by_capability(capability)
        elif query:
            result = service.search(query)
        else:
            agents = service.get_available()
            result = DiscoveryResult(
                agents=agents,
                query="all_available",
                timestamp=datetime.now().timestamp(),
                total_found=len(agents),
            )

        if result.total_found == 0:
            return "No agents found matching your criteria."

        lines = [f"Found {result.total_found} agent(s):"]
        for agent in result.agents:
            lines.append(f"\n**{agent.name}** ({agent.id})")
            lines.append(f"  Status: {agent.status}")
            lines.append(f"  Capabilities: {', '.join(agent.capabilities)}")
            lines.append(f"  {agent.description}")

        return "\n".join(lines)

    return executor
