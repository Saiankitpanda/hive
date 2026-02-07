"""
Sample Knowledge Agent - Example agent for documentation and learning.

This agent demonstrates:
- Node-based architecture
- LLM integration
- Tool usage
- State management
- Human-in-the-loop patterns

Use this as a starting point for building your own agents.
"""

import json
from pathlib import Path
from typing import Any

# Agent configuration
AGENT_CONFIG = {
    "name": "knowledge-agent",
    "version": "1.0.0",
    "description": "A sample agent that answers questions using a knowledge base",
    "author": "Hive Contributors",
}

# Sample knowledge base
KNOWLEDGE_BASE = {
    "hive": {
        "description": "Hive is an open-source AI agent framework",
        "features": [
            "Node-based architecture",
            "Multi-LLM support via LiteLLM",
            "Human-in-the-loop patterns",
            "Goal-driven execution",
            "Real-time dashboard",
        ],
        "github": "https://github.com/adenhq/hive",
    },
    "agents": {
        "description": "Agents are autonomous AI systems that can perform tasks",
        "types": [
            "Coding agents",
            "Research agents",
            "Sales agents",
            "Customer support agents",
        ],
    },
    "llm": {
        "description": "Large Language Models are AI models trained on text data",
        "providers": ["OpenAI", "Anthropic", "Google", "Moonshot (Kimi)"],
    },
}


def search_knowledge(query: str) -> dict[str, Any]:
    """
    Search the knowledge base for relevant information.

    Args:
        query: Search query

    Returns:
        Dictionary with search results
    """
    query_lower = query.lower()
    results = []

    for topic, info in KNOWLEDGE_BASE.items():
        if topic in query_lower or query_lower in info.get("description", "").lower():
            results.append(
                {
                    "topic": topic,
                    "info": info,
                }
            )

    return {
        "query": query,
        "results": results,
        "count": len(results),
    }


def format_response(search_result: dict[str, Any]) -> str:
    """
    Format search results into a human-readable response.

    Args:
        search_result: Result from search_knowledge

    Returns:
        Formatted string response
    """
    if search_result["count"] == 0:
        return f"I couldn't find information about '{search_result['query']}' in my knowledge base."

    response_parts = [f"Here's what I found about '{search_result['query']}':\n"]

    for result in search_result["results"]:
        topic = result["topic"]
        info = result["info"]

        response_parts.append(f"\n## {topic.title()}")
        response_parts.append(info.get("description", ""))

        if "features" in info:
            response_parts.append("\n**Features:**")
            for feature in info["features"]:
                response_parts.append(f"- {feature}")

        if "types" in info:
            response_parts.append("\n**Types:**")
            for t in info["types"]:
                response_parts.append(f"- {t}")

        if "providers" in info:
            response_parts.append("\n**Providers:** " + ", ".join(info["providers"]))

        if "github" in info:
            response_parts.append(f"\n**GitHub:** {info['github']}")

    return "\n".join(response_parts)


class KnowledgeAgent:
    """
    Sample knowledge agent for answering questions.

    Usage:
        agent = KnowledgeAgent()
        response = agent.ask("What is Hive?")
        print(response)

        # With state persistence
        agent = KnowledgeAgent(storage_path="./agent_state")
        agent.ask("What are agents?")
        history = agent.get_history()
    """

    def __init__(
        self,
        storage_path: str | Path | None = None,
        llm_provider: Any = None,
    ):
        """
        Initialize the knowledge agent.

        Args:
            storage_path: Path to store conversation history
            llm_provider: Optional LLM provider for enhanced responses
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.llm = llm_provider
        self.history: list[dict[str, str]] = []

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_history()

    def ask(self, question: str) -> str:
        """
        Ask a question to the knowledge agent.

        Args:
            question: The question to ask

        Returns:
            Answer string
        """
        # Search knowledge base
        results = search_knowledge(question)
        response = format_response(results)

        # Enhance with LLM if available
        if self.llm and results["count"] > 0:
            response = self._enhance_with_llm(question, results, response)

        # Record in history
        self._record_interaction(question, response)

        return response

    def get_history(self) -> list[dict[str, str]]:
        """Get conversation history."""
        return self.history.copy()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []
        if self.storage_path:
            history_file = self.storage_path / "history.json"
            if history_file.exists():
                history_file.unlink()

    def add_knowledge(self, topic: str, info: dict[str, Any]) -> None:
        """
        Add new knowledge to the agent.

        Args:
            topic: Topic name
            info: Dictionary with description and other info
        """
        KNOWLEDGE_BASE[topic] = info

    def _enhance_with_llm(
        self,
        question: str,
        results: dict[str, Any],
        basic_response: str,
    ) -> str:
        """Enhance response using LLM."""
        try:
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Based on this knowledge:\n{basic_response}\n\n"
                        f"Please provide a helpful answer to: {question}"
                    ),
                }
            ]

            llm_response = self.llm.complete(
                messages=messages,
                system="You are a helpful knowledge assistant. Answer concisely.",
                max_tokens=500,
            )

            return llm_response.content

        except Exception:
            # Fall back to basic response
            return basic_response

    def _record_interaction(self, question: str, answer: str) -> None:
        """Record an interaction in history."""
        self.history.append(
            {
                "question": question,
                "answer": answer,
            }
        )

        if self.storage_path:
            self._save_history()

    def _load_history(self) -> None:
        """Load history from storage."""
        if not self.storage_path:
            return

        history_file = self.storage_path / "history.json"
        if history_file.exists():
            with open(history_file) as f:
                self.history = json.load(f)

    def _save_history(self) -> None:
        """Save history to storage."""
        if not self.storage_path:
            return

        history_file = self.storage_path / "history.json"
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=2)


# Example usage
if __name__ == "__main__":
    # Create agent
    agent = KnowledgeAgent()

    # Ask questions
    questions = [
        "What is Hive?",
        "Tell me about agents",
        "What LLM providers are supported?",
    ]

    for q in questions:
        print(f"\n❓ {q}")
        answer = agent.ask(q)
        print(f"💡 {answer}")
