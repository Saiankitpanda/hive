"""
Sample Research Agent - Conducts research and synthesizes information.

Demonstrates:
- Web search integration
- Source aggregation
- Fact verification
- Report generation

From ROADMAP: Sample Agents > Research Agent
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SourceType(Enum):
    """Type of research source."""

    WEB_SEARCH = "web_search"
    DOCUMENT = "document"
    DATABASE = "database"
    API = "api"
    KNOWLEDGE_BASE = "knowledge_base"


@dataclass
class Source:
    """A research source."""

    url: str
    title: str
    snippet: str
    source_type: SourceType = SourceType.WEB_SEARCH
    relevance_score: float = 0.0
    accessed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "source_type": self.source_type.value,
            "relevance_score": self.relevance_score,
            "accessed_at": self.accessed_at,
        }


@dataclass
class ResearchResult:
    """Result of a research query."""

    query: str
    summary: str
    sources: list[Source]
    key_findings: list[str]
    confidence: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "summary": self.summary,
            "sources": [s.to_dict() for s in self.sources],
            "key_findings": self.key_findings,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }

    def to_markdown(self) -> str:
        """Export as markdown report."""
        lines = [
            f"# Research Report: {self.query}",
            "",
            f"*Generated: {self.created_at}*",
            f"*Confidence: {self.confidence:.0%}*",
            "",
            "## Summary",
            self.summary,
            "",
            "## Key Findings",
        ]

        for i, finding in enumerate(self.key_findings, 1):
            lines.append(f"{i}. {finding}")

        lines.extend(["", "## Sources"])

        for source in self.sources:
            lines.append(f"- [{source.title}]({source.url})")

        return "\n".join(lines)


@dataclass
class ResearchTask:
    """A research task to complete."""

    query: str
    depth: str = "standard"  # quick, standard, deep
    max_sources: int = 5
    focus_areas: list[str] = field(default_factory=list)
    exclude_domains: list[str] = field(default_factory=list)


class ResearchAgent:
    """
    Agent for conducting research.

    Usage:
        researcher = ResearchAgent()

        # Quick research
        result = researcher.research("What is quantum computing?")

        # Deep research with focus areas
        task = ResearchTask(
            query="Impact of AI on healthcare",
            depth="deep",
            max_sources=10,
            focus_areas=["diagnosis", "drug discovery", "patient care"]
        )
        result = researcher.research_task(task)

        # Export
        report = result.to_markdown()
    """

    # Sample knowledge base for demo
    KNOWLEDGE_BASE = {
        "ai": {
            "summary": (
                "Artificial Intelligence is the simulation of human intelligence in machines."
            ),
            "key_points": [
                "Machine learning enables systems to learn from data",
                "Deep learning uses neural networks",
                "AI applications include NLP, computer vision, robotics",
            ],
        },
        "agents": {
            "summary": "AI agents are autonomous systems that perceive and act in an environment.",
            "key_points": [
                "Agents can be reactive or deliberative",
                "Multi-agent systems enable collaboration",
                "LLM-based agents use language models for reasoning",
            ],
        },
        "llm": {
            "summary": "Large Language Models are neural networks trained on vast text data.",
            "key_points": [
                "Transformers architecture enables efficient training",
                "Models like GPT and Claude show emergent capabilities",
                "Fine-tuning adapts models to specific tasks",
            ],
        },
    }

    def __init__(
        self,
        web_search_tool: Any = None,
        llm_provider: Any = None,
    ):
        """
        Initialize research agent.

        Args:
            web_search_tool: Tool for web searches
            llm_provider: LLM for synthesis
        """
        self.web_search = web_search_tool
        self.llm = llm_provider
        self._cache: dict[str, ResearchResult] = {}

    def research(
        self,
        query: str,
        max_sources: int = 5,
    ) -> ResearchResult:
        """Conduct research on a query."""
        task = ResearchTask(query=query, max_sources=max_sources)
        return self.research_task(task)

    def research_task(self, task: ResearchTask) -> ResearchResult:
        """Execute a research task."""
        # Check cache
        cache_key = f"{task.query}:{task.depth}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Gather sources
        sources = self._gather_sources(task)

        # Extract key findings
        findings = self._extract_findings(sources, task)

        # Synthesize summary
        summary = self._synthesize(sources, findings, task)

        # Calculate confidence
        confidence = self._calculate_confidence(sources, task)

        result = ResearchResult(
            query=task.query,
            summary=summary,
            sources=sources,
            key_findings=findings,
            confidence=confidence,
        )

        # Cache result
        self._cache[cache_key] = result

        return result

    def _gather_sources(self, task: ResearchTask) -> list[Source]:
        """Gather sources for the research."""
        sources = []

        # Search knowledge base first
        kb_sources = self._search_knowledge_base(task.query)
        sources.extend(kb_sources)

        # Web search if available
        if self.web_search:
            try:
                web_sources = self._web_search(task)
                sources.extend(web_sources)
            except Exception:
                pass

        # Limit to max sources
        sources = sources[: task.max_sources]

        # Score relevance
        for source in sources:
            source.relevance_score = self._score_relevance(source, task.query)

        # Sort by relevance
        sources.sort(key=lambda s: s.relevance_score, reverse=True)

        return sources

    def _search_knowledge_base(self, query: str) -> list[Source]:
        """Search internal knowledge base."""
        sources = []
        query_lower = query.lower()

        for topic, info in self.KNOWLEDGE_BASE.items():
            if topic in query_lower or any(kw in query_lower for kw in topic.split()):
                sources.append(
                    Source(
                        url=f"kb://{topic}",
                        title=f"Knowledge Base: {topic.upper()}",
                        snippet=info["summary"],
                        source_type=SourceType.KNOWLEDGE_BASE,
                        relevance_score=0.8,
                    )
                )

        return sources

    def _web_search(self, task: ResearchTask) -> list[Source]:
        """Perform web search."""
        if not self.web_search:
            return []

        results = self.web_search.search(task.query, num_results=task.max_sources)

        sources = []
        for r in results:
            # Filter excluded domains
            if any(d in r.get("url", "") for d in task.exclude_domains):
                continue

            sources.append(
                Source(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    snippet=r.get("snippet", ""),
                    source_type=SourceType.WEB_SEARCH,
                )
            )

        return sources

    def _score_relevance(self, source: Source, query: str) -> float:
        """Score source relevance to query."""
        score = 0.0
        query_words = set(query.lower().split())
        text = f"{source.title} {source.snippet}".lower()

        # Word overlap
        for word in query_words:
            if word in text and len(word) > 3:
                score += 0.1

        # Source type bonus
        type_bonus = {
            SourceType.KNOWLEDGE_BASE: 0.2,
            SourceType.DATABASE: 0.15,
            SourceType.DOCUMENT: 0.1,
            SourceType.WEB_SEARCH: 0.05,
        }
        score += type_bonus.get(source.source_type, 0)

        return min(score, 1.0)

    def _extract_findings(
        self,
        sources: list[Source],
        task: ResearchTask,
    ) -> list[str]:
        """Extract key findings from sources."""
        if self.llm:
            return self._extract_with_llm(sources, task)

        # Simple extraction without LLM
        findings = []

        for topic, info in self.KNOWLEDGE_BASE.items():
            if topic in task.query.lower():
                findings.extend(info["key_points"][:2])

        if not findings:
            findings = [
                f"Research on '{task.query}' is ongoing",
                "Multiple perspectives exist on this topic",
                "Further investigation recommended",
            ]

        return findings[:5]

    def _extract_with_llm(
        self,
        sources: list[Source],
        task: ResearchTask,
    ) -> list[str]:
        """Extract findings using LLM."""
        source_text = "\n".join([f"- {s.title}: {s.snippet}" for s in sources])

        prompt = f"""Based on these sources about "{task.query}":

{source_text}

List 5 key findings as bullet points."""

        response = self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )

        findings = []
        for line in response.content.split("\n"):
            line = line.strip().lstrip("-•*").strip()
            if line and len(line) > 10:
                findings.append(line)

        return findings[:5]

    def _synthesize(
        self,
        sources: list[Source],
        findings: list[str],
        task: ResearchTask,
    ) -> str:
        """Synthesize a summary from sources and findings."""
        if self.llm:
            prompt = f"""Write a 2-3 paragraph summary about "{task.query}" based on:

Key findings:
{chr(10).join(f"- {f}" for f in findings)}

Make it informative and well-structured."""

            response = self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            return response.content

        # Simple synthesis without LLM
        summary_parts = [
            f"Research on '{task.query}' reveals several important aspects.",
        ]

        for topic, info in self.KNOWLEDGE_BASE.items():
            if topic in task.query.lower():
                summary_parts.append(info["summary"])

        if len(summary_parts) == 1:
            summary_parts.append(
                "This topic requires comprehensive analysis from multiple perspectives. "
                "The gathered sources provide foundational information for further investigation."
            )

        return " ".join(summary_parts)

    def _calculate_confidence(
        self,
        sources: list[Source],
        task: ResearchTask,
    ) -> float:
        """Calculate confidence score for research."""
        if not sources:
            return 0.0

        # Base confidence from number of sources
        source_confidence = min(len(sources) / task.max_sources, 1.0) * 0.5

        # Relevance confidence
        avg_relevance = sum(s.relevance_score for s in sources) / len(sources)
        relevance_confidence = avg_relevance * 0.3

        # Diversity confidence
        source_types = {s.source_type for s in sources}
        diversity_confidence = min(len(source_types) / 3, 1.0) * 0.2

        return source_confidence + relevance_confidence + diversity_confidence

    def compare(
        self,
        queries: list[str],
    ) -> dict[str, ResearchResult]:
        """Compare research on multiple queries."""
        results = {}
        for query in queries:
            results[query] = self.research(query)
        return results


# Example usage
if __name__ == "__main__":
    researcher = ResearchAgent()

    result = researcher.research("What are AI agents?")

    print("# Research Report")
    print(f"Query: {result.query}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"\nSummary:\n{result.summary}")
    print("\nKey Findings:")
    for finding in result.key_findings:
        print(f"  - {finding}")
    print(f"\nSources: {len(result.sources)}")
