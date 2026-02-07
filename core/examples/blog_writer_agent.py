"""
Sample Blog Writer Agent - Generates blog posts from topics.

Demonstrates:
- Multi-step content generation workflow
- State management for drafts
- LLM integration for writing
- Quality revision loop

From ROADMAP: Sample Agents > Blog Writer Agent
"""

from dataclasses import dataclass, field
from typing import Any
from datetime import datetime


@dataclass
class BlogPost:
    """A blog post structure."""

    title: str
    content: str
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    seo_keywords: list[str] = field(default_factory=list)
    word_count: int = 0
    reading_time_min: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "draft"  # draft, review, published

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "tags": self.tags,
            "seo_keywords": self.seo_keywords,
            "word_count": self.word_count,
            "reading_time_min": self.reading_time_min,
            "created_at": self.created_at,
            "status": self.status,
        }

    def to_markdown(self) -> str:
        """Export as markdown."""
        lines = [
            f"# {self.title}",
            "",
            f"*{self.summary}*" if self.summary else "",
            "",
            f"**Tags:** {', '.join(self.tags)}" if self.tags else "",
            f"**Reading time:** {self.reading_time_min} min",
            "",
            "---",
            "",
            self.content,
        ]
        return "\n".join(lines)


@dataclass
class ContentBrief:
    """Brief for content generation."""

    topic: str
    target_audience: str = "general"
    tone: str = "professional"  # professional, casual, academic, friendly
    word_count_target: int = 800
    key_points: list[str] = field(default_factory=list)
    include_sections: list[str] = field(default_factory=list)


class BlogWriterAgent:
    """
    Agent for generating blog posts.

    Usage:
        writer = BlogWriterAgent()

        # Simple generation
        post = writer.generate("Introduction to AI Agents")

        # With brief
        brief = ContentBrief(
            topic="Building AI Agents with Python",
            target_audience="developers",
            tone="technical",
            word_count_target=1200,
            key_points=["Architecture", "Best practices", "Examples"]
        )
        post = writer.generate_from_brief(brief)

        # Export
        markdown = post.to_markdown()
    """

    # Templates for different sections
    INTRO_TEMPLATE = """Write an engaging introduction for a blog post about: {topic}
Target audience: {audience}
Tone: {tone}
Keep it to 2-3 paragraphs."""

    BODY_TEMPLATE = """Write the main body content for a blog post about: {topic}
Key points to cover: {key_points}
Target word count: {word_count}
Tone: {tone}
Use clear headings and subheadings."""

    CONCLUSION_TEMPLATE = """Write a compelling conclusion for the blog post about: {topic}
Summarize key takeaways and include a call to action.
Keep it to 1-2 paragraphs."""

    def __init__(self, llm_provider: Any = None):
        """
        Initialize blog writer.

        Args:
            llm_provider: LLM provider for content generation
        """
        self.llm = llm_provider
        self._drafts: dict[str, BlogPost] = {}

    def generate(
        self,
        topic: str,
        tone: str = "professional",
        word_count: int = 800,
    ) -> BlogPost:
        """Generate a blog post from a topic."""
        brief = ContentBrief(
            topic=topic,
            tone=tone,
            word_count_target=word_count,
        )
        return self.generate_from_brief(brief)

    def generate_from_brief(self, brief: ContentBrief) -> BlogPost:
        """Generate a blog post from a content brief."""
        # Generate title
        title = self._generate_title(brief)

        # Generate sections
        intro = self._generate_section("intro", brief)
        body = self._generate_section("body", brief)
        conclusion = self._generate_section("conclusion", brief)

        # Combine content
        content = f"{intro}\n\n{body}\n\n{conclusion}"

        # Generate metadata
        summary = self._generate_summary(content, brief)
        tags = self._generate_tags(brief)
        keywords = self._generate_keywords(brief)

        # Calculate stats
        word_count = len(content.split())
        reading_time = max(1, word_count // 200)

        post = BlogPost(
            title=title,
            content=content,
            summary=summary,
            tags=tags,
            seo_keywords=keywords,
            word_count=word_count,
            reading_time_min=reading_time,
        )

        # Store draft
        self._drafts[post.title] = post

        return post

    def revise(
        self,
        post: BlogPost,
        feedback: str,
    ) -> BlogPost:
        """Revise a blog post based on feedback."""
        if self.llm:
            prompt = f"""Revise this blog post based on the feedback:

Current content:
{post.content}

Feedback:
{feedback}

Provide the revised content."""

            response = self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )
            post.content = response.content
            post.word_count = len(post.content.split())

        return post

    def generate_outline(self, brief: ContentBrief) -> list[str]:
        """Generate an outline before full content."""
        if self.llm:
            prompt = f"""Create a detailed outline for a blog post about: {brief.topic}
Target audience: {brief.target_audience}
Include main sections and subsections."""

            response = self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            return response.content.split("\n")

        # Fallback outline
        return [
            "1. Introduction",
            "2. Background",
            "3. Main Points",
            "4. Examples",
            "5. Conclusion",
        ]

    def get_drafts(self) -> list[BlogPost]:
        """Get all drafts."""
        return list(self._drafts.values())

    def _generate_title(self, brief: ContentBrief) -> str:
        """Generate a title."""
        if self.llm:
            prompt = f"Generate a catchy blog title for: {brief.topic}"
            response = self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
            )
            return response.content.strip().strip('"')

        return f"A Guide to {brief.topic.title()}"

    def _generate_section(self, section: str, brief: ContentBrief) -> str:
        """Generate a section of the blog post."""
        if not self.llm:
            return self._get_placeholder_content(section, brief)

        templates = {
            "intro": self.INTRO_TEMPLATE,
            "body": self.BODY_TEMPLATE,
            "conclusion": self.CONCLUSION_TEMPLATE,
        }

        template = templates.get(section, self.BODY_TEMPLATE)
        prompt = template.format(
            topic=brief.topic,
            audience=brief.target_audience,
            tone=brief.tone,
            key_points=", ".join(brief.key_points) if brief.key_points else "relevant aspects",
            word_count=brief.word_count_target // 2,
        )

        response = self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )
        return response.content

    def _generate_summary(self, content: str, brief: ContentBrief) -> str:
        """Generate a summary."""
        if self.llm:
            prompt = f"Write a one-sentence summary of this blog post:\n{content[:500]}..."
            response = self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            return response.content.strip()

        return f"An exploration of {brief.topic}"

    def _generate_tags(self, brief: ContentBrief) -> list[str]:
        """Generate tags."""
        # Extract from topic words
        words = brief.topic.lower().split()
        tags = [w for w in words if len(w) > 3][:5]
        return tags

    def _generate_keywords(self, brief: ContentBrief) -> list[str]:
        """Generate SEO keywords."""
        keywords = [brief.topic.lower()]
        keywords.extend(brief.key_points[:3])
        return keywords

    def _get_placeholder_content(self, section: str, brief: ContentBrief) -> str:
        """Get placeholder content when LLM not available."""
        placeholders = {
            "intro": f"Welcome to this comprehensive guide about {brief.topic}. "
            f"In this article, we'll explore everything you need to know.",
            "body": f"## Understanding {brief.topic}\n\n"
            f"Let's dive into the key aspects of {brief.topic}...\n\n"
            f"## Key Points\n\n"
            f"Here are the most important things to consider...",
            "conclusion": f"In conclusion, {brief.topic} is a fascinating subject. "
            f"We hope this guide has been helpful.",
        }
        return placeholders.get(section, "Content placeholder")


# Example usage
if __name__ == "__main__":
    agent = BlogWriterAgent()

    post = agent.generate(
        topic="Getting Started with AI Agents",
        tone="friendly",
        word_count=600,
    )

    print(post.to_markdown())
