"""
Sample SDR (Sales Development Rep) Agent - Automates sales outreach.

Demonstrates:
- Lead qualification workflow
- Personalized email generation
- Follow-up scheduling
- CRM integration patterns

From ROADMAP: Sample Agents > SDR Agent
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class LeadStatus(Enum):
    """Lead status in the pipeline."""

    NEW = "new"
    QUALIFIED = "qualified"
    CONTACTED = "contacted"
    RESPONDED = "responded"
    MEETING_SCHEDULED = "meeting_scheduled"
    NOT_INTERESTED = "not_interested"
    NURTURING = "nurturing"


class EmailType(Enum):
    """Types of outreach emails."""

    COLD_OUTREACH = "cold_outreach"
    FOLLOW_UP = "follow_up"
    VALUE_PROPOSITION = "value_proposition"
    CASE_STUDY = "case_study"
    MEETING_REQUEST = "meeting_request"


@dataclass
class Lead:
    """A sales lead."""

    id: str
    name: str
    email: str
    company: str
    title: str = ""
    industry: str = ""
    company_size: str = ""
    status: LeadStatus = LeadStatus.NEW
    score: int = 0
    notes: list[str] = field(default_factory=list)
    last_contacted: str | None = None
    next_follow_up: str | None = None
    touchpoints: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "company": self.company,
            "title": self.title,
            "industry": self.industry,
            "company_size": self.company_size,
            "status": self.status.value,
            "score": self.score,
            "notes": self.notes,
            "last_contacted": self.last_contacted,
            "next_follow_up": self.next_follow_up,
            "touchpoints": self.touchpoints,
        }


@dataclass
class EmailDraft:
    """A draft email."""

    to: str
    subject: str
    body: str
    email_type: EmailType
    lead_id: str
    scheduled_send: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "to": self.to,
            "subject": self.subject,
            "body": self.body,
            "email_type": self.email_type.value,
            "lead_id": self.lead_id,
            "scheduled_send": self.scheduled_send,
        }


@dataclass
class OutreachSequence:
    """A sequence of outreach emails."""

    name: str
    emails: list[dict[str, Any]]
    delay_days: list[int]

    @classmethod
    def default_sequence(cls) -> "OutreachSequence":
        """Get default outreach sequence."""
        return cls(
            name="default",
            emails=[
                {"type": EmailType.COLD_OUTREACH, "template": "initial"},
                {"type": EmailType.FOLLOW_UP, "template": "follow_up_1"},
                {"type": EmailType.VALUE_PROPOSITION, "template": "value"},
                {"type": EmailType.CASE_STUDY, "template": "case_study"},
                {"type": EmailType.MEETING_REQUEST, "template": "meeting"},
            ],
            delay_days=[0, 3, 7, 14, 21],
        )


class SDRAgent:
    """
    Sales Development Representative Agent.

    Usage:
        sdr = SDRAgent(product_name="Hive AI Platform")

        # Add leads
        lead = Lead(
            id="lead-001",
            name="John Smith",
            email="john@example.com",
            company="Acme Corp",
            title="VP of Engineering",
            industry="Technology"
        )
        sdr.add_lead(lead)

        # Qualify leads
        qualified = sdr.qualify_leads()

        # Generate personalized emails
        email = sdr.generate_email(lead, EmailType.COLD_OUTREACH)

        # Process follow-ups
        follow_ups = sdr.get_due_follow_ups()
    """

    # Email templates
    TEMPLATES = {
        EmailType.COLD_OUTREACH: """Hi {first_name},

I noticed that {company} is {observation}. Many {industry} companies \
like yours are looking for ways to {pain_point}.

{product_name} helps teams like yours {value_proposition}.

Would you be open to a quick 15-minute call this week to explore if \
this could be valuable for {company}?

Best regards,
{sender_name}""",
        EmailType.FOLLOW_UP: """Hi {first_name},

I wanted to follow up on my previous email. I understand you're busy, so I'll keep this brief.

{product_name} has helped companies similar to {company} achieve {result}.

Would it make sense to schedule a brief call?

Best,
{sender_name}""",
        EmailType.VALUE_PROPOSITION: """Hi {first_name},

I've been thinking about {company}'s goals in {industry}, and I believe \
{product_name} could help you:

• {benefit_1}
• {benefit_2}
• {benefit_3}

Our customers typically see {metric} improvement within {timeframe}.

Let me know if you'd like to learn more.

Best,
{sender_name}""",
        EmailType.MEETING_REQUEST: """Hi {first_name},

I'd love to show you how {product_name} works in practice. I have availability:

• {slot_1}
• {slot_2}
• {slot_3}

Would any of these work for a quick demo?

Thanks,
{sender_name}""",
    }

    def __init__(
        self,
        product_name: str = "Our Product",
        sender_name: str = "Sales Team",
        llm_provider: Any = None,
    ):
        """
        Initialize SDR agent.

        Args:
            product_name: Name of the product being sold
            sender_name: Name of the sender
            llm_provider: LLM provider for personalization
        """
        self.product_name = product_name
        self.sender_name = sender_name
        self.llm = llm_provider
        self._leads: dict[str, Lead] = {}
        self._sequences: dict[str, OutreachSequence] = {
            "default": OutreachSequence.default_sequence()
        }

    # === Lead Management ===

    def add_lead(self, lead: Lead) -> None:
        """Add a lead."""
        self._leads[lead.id] = lead

    def get_lead(self, lead_id: str) -> Lead | None:
        """Get a lead by ID."""
        return self._leads.get(lead_id)

    def get_leads_by_status(self, status: LeadStatus) -> list[Lead]:
        """Get leads by status."""
        return [lead for lead in self._leads.values() if lead.status == status]

    def update_lead_status(self, lead_id: str, status: LeadStatus) -> bool:
        """Update lead status."""
        if lead_id in self._leads:
            self._leads[lead_id].status = status
            return True
        return False

    # === Lead Qualification ===

    def qualify_leads(self) -> list[Lead]:
        """Qualify new leads based on scoring."""
        qualified = []

        for lead in self.get_leads_by_status(LeadStatus.NEW):
            score = self._calculate_lead_score(lead)
            lead.score = score

            if score >= 50:
                lead.status = LeadStatus.QUALIFIED
                qualified.append(lead)

        return qualified

    def _calculate_lead_score(self, lead: Lead) -> int:
        """Calculate lead qualification score."""
        score = 0

        # Title scoring
        title_keywords = {
            "vp": 20,
            "director": 15,
            "manager": 10,
            "head": 15,
            "chief": 25,
            "cto": 25,
            "ceo": 20,
            "founder": 20,
        }
        title_lower = lead.title.lower()
        for keyword, points in title_keywords.items():
            if keyword in title_lower:
                score += points
                break

        # Company size scoring
        size_scores = {
            "enterprise": 30,
            "large": 25,
            "medium": 20,
            "small": 10,
            "startup": 15,
        }
        if lead.company_size.lower() in size_scores:
            score += size_scores[lead.company_size.lower()]

        # Industry scoring (customize based on ICP)
        target_industries = ["technology", "finance", "healthcare", "saas"]
        if lead.industry.lower() in target_industries:
            score += 20

        return min(score, 100)

    # === Email Generation ===

    def generate_email(
        self,
        lead: Lead,
        email_type: EmailType,
        custom_context: dict[str, str] | None = None,
    ) -> EmailDraft:
        """Generate a personalized email."""
        context = self._build_email_context(lead, custom_context)

        if self.llm:
            body = self._generate_with_llm(lead, email_type, context)
        else:
            body = self._generate_from_template(email_type, context)

        subject = self._generate_subject(lead, email_type)

        return EmailDraft(
            to=lead.email,
            subject=subject,
            body=body,
            email_type=email_type,
            lead_id=lead.id,
        )

    def _build_email_context(
        self,
        lead: Lead,
        custom: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Build context for email templates."""
        first_name = lead.name.split()[0] if lead.name else "there"

        context = {
            "first_name": first_name,
            "company": lead.company,
            "title": lead.title,
            "industry": lead.industry or "your industry",
            "product_name": self.product_name,
            "sender_name": self.sender_name,
            "observation": f"growing quickly in the {lead.industry} space",
            "pain_point": "improve efficiency and reduce costs",
            "value_proposition": "automate workflows and save time",
            "result": "30% improvement in productivity",
            "benefit_1": "Reduce manual work by 50%",
            "benefit_2": "Get real-time insights",
            "benefit_3": "Scale operations efficiently",
            "metric": "2x",
            "timeframe": "30 days",
            "slot_1": "Tuesday at 2pm",
            "slot_2": "Wednesday at 10am",
            "slot_3": "Thursday at 3pm",
        }

        if custom:
            context.update(custom)

        return context

    def _generate_from_template(
        self,
        email_type: EmailType,
        context: dict[str, str],
    ) -> str:
        """Generate email from template."""
        template = self.TEMPLATES.get(email_type, self.TEMPLATES[EmailType.COLD_OUTREACH])
        return template.format(**context)

    def _generate_with_llm(
        self,
        lead: Lead,
        email_type: EmailType,
        context: dict[str, str],
    ) -> str:
        """Generate personalized email using LLM."""
        prompt = f"""Write a {email_type.value} sales email for:
- Lead: {lead.name}, {lead.title} at {lead.company}
- Industry: {lead.industry}
- Product: {self.product_name}

Make it personalized, professional, and concise. No more than 150 words."""

        response = self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return response.content

    def _generate_subject(self, lead: Lead, email_type: EmailType) -> str:
        """Generate email subject."""
        subjects = {
            EmailType.COLD_OUTREACH: f"Quick question about {lead.company}",
            EmailType.FOLLOW_UP: f"Following up - {lead.company}",
            EmailType.VALUE_PROPOSITION: f"Idea for {lead.company}",
            EmailType.CASE_STUDY: f"How companies like {lead.company} are succeeding",
            EmailType.MEETING_REQUEST: f"15 min call, {lead.name.split()[0]}?",
        }
        return subjects.get(email_type, f"Re: {lead.company}")

    # === Follow-up Management ===

    def schedule_follow_up(
        self,
        lead_id: str,
        days_from_now: int = 3,
    ) -> str | None:
        """Schedule a follow-up for a lead."""
        lead = self._leads.get(lead_id)
        if not lead:
            return None

        follow_up_date = datetime.now() + timedelta(days=days_from_now)
        lead.next_follow_up = follow_up_date.isoformat()
        lead.last_contacted = datetime.now().isoformat()
        lead.touchpoints += 1

        return lead.next_follow_up

    def get_due_follow_ups(self) -> list[Lead]:
        """Get leads with follow-ups due today or earlier."""
        now = datetime.now()
        due = []

        for lead in self._leads.values():
            if lead.next_follow_up:
                follow_up = datetime.fromisoformat(lead.next_follow_up)
                if follow_up <= now:
                    due.append(lead)

        return due

    def get_pipeline_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        stats = {status.value: 0 for status in LeadStatus}

        for lead in self._leads.values():
            stats[lead.status.value] += 1

        total = len(self._leads)
        qualified = stats[LeadStatus.QUALIFIED.value]

        return {
            "total_leads": total,
            "by_status": stats,
            "qualification_rate": round(qualified / total * 100, 1) if total else 0,
            "avg_score": sum(lead.score for lead in self._leads.values()) / total if total else 0,
        }


# Example usage
if __name__ == "__main__":
    sdr = SDRAgent(
        product_name="Hive AI Platform",
        sender_name="Alex from Hive",
    )

    # Add sample leads
    leads = [
        Lead(
            id="lead-001",
            name="Sarah Chen",
            email="sarah@techcorp.com",
            company="TechCorp",
            title="VP of Engineering",
            industry="Technology",
            company_size="Medium",
        ),
        Lead(
            id="lead-002",
            name="Michael Johnson",
            email="michael@financeplus.com",
            company="Finance Plus",
            title="CTO",
            industry="Finance",
            company_size="Large",
        ),
    ]

    for lead in leads:
        sdr.add_lead(lead)

    # Qualify leads
    qualified = sdr.qualify_leads()
    print(f"Qualified {len(qualified)} leads")

    # Generate email for first qualified lead
    if qualified:
        email = sdr.generate_email(qualified[0], EmailType.COLD_OUTREACH)
        print(f"\nEmail to: {email.to}")
        print(f"Subject: {email.subject}")
        print(f"Body:\n{email.body}")
