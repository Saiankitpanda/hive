"""
HITL (Human-in-the-Loop) Tool - Pause execution for human approval.

Provides mechanisms for:
- Pausing agent execution for human review
- Requesting approval for actions
- Getting human input on decisions
- Managing approval workflows

Part of the Core Agent Tools from ROADMAP.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from collections.abc import Callable


class ApprovalStatus(Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ApprovalType(Enum):
    """Type of approval being requested."""

    ACTION = "action"  # Approve an action before execution
    DECISION = "decision"  # Choose between options
    CONFIRMATION = "confirmation"  # Simple yes/no
    INPUT = "input"  # Request text input
    REVIEW = "review"  # Review and optionally modify


@dataclass
class ApprovalRequest:
    """A request for human approval."""

    id: str
    request_type: ApprovalType
    title: str
    description: str
    options: list[str] | None = None
    context: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 300.0
    created_at: float = field(default_factory=time.time)
    status: ApprovalStatus = ApprovalStatus.PENDING
    response: str | None = None
    responded_at: float | None = None
    execution_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "request_type": self.request_type.value,
            "title": self.title,
            "description": self.description,
            "options": self.options,
            "context": self.context,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at,
            "created_at_iso": datetime.fromtimestamp(self.created_at).isoformat(),
            "status": self.status.value,
            "response": self.response,
            "responded_at": self.responded_at,
            "execution_id": self.execution_id,
        }


@dataclass
class HITLConfig:
    """Configuration for HITL service."""

    default_timeout: float = 300.0  # 5 minutes
    auto_approve_development: bool = False
    notification_handler: Callable[[ApprovalRequest], None] | None = None


class HITLService:
    """
    Human-in-the-Loop service for managing approval workflows.

    Usage:
        hitl = HITLService()

        # Request approval for an action
        request = await hitl.request_approval(
            title="Delete old files",
            description="Agent wants to delete 50 files older than 30 days",
            request_type=ApprovalType.CONFIRMATION
        )

        # Wait for response
        result = await hitl.wait_for_response(request.id)

        if result.status == ApprovalStatus.APPROVED:
            # Proceed with action
            pass

        # For choices
        request = await hitl.request_choice(
            title="Select deployment target",
            description="Choose where to deploy the application",
            options=["production", "staging", "development"]
        )
    """

    def __init__(self, config: HITLConfig | None = None):
        self.config = config or HITLConfig()
        self._requests: dict[str, ApprovalRequest] = {}
        self._response_events: dict[str, asyncio.Event] = {}
        self._request_counter = 0
        self._handlers: list[Callable[[ApprovalRequest], None]] = []

        if self.config.notification_handler:
            self._handlers.append(self.config.notification_handler)

    # === Request Methods ===

    async def request_approval(
        self,
        title: str,
        description: str,
        request_type: ApprovalType = ApprovalType.CONFIRMATION,
        options: list[str] | None = None,
        context: dict[str, Any] | None = None,
        timeout: float | None = None,
        execution_id: str | None = None,
    ) -> ApprovalRequest:
        """
        Create an approval request.

        Args:
            title: Short title for the request
            description: Detailed description of what needs approval
            request_type: Type of approval
            options: Options for DECISION type
            context: Additional context
            timeout: Timeout in seconds
            execution_id: Associated execution ID

        Returns:
            ApprovalRequest object
        """
        self._request_counter += 1
        request_id = f"hitl-{self._request_counter:06d}"

        request = ApprovalRequest(
            id=request_id,
            request_type=request_type,
            title=title,
            description=description,
            options=options,
            context=context or {},
            timeout_seconds=timeout or self.config.default_timeout,
            execution_id=execution_id,
        )

        self._requests[request_id] = request
        self._response_events[request_id] = asyncio.Event()

        # Notify handlers
        for handler in self._handlers:
            try:
                handler(request)
            except Exception:
                pass

        return request

    async def request_confirmation(
        self,
        title: str,
        description: str,
        context: dict[str, Any] | None = None,
        **kwargs,
    ) -> ApprovalRequest:
        """Request a yes/no confirmation."""
        return await self.request_approval(
            title=title,
            description=description,
            request_type=ApprovalType.CONFIRMATION,
            options=["yes", "no"],
            context=context,
            **kwargs,
        )

    async def request_choice(
        self,
        title: str,
        description: str,
        options: list[str],
        context: dict[str, Any] | None = None,
        **kwargs,
    ) -> ApprovalRequest:
        """Request a choice from multiple options."""
        return await self.request_approval(
            title=title,
            description=description,
            request_type=ApprovalType.DECISION,
            options=options,
            context=context,
            **kwargs,
        )

    async def request_input(
        self,
        title: str,
        description: str,
        context: dict[str, Any] | None = None,
        **kwargs,
    ) -> ApprovalRequest:
        """Request text input from user."""
        return await self.request_approval(
            title=title,
            description=description,
            request_type=ApprovalType.INPUT,
            context=context,
            **kwargs,
        )

    # === Response Methods ===

    async def wait_for_response(
        self,
        request_id: str,
        timeout: float | None = None,
    ) -> ApprovalRequest:
        """
        Wait for a response to an approval request.

        Args:
            request_id: ID of the request to wait for
            timeout: Override timeout

        Returns:
            Updated ApprovalRequest with response
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request {request_id} not found")

        event = self._response_events.get(request_id)
        if not event:
            return request

        wait_timeout = timeout or request.timeout_seconds

        try:
            await asyncio.wait_for(event.wait(), timeout=wait_timeout)
        except asyncio.TimeoutError:
            request.status = ApprovalStatus.TIMEOUT
            request.responded_at = time.time()

        return request

    def respond(
        self,
        request_id: str,
        approved: bool,
        response: str | None = None,
    ) -> ApprovalRequest | None:
        """
        Respond to an approval request.

        Args:
            request_id: ID of the request
            approved: Whether the request is approved
            response: Optional response text

        Returns:
            Updated ApprovalRequest
        """
        request = self._requests.get(request_id)
        if not request:
            return None

        request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        request.response = response
        request.responded_at = time.time()

        # Signal waiting coroutines
        event = self._response_events.get(request_id)
        if event:
            event.set()

        return request

    def approve(self, request_id: str, response: str | None = None) -> ApprovalRequest | None:
        """Approve a request."""
        return self.respond(request_id, approved=True, response=response)

    def reject(self, request_id: str, reason: str | None = None) -> ApprovalRequest | None:
        """Reject a request."""
        return self.respond(request_id, approved=False, response=reason)

    def cancel(self, request_id: str) -> ApprovalRequest | None:
        """Cancel a pending request."""
        request = self._requests.get(request_id)
        if not request:
            return None

        request.status = ApprovalStatus.CANCELLED
        request.responded_at = time.time()

        event = self._response_events.get(request_id)
        if event:
            event.set()

        return request

    # === Query Methods ===

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        """Get a specific request."""
        return self._requests.get(request_id)

    def get_pending(self) -> list[ApprovalRequest]:
        """Get all pending requests."""
        return [r for r in self._requests.values() if r.status == ApprovalStatus.PENDING]

    def get_by_execution(self, execution_id: str) -> list[ApprovalRequest]:
        """Get requests for a specific execution."""
        return [r for r in self._requests.values() if r.execution_id == execution_id]

    # === Handler Methods ===

    def on_request(self, handler: Callable[[ApprovalRequest], None]) -> None:
        """Register a request notification handler."""
        self._handlers.append(handler)


# Tool definition for LLM providers
HITL_TOOL = {
    "name": "request_human_approval",
    "description": "Pause execution and request human approval before proceeding",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Brief title for the approval request",
            },
            "description": {
                "type": "string",
                "description": "Detailed description of what needs approval",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Options to choose from (for choice requests)",
            },
        },
        "required": ["title", "description"],
    },
}


def create_hitl_tool_executor(service: HITLService):
    """Create a tool executor function for the HITL service."""

    async def executor(tool_input: dict[str, Any]) -> str:
        title = tool_input.get("title", "Approval Required")
        description = tool_input.get("description", "")
        options = tool_input.get("options")

        if options:
            request = await service.request_choice(
                title=title,
                description=description,
                options=options,
            )
        else:
            request = await service.request_confirmation(
                title=title,
                description=description,
            )

        # Wait for response
        result = await service.wait_for_response(request.id)

        if result.status == ApprovalStatus.APPROVED:
            return f"✅ Approved: {result.response or 'yes'}"
        elif result.status == ApprovalStatus.REJECTED:
            return f"❌ Rejected: {result.response or 'no'}"
        elif result.status == ApprovalStatus.TIMEOUT:
            return "⏰ Request timed out waiting for human response"
        else:
            return f"Request status: {result.status.value}"

    return executor
