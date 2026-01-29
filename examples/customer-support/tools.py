from framework.runner.tool_registry import tool


@tool(description="Search the internal knowledge base for technical solutions.")
def search_knowledge_base(query: str) -> str:
    """Simulates searching a vector database or docs."""
    print(f"   [Mock Tool] Searching KB for: {query}")
    return """
    Found 2 relevant articles:
    1. 'Fixing 500 Errors': Check your API key and ensure it is not expired.
    2. 'Rate Limiting': You may be hitting the 100 req/min limit. Retry with exponential backoff.
    """


@tool(description="Check a user's subscription status and details.")
def check_subscription_status(user_id: str = "unknown") -> dict:
    """Simulates a CRM lookup."""
    print(f"   [Mock Tool] Checking subscription for: {user_id}")
    return {
        "status": "active",
        "plan": "pro",
        "last_payment": "2024-01-01",
        "user_name": "Jane Doe",
    }


@tool(description="Send an email to the customer.")
def send_email(to_address: str, subject: str, body: str) -> str:
    """Simulates sending an email via SMTP/API."""
    print(f"   [Mock Tool] Sending email to {to_address}...")
    print(f"   Subject: {subject}")
    print(f"   Body: {body}")
    return "sent"
