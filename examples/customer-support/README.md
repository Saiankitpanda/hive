# Customer Support Agent Example

This example demonstrates a production-grade **Customer Support Agent** built with Aden Hive.

## Features Demonstrated

*   **Goal-Driven Execution**: The agent follows a high-level goal to resolve tickets.
*   **Routing**: Intelligent triage between "Technical", "Billing", and "General" issues.
*   **Tool Use**: Uses mock tools to search documentation and check user status.
*   **Human-in-the-Loop (HITL)**: Pauses execution for a human to review the drafted response before sending.

## Anatomy

*   `agent.json`: Defines the graph steps and the goal.
*   `tools.py`: Python functions decorated with `@tool` that the agent can use.
*   `run.py`: A script to run the agent and handle the human review step interactively.

## How to Run

1.  **Set your API Key**:
    ```bash
    export OPENAI_API_KEY=sk-...
    # OR
    export CEREBRAS_API_KEY=...
    ```

2.  **Run the script**:
    ```bash
    python run.py
    ```

3.  **Interact**:
    The agent will analyze a sample ticket, search the mock knowledge base, draft a response, and then **pause**.
    Follow the prompt to approve the email.

## Graph Logic

1.  **Triage Node**: Analyzes the input ticket.
2.  **Research/Check Node**: Calls appropriate tools based on triage.
3.  **Draft Node**: Generates a response.
4.  **Human Review Node**: Pauses for approval.
5.  **Send Node**: Sends the email if approved.
