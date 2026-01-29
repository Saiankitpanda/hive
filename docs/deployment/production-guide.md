# Production Deployment Guide

This guide explains how to run Aden/Hive in production with reliability, observability, and safe failure handling.

If you're new, start with a single-machine deployment. If you're operating at scale, use a queue-based worker pool with centralized state and metrics.

---

## Mental model (simple)

Think of Aden like this:

- **Control Plane**: decides what should happen (orchestration, policies, budgets)
- **Worker Plane**: does the work (tools, LLM calls, task execution)
- **State Plane**: remembers safely (shared memory, scoped context, stores)
- **Observability Plane**: shows what's happening (logs, traces, cost, alerts)

A production setup makes these explicit so the system stays stable under load.

---

## Reference architecture

### Small / Single Node (best first production step)

Use this if you have one team, low-to-medium traffic, and want fast deployment.

**Components on one machine:**
- Hive runtime + API server
- Worker processes (local)
- Persistent store (for runs, metadata, memory)
- Observability exporter (logs + metrics)

Pros: simplest, low ops cost  
Cons: limited horizontal scaling

---

### Scalable / Multi-Node (recommended for real scale)

Use this when you need concurrency, isolation, and predictable performance.

```

      ┌───────────────────────────┐
      │        API Gateway         │
      └─────────────┬─────────────┘
                    │
      ┌─────────────▼─────────────┐
      │   Control Plane / Orchestr │
      │  - policies, budgets       │
      │  - scheduling, retries     │
      └─────────────┬─────────────┘
                    │ enqueues work
            ┌───────▼────────┐
            │  Work Queue     │
            └───────┬────────┘
                    │
 ┌──────────────────▼──────────────────┐
 │             Worker Pool              │
 │  - tool calls (MCP tools)            │
 │  - LLM calls via provider gateway    │
 │  - sandboxing / timeouts             │
 └───────────────┬─────────────────────┘
                 │ reads/writes
       ┌─────────▼─────────┐
       │  State Store        │
       │  - runs, memory     │
       │  - artifacts        │
       └─────────┬─────────┘
                 │ emits
       ┌─────────▼─────────┐
       │ Observability       │
       │ - logs, metrics     │
       │ - cost + latency    │
       └─────────────────────┘
```

Pros: scales horizontally, isolates failures  
Cons: more ops work

---

## Deployment checklist (must-do)

### 1) Secrets & credentials
- Never hardcode API keys in code or docs.
- Use environment variables or a secrets manager.
- Rotate keys and restrict scopes where possible.

**Minimum practice:**
- one key per environment (dev/staging/prod)
- separate keys for tool services (DB, email, CRM, etc.)

---

### 2) Reliability defaults
Production failures are normal. What matters is controlled behavior:

**Use:**
- timeouts on every external call (LLM + tools)
- retry with backoff (only on safe idempotent calls)
- circuit breakers for flaky tools
- hard budgets (team / agent / workflow)

---

### 3) Observability (non-negotiable)
You need to answer:
- what did the agent do?
- why did it do it?
- how much did it cost?
- what failed and where?

**Minimum signals to emit:**
- request_id / run_id per execution
- tool name + duration + success/failure
- LLM model + tokens + latency + cost estimate
- node-to-node transitions (graph steps)

---

### 4) Data safety
Agents often touch sensitive info.

**Minimum practices:**
- redact secrets from logs
- avoid storing raw prompt/response unless required
- encrypt persistent stores at rest
- enforce least privilege on tool integrations

---

## Scaling strategy (how to grow safely)

### Start with concurrency limits
Before adding machines, cap concurrency:
- max parallel runs per agent
- max tool calls per minute
- max LLM spend per hour

### Then scale workers horizontally
When CPU/time is the bottleneck:
- add more worker replicas
- keep control plane stable
- prefer a queue (so spikes don’t crash you)

### Separate noisy workloads
If one agent type is heavy (e.g., long research runs):
- separate worker pools by queue/topic
- isolate rate limits and budgets per pool

---

## Failure mode playbook

### Tool outage (e.g., CRM is down)
Expected behavior:
- fail fast with clear classification: TOOL_UNAVAILABLE
- retry only if safe and within budget
- fallback response: "need human approval" (HITL)

### LLM slowdown or cost spike
Expected behavior:
- degrade model automatically (cheaper/faster)
- reduce context size (use scoped memory)
- enforce budget ceilings

### Worker crash mid-run
Expected behavior:
- run is recoverable
- steps are checkpointed
- retries continue from safe boundary

---

## Human-in-the-loop (HITL) in production

Use HITL when:
- money moves
- data deletes/overwrites
- legal/compliance risk exists
- confidence is low or tool results conflict

Operationally:
- define timeout + escalation owner
- persist pending actions
- log "why paused" and "what is needed"

---

## Recommended next docs
- Cost & Scaling: `docs/deployment/cost-and-scaling.md`
- Security hardening: least privilege + audit logging
- Runbooks: incident response + rollback procedure
