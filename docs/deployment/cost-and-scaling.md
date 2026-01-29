# Cost & Scaling Guide

This guide covers budget control, throttling, model degradation, and scaling patterns for Aden/Hive in real environments.

---

## Why this matters

In production, the two fastest ways to break an agent system are:
1) **unbounded cost** (token spend spikes)
2) **unbounded concurrency** (tool/LLM overload + cascading failure)

A good system enforces predictable spend + predictable load.

---

## Cost control layers (recommended)

### Layer 1 — Hard budgets
Set budgets at:
- team level (monthly ceiling)
- agent level (daily ceiling)
- workflow/run level (per execution cap)

A hard budget should stop or degrade work, not silently continue.

---

### Layer 2 — Throttles / rate limits
Limit:
- LLM requests per minute
- tool calls per minute
- max concurrent runs per agent

This prevents burst traffic from collapsing your stack.

---

### Layer 3 — Automatic model degradation
When you hit cost or latency thresholds:
- switch to a cheaper model
- reduce max tokens
- reduce context size (lean on scoped memory)

Degradation keeps the system alive instead of failing everything.

---

## Scaling patterns

### Pattern A — Vertical first (single node)
If you're early:
- raise worker count gradually
- measure latency and tool saturation
- add caching where safe

---

### Pattern B — Queue-based horizontal scaling
When load becomes spiky or unpredictable:
- enqueue tasks
- scale workers based on queue depth
- isolate worker pools by workload class

---

### Pattern C — Separate “fast path” and “slow path”
Fast path: ticket triage, routing, simple lookups  
Slow path: deep research, multi-tool workflows

Give them separate queues and separate budgets.

---

## What to measure (minimum metrics)

### Cost metrics
- tokens in/out per run
- cost per run (estimate)
- cost per agent per day
- cost spikes (p95/p99)

### Performance metrics
- end-to-end latency (p50/p95/p99)
- tool call latency + failure rate
- LLM latency + failure rate
- queue depth and worker utilization

### Reliability metrics
- retries per run
- percentage of runs needing HITL
- top failure categories (tool/LLM/timeouts)

---

## Practical guardrails (copy-paste policy ideas)

- If **run cost** exceeds threshold → stop and request HITL approval
- If **LLM latency p95** exceeds threshold → degrade model
- If **tool failure rate** spikes → open circuit breaker and fallback
- If **queue depth** grows → autoscale workers up to max replicas

---

## Common scaling mistakes (avoid these)

1) No per-run budget → one bad prompt burns your monthly spend  
2) No timeout on tools → threads hang, concurrency collapses  
3) One worker pool for everything → heavy tasks starve critical tasks  
4) Logging raw prompts with secrets → security incident waiting to happen  

---

## When to add infrastructure
Add infra when you see:
- p95 latency climbing steadily
- frequent tool rate-limit errors
- workers at 80–90% utilization for long periods
- spend spikes caused by retries or long contexts

If you don’t have these symptoms, keep it simple.
