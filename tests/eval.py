#!/usr/bin/env python3
"""
scionic rapid eval — exercises every feature with real LLM calls.

No mocks. Every test hits OpenRouter. Tests are ordered by feature complexity.
Run: python tests/eval.py
"""

import asyncio
import json
import os
import sys
import time

# ── API key ──────────────────────────────────────────────────────────

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    API_KEY = line.split("=", 1)[1]
                    break
if not API_KEY:
    print("FATAL: No OPENROUTER_API_KEY found.")
    sys.exit(1)

from scionic import (
    Conductor, IRQPriority, IRQType, PathPolicy, TrustDomain,
    TriageRouter, FlowController, CircuitBreaker, CircuitState,
)
from scionic.adapters.llm import LLMNodeHandler
from scionic.node import NodeHandler
from scionic.types import Hop, HopStatus, Task

MODEL = "anthropic/claude-haiku-4.5"
BASE = "https://openrouter.ai/api/v1"
PASSED = 0
FAILED = 0
TOTAL_START = time.time()


def report(name: str, passed: bool, detail: str = ""):
    global PASSED, FAILED
    if passed:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}: {detail}")


def llm(caps: list[str], prompt: str, max_tokens: int = 64, temp: float = 0.0) -> LLMNodeHandler:
    """Shorthand to create an LLM node handler."""
    return LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        node_capabilities=caps,
        system_prompt=prompt,
        max_tokens=max_tokens,
        temperature=temp,
    )


# ── Failing handler (for reroute/retry tests) ───────────────────────

class FailOnceHandler:
    """Fails the first call, succeeds the second. For testing reroute/retry."""
    def __init__(self, caps: list[str], output: str = "recovered"):
        self._caps = caps
        self._output = output
        self._calls = 0

    def capabilities(self) -> list[str]:
        return self._caps

    async def process(self, task: Task, hop: Hop) -> str:
        self._calls += 1
        if self._calls == 1:
            raise RuntimeError("Simulated failure")
        return self._output


class AlwaysFailHandler:
    """Always fails. For testing that reroute finds an alternate."""
    def __init__(self, caps: list[str]):
        self._caps = caps

    def capabilities(self) -> list[str]:
        return self._caps

    async def process(self, task: Task, hop: Hop) -> str:
        raise RuntimeError("This node always fails")


# ═══════════════════════════════════════════════════════════════════
# EVAL 1: Basic forwarding + hop signing
# ═══════════════════════════════════════════════════════════════════

async def eval_basic_forwarding():
    """Single node, real LLM call. Verify hop signing and signature."""
    c = Conductor()
    c.add_node("oracle", llm(["answer"], "Reply with ONLY the number 42. Nothing else.", max_tokens=16))

    task = c.create_task(payload="What is the answer?", path=["oracle"])
    result = await c.execute(task)

    report("task completes", result.is_complete)
    report("one hop", len(result.hops) == 1, f"got {len(result.hops)}")

    hop = result.hops[0]
    report("hop signed", hop.signature != "")
    report("signature valid", hop.verify(c._nodes["oracle"].secret))
    report("has output", "42" in str(result.context.get("oracle", "")),
           f"got: {result.context.get('oracle', '')!r}")
    report("timing recorded", hop.duration_ms > 0)
    report("tokens tracked", hop.tokens_used > 0, f"tokens={hop.tokens_used}")


# ═══════════════════════════════════════════════════════════════════
# EVAL 2: Context accumulation across hops
# ═══════════════════════════════════════════════════════════════════

async def eval_context_accumulation():
    """Two LLM nodes in sequence. Second must use first's output."""
    c = Conductor()
    c.add_node("namer", llm(["generate"],
        "Generate exactly 3 animal names, comma-separated. Nothing else.",
        max_tokens=32, temp=0.9))
    c.add_node("counter", llm(["count"],
        "You receive animal names from a previous step. Count them. Reply with ONLY the number.",
        max_tokens=8))

    task = c.create_task(payload="Name some animals.", path=["namer", "counter"])
    result = await c.execute(task)

    report("pipeline completes", result.is_complete)
    report("two hops", len(result.hops) == 2, f"got {len(result.hops)}")
    report("both signed", all(h.signature != "" for h in result.hops))
    report("chain verifies",
           all(h.verify(c._nodes[h.node_id].secret) for h in result.hops))

    namer_out = str(result.context.get("namer", ""))
    counter_out = str(result.context.get("counter", ""))
    report("namer produced output", len(namer_out) > 0, f"got: {namer_out!r}")
    report("counter saw context", len(counter_out) > 0, f"got: {counter_out!r}")
    report("counter found 3", "3" in counter_out, f"got: {counter_out!r}")


# ═══════════════════════════════════════════════════════════════════
# EVAL 3: Auto path selection by capability + policy
# ═══════════════════════════════════════════════════════════════════

async def eval_path_selection():
    """Register nodes with capabilities. Selector picks cheapest path."""
    c = Conductor()
    c.add_node("cheap", llm(["summarize"], "Summarize in one sentence."),
               cost_per_call=0.001)
    c.add_node("expensive", llm(["summarize"], "Summarize in detail."),
               cost_per_call=0.10)

    task = c.create_task(
        payload="Explain gravity.",
        required_capabilities=["summarize"],
        policy=PathPolicy(prefer_low_cost=True),
    )
    report("selected cheap", task.path == ["cheap"], f"got: {task.path}")

    result = await c.execute(task)
    report("executed successfully", result.is_complete)
    report("has LLM output", len(str(result.context.get("cheap", ""))) > 0)


# ═══════════════════════════════════════════════════════════════════
# EVAL 4: Multi-path parallel execution
# ═══════════════════════════════════════════════════════════════════

async def eval_multipath():
    """Same question, two paths. Both run in parallel with real LLM calls."""
    c = Conductor()
    c.add_node("path_a", llm(["answer"], "You are optimistic. One sentence.", temp=0.7))
    c.add_node("path_b", llm(["answer"], "You are pessimistic. One sentence.", temp=0.7))

    task = Task(payload="Will AI help humanity?")
    results = await c.forwarder.forward_multipath(task, [["path_a"], ["path_b"]])

    report("two results", len(results) == 2, f"got {len(results)}")
    for i, r in enumerate(results):
        node = r.path[0]
        out = str(r.context.get(node, ""))
        report(f"path {i+1} complete", r.is_complete)
        report(f"path {i+1} has output", len(out) > 0, f"got: {out!r}")


# ═══════════════════════════════════════════════════════════════════
# EVAL 5: IRQ interrupt propagation
# ═══════════════════════════════════════════════════════════════════

async def eval_irq():
    """Fire IRQ between real LLM hops. Verify delivery and logging."""
    c = Conductor()
    c.add_node("step1", llm(["research"], "Summarize in one sentence."))
    c.add_node("step2", llm(["write"], "Write a one-sentence conclusion."))

    task = c.create_task(payload="SCION networking", path=["step1", "step2"])

    # Execute first hop manually
    node1 = c._nodes["step1"]
    task = await node1.execute(task)
    report("hop 1 done", task.hops[0].status == HopStatus.COMPLETE)

    # Fire IRQ between hops
    irq_count = await c.fire_irq(
        source="step1",
        irq_type=IRQType.CONTEXT_UPDATE,
        payload="Additional: SCION is in production at SIX Exchange",
        reason="Late-breaking context",
        task=task,
        priority=IRQPriority.HIGH,
    )
    report("IRQ delivered", irq_count > 0, f"delivered to {irq_count}")
    report("IRQ logged", len(c._irq_log) > 0)
    report("IRQ type correct", c._irq_log[-1].irq_type == IRQType.CONTEXT_UPDATE)

    # Continue execution
    node2 = c._nodes["step2"]
    task = await node2.execute(task)
    report("hop 2 done", task.hops[1].status == HopStatus.COMPLETE)
    report("task complete", task.is_complete)


# ═══════════════════════════════════════════════════════════════════
# EVAL 6: IRQ masking
# ═══════════════════════════════════════════════════════════════════

async def eval_irq_masking():
    """Masked IRQs should not be delivered to nodes (but conductor always gets them)."""
    c = Conductor()
    c.add_node("worker", llm(["work"], "Say OK."))

    task = c.create_task(payload="test", path=["worker"])

    # Mask LOW priority for this task
    c.irq_bus.mask(task.id, IRQPriority.LOW)

    count = await c.fire_irq(
        source="external",
        irq_type=IRQType.CONTEXT_UPDATE,
        reason="Low priority noise",
        task=task,
        priority=IRQPriority.LOW,
    )
    # Conductor always receives (global handler), but worker is masked
    report("masked IRQ: conductor got it", len(c._irq_log) > 0)
    report("masked IRQ: only conductor", count == 1, f"delivered to {count}")

    # HIGH priority should always deliver
    count_high = await c.fire_irq(
        source="external",
        irq_type=IRQType.CONTEXT_UPDATE,
        reason="High priority alert",
        task=task,
        priority=IRQPriority.HIGH,
    )
    report("unmasked HIGH delivered", count_high > 0, f"delivered to {count_high}")


# ═══════════════════════════════════════════════════════════════════
# EVAL 7: Peer context injection
# ═══════════════════════════════════════════════════════════════════

async def eval_peer_injection():
    """Send peer message before execution. Verify it's injected into task context."""
    c = Conductor()
    c.add_node("sender", llm(["research"], "Say OK."))
    c.add_node("receiver", llm(["write"],
        "Check context for peer messages. If you see any, mention the sender's name "
        "and what they said. Otherwise say 'no peers'.",
        max_tokens=128))

    c.add_peer_link("sender", "receiver")

    # Send peer message BEFORE the receiver processes
    await c.send_peer_message(
        source="sender",
        target="receiver",
        payload="Hey receiver, the answer is definitely 42",
        message_type="hint",
    )

    task = c.create_task(payload="What do you know?", path=["receiver"])
    result = await c.execute(task)

    report("task complete", result.is_complete)
    # Check that peer context was injected
    ctx_key = "_peer_messages_for_receiver"
    report("peer context injected", ctx_key in result.context,
           f"keys: {list(result.context.keys())}")

    if ctx_key in result.context:
        peers = result.context[ctx_key]
        report("peer message from sender", any(p["from"] == "sender" for p in peers),
               f"peers: {peers}")
        report("peer payload present", any("42" in str(p["payload"]) for p in peers),
               f"peers: {peers}")


# ═══════════════════════════════════════════════════════════════════
# EVAL 8: Trust domain enforcement
# ═══════════════════════════════════════════════════════════════════

async def eval_trust_domains():
    """Cross-domain forwarding should fail without peering. Succeed with it."""
    c = Conductor(verify_signatures=False)  # Focus on trust domain logic

    c.add_trust_domain("internal", "Internal", allowed_peers=[])
    c.add_trust_domain("external", "External", allowed_peers=[])

    c.add_node("secure_node", llm(["process"], "Say 'processed'."),
               trust_domain="internal")
    c.add_node("untrusted_node", llm(["process"], "Say 'processed'."),
               trust_domain="external")

    # Cross-domain path should fail
    task = c.create_task(payload="test", path=["secure_node", "untrusted_node"])
    result = await c.execute(task)

    has_domain_failure = any(
        "Trust domain" in (h.error or "") for h in result.hops
    )
    report("cross-domain blocked", has_domain_failure,
           f"errors: {[h.error for h in result.hops if h.error]}")

    # Now add peering and retry
    c2 = Conductor(verify_signatures=False)
    c2.add_trust_domain("internal", "Internal", allowed_peers=["external"])
    c2.add_trust_domain("external", "External", allowed_peers=["internal"])

    c2.add_node("secure_node", llm(["process"], "Say 'step1'."),
                trust_domain="internal")
    c2.add_node("untrusted_node", llm(["process"], "Say 'step2'."),
                trust_domain="external")

    task2 = c2.create_task(payload="test", path=["secure_node", "untrusted_node"])
    result2 = await c2.execute(task2)

    report("peered domains pass", result2.is_complete,
           f"hops: {[(h.node_id, h.status.value) for h in result2.hops]}")
    report("both hops succeeded",
           all(h.status == HopStatus.COMPLETE for h in result2.hops),
           f"statuses: {[h.status.value for h in result2.hops]}")


# ═══════════════════════════════════════════════════════════════════
# EVAL 9: Hop signature verification in forwarder
# ═══════════════════════════════════════════════════════════════════

async def eval_hop_verification():
    """Tampered signatures should halt forwarding."""
    c = Conductor(verify_signatures=True)
    c.add_node("node_a", llm(["step"], "Say 'hello'."))
    c.add_node("node_b", llm(["step"], "Say 'world'."))

    task = c.create_task(payload="test", path=["node_a", "node_b"])

    # Execute first hop
    node_a = c._nodes["node_a"]
    task = await node_a.execute(task)
    report("hop 1 ok", task.hops[0].status == HopStatus.COMPLETE)

    # Tamper with the signature
    task.hops[0].signature = "tampered_signature_abc123"

    # Forward should fail at verification
    result = await c.forwarder.forward(task)
    has_sig_failure = any(
        "signature" in (h.error or "").lower() for h in result.hops
    )
    report("tampered sig detected", has_sig_failure,
           f"errors: {[h.error for h in result.hops if h.error]}")


# ═══════════════════════════════════════════════════════════════════
# EVAL 10: Reroute on failure
# ═══════════════════════════════════════════════════════════════════

async def eval_reroute():
    """When a node fails, conductor should reroute to an alternate with same capability."""
    c = Conductor(auto_reroute=True, verify_signatures=False)

    c.add_node("broken", AlwaysFailHandler(["compute"]))
    c.add_node("backup", llm(["compute"], "Reply with 'backup worked'. Nothing else.", max_tokens=16))

    task = c.create_task(payload="test", path=["broken"])
    result = await c.execute(task)

    report("reroute happened", result.is_complete,
           f"hops: {[(h.node_id, h.status.value) for h in result.hops]}")

    has_retried = any(h.status == HopStatus.RETRIED for h in result.hops)
    report("original marked retried", has_retried,
           f"statuses: {[(h.node_id, h.status.value) for h in result.hops]}")

    backup_ran = any(h.node_id == "backup" and h.status == HopStatus.COMPLETE for h in result.hops)
    report("backup node executed", backup_ran,
           f"hops: {[(h.node_id, h.status.value) for h in result.hops]}")

    output = str(result.context.get("backup", ""))
    report("backup produced output", "backup" in output.lower(), f"got: {output!r}")


# ═══════════════════════════════════════════════════════════════════
# EVAL 11: IRQ-based retry with feedback
# ═══════════════════════════════════════════════════════════════════

async def eval_irq_retry():
    """Execute, request retry with feedback, re-execute with feedback injected."""
    c = Conductor(verify_signatures=False)

    c.add_node("writer", llm(["write"],
        "Write about the topic. If you see feedback in the context "
        "(look for _feedback_for_writer), incorporate it. "
        "Mention the feedback explicitly in your response.",
        max_tokens=128, temp=0.3))

    task = c.create_task(payload="Write about dogs.", path=["writer"], max_retries=1)
    result = await c.execute(task)

    report("first pass complete", result.is_complete)
    first_output = str(result.context.get("writer", ""))
    report("first output exists", len(first_output) > 0)

    # Request retry with feedback
    await c.request_retry(
        task=result,
        source="reviewer",
        target="writer",
        feedback="Mention golden retrievers specifically.",
    )

    result = await c.execute_with_retry(result)

    report("retry executed", result.retry_count > 0 or len(result.hops) > 1,
           f"hops: {len(result.hops)}, retries: {result.retry_count}")

    # Check feedback was injected
    feedback_key = "_feedback_for_writer"
    report("feedback injected", feedback_key in result.context,
           f"keys: {[k for k in result.context if k.startswith('_')]}")


# ═══════════════════════════════════════════════════════════════════
# EVAL 12: Full pipeline — triage → plan → execute → review
# ═══════════════════════════════════════════════════════════════════

async def eval_full_pipeline():
    """End-to-end: 4 LLM nodes, peer messaging, full traceroute."""
    c = Conductor(verify_signatures=True)

    c.add_node("triager", llm(["triage"],
        'Classify the task. Respond with JSON: {"priority": "high", "type": "design"}. '
        'ONLY the JSON, no markdown.',
        max_tokens=32))
    c.add_node("planner", llm(["plan"],
        'Break the task into 2 subtasks. Respond with JSON: {"subtasks": ["s1", "s2"]}. '
        'ONLY the JSON.',
        max_tokens=64))
    c.add_node("executor", llm(["execute"],
        "Execute the task based on the plan from previous steps. 2-3 sentences.",
        max_tokens=128))
    c.add_node("reviewer", llm(["review"],
        'Review the execution. Respond with JSON: {"approved": true, "quality": "good"}. '
        'ONLY the JSON.',
        max_tokens=32))

    c.add_peer_link("planner", "executor")

    # Send peer hint before execution
    await c.send_peer_message(
        source="planner", target="executor",
        payload="Focus on practical implementation details.",
        message_type="hint",
    )

    task = c.create_task(
        payload="Design a caching layer for a web API",
        path=["triager", "planner", "executor", "reviewer"],
    )
    result = await c.execute(task)

    report("4-hop pipeline complete", result.is_complete,
           f"hops: {len(result.hops)}")
    report("all hops succeeded",
           all(h.status == HopStatus.COMPLETE for h in result.hops),
           f"statuses: {[(h.node_id, h.status.value) for h in result.hops]}")
    report("all hops signed",
           all(h.signature != "" for h in result.hops))
    report("full chain verifies",
           all(h.verify(c._nodes[h.node_id].secret) for h in result.hops))

    # Verify each node produced output
    for node_id in ["triager", "planner", "executor", "reviewer"]:
        out = str(result.context.get(node_id, ""))
        report(f"{node_id} has output", len(out) > 0, f"got: {out[:50]!r}")

    # Verify peer context was injected
    peer_key = "_peer_messages_for_executor"
    report("peer hint reached executor", peer_key in result.context,
           f"keys: {[k for k in result.context if '_peer' in k]}")

    # Print the traceroute
    print(f"\n  {result.traceroute()}")


# ═══════════════════════════════════════════════════════════════════
# EVAL 13: Traceroute readability
# ═══════════════════════════════════════════════════════════════════

async def eval_traceroute():
    """Verify traceroute output contains all expected markers."""
    c = Conductor()
    c.add_node("a", llm(["step"], "Say 'done'."))
    c.add_node("b", llm(["step"], "Say 'done'."))

    task = c.create_task(payload="test", path=["a", "b"])
    result = await c.execute(task)

    trace = result.traceroute()
    report("has task ID", result.id[:8] in trace)
    report("has [+] markers", trace.count("[+]") == 2, f"found {trace.count('[+]')}")
    report("has ms timing", "ms" in trace)
    report("has tokens", "tokens" in trace)
    report("has node names", "a" in trace and "b" in trace)


# ═══════════════════════════════════════════════════════════════════
# EVAL 14: Hermes agent node
# ═══════════════════════════════════════════════════════════════════

async def eval_hermes():
    """Test the Hermes adapter with a real hermes chat invocation."""
    from scionic.adapters.hermes import HermesNodeHandler

    c = Conductor(verify_signatures=True)

    hermes = HermesNodeHandler(
        model=MODEL,
        node_capabilities=["answer"],
        system_prompt="Reply with ONLY the word 'HERMES'. Nothing else.",
        max_turns=1,
        toolsets=[],
    )
    c.add_node("hermes_node", hermes)

    task = c.create_task(payload="Identify yourself.", path=["hermes_node"])
    result = await c.execute(task)

    report("hermes task completes", result.is_complete)
    report("hermes hop signed", result.hops[0].signature != "" if result.hops else False)

    output = str(result.context.get("hermes_node", "")).strip()
    report("hermes produced output", len(output) > 0, f"got: {output!r}")
    # Hermes may add formatting — just check it ran and returned something
    report("hermes hop succeeded",
           result.hops[0].status == HopStatus.COMPLETE if result.hops else False,
           f"status: {result.hops[0].status.value if result.hops else 'no hops'}")


# ═══════════════════════════════════════════════════════════════════
# EVAL 15: SmartConductor (LLM-driven routing)
# ═══════════════════════════════════════════════════════════════════

async def eval_smart_conductor():
    """SmartConductor asks the LLM to pick a path based on beacons."""
    from scionic import SmartConductor

    sc = SmartConductor(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        verify_signatures=False,
    )

    sc.add_node("researcher", llm(["search", "rag"],
        "Research the topic. One sentence summary."))
    sc.add_node("writer", llm(["draft", "edit"],
        "Write a polished version based on research. Two sentences."))
    sc.add_node("calculator", llm(["math", "compute"],
        "Do math calculations."))

    # LLM should pick researcher → writer (not calculator)
    task = sc.create_task(payload="Write a summary of quantum computing")

    report("smart path selected", len(task.path) > 0, f"path: {task.path}")
    report("path excludes calculator", "calculator" not in task.path,
           f"path: {task.path}")
    report("path includes writer", "writer" in task.path,
           f"path: {task.path}")

    result = await sc.execute(task)
    report("smart execution completes", result.is_complete)
    report("has output", any(len(str(v)) > 0 for k, v in result.context.items() if not k.startswith("_")))


# ═══════════════════════════════════════════════════════════════════
# EVAL 16: Transport serialization round-trip
# ═══════════════════════════════════════════════════════════════════

async def eval_transport():
    """Serialize a completed task to JSON and back. Verify nothing is lost."""
    from scionic.transport import (
        LocalTransport, task_to_json, task_from_json, task_to_dict,
    )

    # First, create a real task with hops
    c = Conductor()
    c.add_node("a", llm(["step"], "Say 'alpha'."))
    c.add_node("b", llm(["step"], "Say 'beta'."))

    task = c.create_task(payload="test transport", path=["a", "b"])
    result = await c.execute(task)

    # Serialize to JSON
    json_str = task_to_json(result)
    report("serializes to JSON", len(json_str) > 0)

    # Deserialize back
    restored = task_from_json(json_str)
    report("deserializes back", restored.id == result.id)
    report("path preserved", restored.path == result.path)
    report("hop count preserved", len(restored.hops) == len(result.hops),
           f"original={len(result.hops)}, restored={len(restored.hops)}")
    report("context preserved", "a" in restored.context and "b" in restored.context)
    report("signatures preserved",
           all(h.signature != "" for h in restored.hops))

    # Test LocalTransport queue round-trip
    transport = LocalTransport(serialize=True)
    await transport.send("node_x", result)
    received = await transport.receive("node_x")
    report("queue round-trip", received.id == result.id)
    report("queue preserves hops", len(received.hops) == len(result.hops))


# ═══════════════════════════════════════════════════════════════════
# EVAL 17: Code review pipeline end-to-end
# ═══════════════════════════════════════════════════════════════════

async def eval_code_review():
    """Run the code review pipeline with real LLM calls on vulnerable code."""
    c = Conductor(verify_signatures=True)

    c.add_trust_domain("tooling", "Tooling", allowed_peers=["analysis"])
    c.add_trust_domain("analysis", "Analysis", allowed_peers=["tooling"])

    c.add_node("security", llm(["security"],
        'Check this code for SQL injection, eval(), and other security issues. '
        'List each issue found. Be thorough.',
        max_tokens=256), trust_domain="tooling")

    c.add_node("reviewer", llm(["review"],
        "Review the code for correctness and style. 2-3 sentences.",
        max_tokens=128), trust_domain="analysis")

    c.add_node("approver", llm(["approve"],
        'Based on security and review findings, decide. Respond with JSON: '
        '{"approved": true|false, "reason": "one sentence"}. '
        'ONLY the JSON, no markdown.',
        max_tokens=64), trust_domain="analysis")

    c.add_peer_link("security", "reviewer")

    bad_code = 'query = f"SELECT * FROM users WHERE id = {user_input}"\nresult = eval(data)'

    task = c.create_task(
        payload=f"Review this code:\n```\n{bad_code}\n```",
        path=["security", "reviewer", "approver"],
    )
    result = await c.execute(task)

    report("pipeline completes", result.is_complete)
    report("3 hops", len(result.hops) == 3, f"got {len(result.hops)}")
    report("all hops signed", all(h.signature != "" for h in result.hops))
    report("chain verifies",
           all(h.verify(c._nodes[h.node_id].secret) for h in result.hops))

    sec_output = str(result.context.get("security", "")).lower()
    report("security found issues",
           any(w in sec_output for w in ["sql", "injection", "eval", "high", "critical"]),
           f"got: {sec_output[:100]}")

    approver_output = str(result.context.get("approver", ""))
    report("approver responded", len(approver_output) > 0, f"got: {approver_output[:100]}")


# ═══════════════════════════════════════════════════════════════════
# EVAL 18: FlowController + circuit breaker
# ═══════════════════════════════════════════════════════════════════

async def eval_flow_controller():
    """Test backpressure tracking and circuit breaker behavior."""
    flow = FlowController(circuit_failure_threshold=2, circuit_cooldown_seconds=0.5)
    flow.register_node("fast", max_capacity=3)
    flow.register_node("slow", max_capacity=1)

    # Fast node should accept
    report("fast node available", flow.can_accept("fast"))

    # Record load on slow node
    flow.record_start("slow")
    report("slow node at capacity", not flow.can_accept("slow"))

    # Release slow node
    flow.record_success("slow", 100.0)
    report("slow node available after release", flow.can_accept("slow"))

    # Pressure selection: should pick fast (lower pressure)
    flow.record_start("fast")  # fast: 1/3 = 33%
    flow.record_start("slow")  # slow: 1/1 = 100%
    selected = flow.select_by_pressure(["fast", "slow"])
    report("selects lower pressure", selected == "fast", f"got {selected}")
    flow.record_success("fast", 50.0)
    flow.record_success("slow", 200.0)

    # Circuit breaker: trip after 2 failures
    flow.record_failure("slow")
    report("circuit closed after 1 fail",
           flow.get_breaker("slow").state == CircuitState.CLOSED)
    flow.record_failure("slow")
    report("circuit opens after 2 fails",
           flow.get_breaker("slow").state == CircuitState.OPEN)
    report("slow blocked by circuit", not flow.can_accept("slow"))

    # Wait for cooldown
    import asyncio
    await asyncio.sleep(0.6)
    report("circuit half-open after cooldown", flow.can_accept("slow"))

    # Success resets
    flow.record_success("slow", 100.0)
    report("circuit closes on success",
           flow.get_breaker("slow").state == CircuitState.CLOSED)

    # Flow summary is readable
    summary = flow.flow_summary()
    report("flow summary contains nodes",
           "fast" in summary and "slow" in summary)


# ═══════════════════════════════════════════════════════════════════
# EVAL 19: TriageRouter — deterministic routing
# ═══════════════════════════════════════════════════════════════════

async def eval_triage_router():
    """TriageRouter: triage classifies, deterministic rules pick path."""
    tr = TriageRouter(
        triage_node_id="triager",
        verify_signatures=False,
        auto_reroute=False,
    )

    tr.add_node("triager", llm(["triage", "classify"],
        "Classify this task. Respond with JSON:\n"
        '{"complexity": "trivial|simple|complex", '
        '"needs_planning": true|false, '
        '"needs_research": true|false, '
        '"needs_code": true|false, '
        '"needs_tools": false, '
        '"reformulated": "clearer version of the task"}\n'
        "ONLY JSON, no markdown.",
        max_tokens=128))
    tr.add_node("quick_answer", llm(["answer", "quick"],
        "Answer directly in 1 sentence.", max_tokens=64))
    tr.add_node("planner", llm(["plan"],
        "Break into 2 subtasks. JSON: {\"subtasks\": [...]}", max_tokens=128))
    tr.add_node("executor", llm(["execute"],
        "Execute the task. 2 sentences.", max_tokens=128))
    tr.add_node("reviewer", llm(["review"],
        "Review. JSON: {\"approved\": true}", max_tokens=32))

    tr.add_default_rules()

    # Simple question should route to quick_answer
    task1 = await tr.create_task_routed(payload="What is 2+2?")
    report("trivial → quick_answer", task1.path == ["quick_answer"],
           f"got: {task1.path}")
    report("routing method is deterministic",
           task1.metadata.get("routing") == "deterministic",
           f"got: {task1.metadata.get('routing')}")

    # Execute it
    result1 = await tr.execute(task1)
    report("trivial task completes", result1.is_complete)
    report("trivial has output",
           len(str(result1.context.get("quick_answer", ""))) > 0)

    # Complex question should route through planner
    task2 = await tr.create_task_routed(
        payload="Design a database schema for a social network"
    )
    report("complex task has path", len(task2.path) > 1, f"got: {task2.path}")
    report("complex includes reviewer", "reviewer" in task2.path,
           f"got: {task2.path}")

    # Check reformulation
    if task2.metadata.get("original_payload"):
        report("task was reformulated", True)
    else:
        report("task has triage metadata", "triage" in task2.metadata or "routing" in task2.metadata,
               f"metadata keys: {list(task2.metadata.keys())}")


# ═══════════════════════════════════════════════════════════════════
# EVAL 20: execute_batch — concurrent tasks with semaphore
# ═══════════════════════════════════════════════════════════════════

async def eval_batch():
    """Run multiple tasks concurrently. Should be faster than sequential."""
    c = Conductor(verify_signatures=False)
    c.add_node("worker", llm(["work"], "Say 'done'. Nothing else.", max_tokens=8))

    tasks = [
        c.create_task(payload=f"task {i}", path=["worker"])
        for i in range(3)
    ]

    start = time.time()
    results = await c.execute_batch(tasks, concurrency=3)
    elapsed = time.time() - start

    report("all 3 completed", len(results) == 3, f"got {len(results)}")
    report("all successful",
           all(r.is_complete for r in results),
           f"statuses: {[r.is_complete for r in results]}")

    # With concurrency=3, should be roughly 1x single-task time, not 3x
    # (allowing generous margin for API variance)
    single_task_time = max(
        (r.hops[0].duration_ms if r.hops else 0) for r in results
    )
    report("parallel speedup",
           elapsed * 1000 < single_task_time * 2.5,
           f"wall={elapsed*1000:.0f}ms, slowest_single={single_task_time:.0f}ms")


# ═══════════════════════════════════════════════════════════════════
# EVAL 21: Pipeline timeout
# ═══════════════════════════════════════════════════════════════════

async def eval_pipeline_timeout():
    """execute_batch with a very short timeout should fail gracefully."""
    c = Conductor(verify_signatures=False)
    c.add_node("slow", llm(["work"], "Write a 500-word essay.", max_tokens=512))

    tasks = [c.create_task(payload="test", path=["slow"])]

    results = await c.execute_batch(tasks, concurrency=1, timeout_per_task=0.001)

    report("returns result despite timeout", len(results) == 1)
    if results:
        has_timeout = any("timeout" in (h.error or "").lower() for h in results[0].hops)
        report("timeout recorded in hop", has_timeout,
               f"errors: {[h.error for h in results[0].hops if h.error]}")


# ═══════════════════════════════════════════════════════════════════
# EVAL 22: Peer context visible in traceroute
# ═══════════════════════════════════════════════════════════════════

async def eval_peer_in_traceroute():
    """Peer messages should show up in the traceroute output."""
    c = Conductor(verify_signatures=False)
    c.add_node("a", llm(["step"], "Say 'hello'."))
    c.add_node("b", llm(["step"], "Say 'world'."))
    c.add_peer_link("a", "b")

    # Send peer message before executing
    await c.send_peer_message("a", "b", payload="hint from a")

    task = c.create_task(payload="test", path=["b"])
    result = await c.execute(task)

    trace = result.traceroute()
    report("traceroute shows peer count", "peer" in trace.lower(),
           f"trace: {trace}")
    report("hop records peer count",
           result.hops[0].peer_messages_received == 1 if result.hops else False,
           f"peer_count: {result.hops[0].peer_messages_received if result.hops else 'n/a'}")


# ═══════════════════════════════════════════════════════════════════
# EVAL 23: pip install works (no sys.path hack)
# ═══════════════════════════════════════════════════════════════════

async def eval_packaging():
    """Verify scionic is importable without sys.path manipulation."""
    import subprocess
    result = subprocess.run(
        ["python3", "-c", "import scionic; print(scionic.__version__)"],
        capture_output=True, text=True, timeout=10,
    )
    report("import scionic works", result.returncode == 0,
           f"stderr: {result.stderr.strip()}")
    report("version is 0.2.0", "0.2.0" in result.stdout,
           f"got: {result.stdout.strip()}")


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════

async def main():
    print("=" * 70)
    print("  scionic rapid eval — every feature, real API calls, no mocks")
    print(f"  Model: {MODEL} via OpenRouter")
    print("=" * 70)

    evals = [
        ("1. Basic forwarding + hop signing", eval_basic_forwarding),
        ("2. Context accumulation", eval_context_accumulation),
        ("3. Path selection by capability", eval_path_selection),
        ("4. Multi-path parallel", eval_multipath),
        ("5. IRQ propagation", eval_irq),
        ("6. IRQ masking", eval_irq_masking),
        ("7. Peer context injection", eval_peer_injection),
        ("8. Trust domain enforcement", eval_trust_domains),
        ("9. Hop signature verification", eval_hop_verification),
        ("10. Reroute on failure", eval_reroute),
        ("11. IRQ retry with feedback", eval_irq_retry),
        ("12. Full 4-node pipeline", eval_full_pipeline),
        ("13. Traceroute readability", eval_traceroute),
        ("14. Hermes agent node", eval_hermes),
        ("15. SmartConductor LLM routing", eval_smart_conductor),
        ("16. Transport serialization", eval_transport),
        ("17. Code review pipeline", eval_code_review),
        ("18. FlowController + circuit breaker", eval_flow_controller),
        ("19. TriageRouter deterministic routing", eval_triage_router),
        ("20. Batch execution (concurrent)", eval_batch),
        ("21. Pipeline timeout", eval_pipeline_timeout),
        ("22. Peer context in traceroute", eval_peer_in_traceroute),
        ("23. Packaging (pip install)", eval_packaging),
    ]

    for name, fn in evals:
        print(f"\n{'─' * 70}")
        print(f"  {name}")
        print(f"{'─' * 70}")
        start = time.time()
        try:
            await fn()
        except Exception as e:
            report(name, False, f"EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
        elapsed = time.time() - start
        print(f"  [{elapsed:.1f}s]")

    total_time = time.time() - TOTAL_START
    print(f"\n{'═' * 70}")
    print(f"  Results: {PASSED} passed, {FAILED} failed ({total_time:.1f}s)")
    print(f"{'═' * 70}")
    sys.exit(1 if FAILED > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
