#!/usr/bin/env python3
"""
Integration tests for scion-graph.

These tests make REAL API calls to OpenRouter. No mocks.
Requires OPENROUTER_API_KEY environment variable.
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load API key from Hermes config if not in env
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
    print("FATAL: No OPENROUTER_API_KEY found. Cannot run integration tests.")
    sys.exit(1)

from scion_graph import Conductor, IRQPriority, IRQType, PathPolicy
from scion_graph.adapters.llm import LLMNodeHandler
from scion_graph.types import Hop, Task

# Use haiku for speed and cost — this is testing the protocol, not the model
MODEL = "anthropic/claude-haiku-4.5"
PASSED = 0
FAILED = 0


def report(name: str, passed: bool, detail: str = ""):
    global PASSED, FAILED
    if passed:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}: {detail}")


async def test_single_node_real_llm():
    """One node, one real LLM call, verify hop signing and output."""
    conductor = Conductor()

    handler = LLMNodeHandler(
        model=MODEL,
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
        node_capabilities=["answer"],
        system_prompt="Reply with ONLY the number 42. Nothing else.",
        max_tokens=16,
        temperature=0.0,
    )
    conductor.add_node("answerer", handler)

    task = conductor.create_task(
        payload="What is the answer to life, the universe, and everything?",
        path=["answerer"],
    )

    result = await conductor.execute(task)

    # Verify the task completed
    report("task completed", result.is_complete)

    # Verify we got exactly one hop
    report("one hop recorded", len(result.hops) == 1, f"got {len(result.hops)}")

    hop = result.hops[0]

    # Verify hop was signed
    report("hop is signed", hop.signature != "", "empty signature")

    # Verify hop signature is valid
    node = conductor._nodes["answerer"]
    report("signature verifies", hop.verify(node.secret), "signature mismatch")

    # Verify we got a real response containing "42"
    output = str(result.context.get("answerer", ""))
    report("LLM returned output", len(output) > 0, "empty output")
    report("output contains 42", "42" in output, f"got: {output!r}")

    # Verify timing was recorded
    report("timing recorded", hop.duration_ms > 0, f"duration={hop.duration_ms}")

    # Verify tokens were tracked
    report("tokens tracked", hop.tokens_used > 0, f"tokens={hop.tokens_used}")


async def test_two_node_pipeline_real_llm():
    """Two real LLM nodes in sequence. Second node sees first node's output."""
    conductor = Conductor()

    node1 = LLMNodeHandler(
        model=MODEL,
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
        node_capabilities=["generate"],
        system_prompt="Generate exactly 3 random words, separated by commas. Nothing else.",
        max_tokens=32,
        temperature=0.9,
    )

    node2 = LLMNodeHandler(
        model=MODEL,
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
        node_capabilities=["count"],
        system_prompt=(
            "You will receive words from a previous step. "
            "Count how many words there are and reply with ONLY the number. Nothing else."
        ),
        max_tokens=16,
        temperature=0.0,
    )

    conductor.add_node("generator", node1)
    conductor.add_node("counter", node2)

    task = conductor.create_task(
        payload="Generate some words.",
        path=["generator", "counter"],
    )

    result = await conductor.execute(task)

    report("pipeline completed", result.is_complete)
    report("two hops recorded", len(result.hops) == 2, f"got {len(result.hops)}")

    # Verify both hops succeeded
    for i, hop in enumerate(result.hops):
        report(f"hop {i+1} succeeded", hop.status.value == "complete", f"status={hop.status}")
        report(f"hop {i+1} signed", hop.signature != "")

    # Verify second node got context from first
    gen_output = str(result.context.get("generator", ""))
    count_output = str(result.context.get("counter", ""))
    report("generator produced output", len(gen_output) > 0, f"got: {gen_output!r}")
    report("counter produced output", len(count_output) > 0, f"got: {count_output!r}")
    report("counter saw generator context", "3" in count_output,
           f"expected '3', got: {count_output!r}")

    # Verify traceroute is readable
    trace = result.traceroute()
    report("traceroute contains both nodes",
           "generator" in trace and "counter" in trace, trace)


async def test_multipath_real_llm():
    """Same question, two different models/prompts, parallel execution."""
    conductor = Conductor()

    optimist = LLMNodeHandler(
        model=MODEL,
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
        node_capabilities=["answer"],
        system_prompt="You are an optimist. Reply in one short sentence.",
        max_tokens=64,
        temperature=0.7,
    )

    pessimist = LLMNodeHandler(
        model=MODEL,
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
        node_capabilities=["answer"],
        system_prompt="You are a pessimist. Reply in one short sentence.",
        max_tokens=64,
        temperature=0.7,
    )

    conductor.add_node("optimist", optimist)
    conductor.add_node("pessimist", pessimist)

    task = Task(payload="How will AI change the world?")
    results = await conductor.forwarder.forward_multipath(
        task,
        alternate_paths=[["optimist"], ["pessimist"]],
    )

    report("two results returned", len(results) == 2, f"got {len(results)}")

    for i, r in enumerate(results):
        report(f"path {i+1} completed", r.is_complete)
        node_name = r.path[0] if r.path else "unknown"
        output = str(r.context.get(node_name, ""))
        report(f"path {i+1} ({node_name}) has output", len(output) > 0, f"got: {output!r}")


async def test_irq_during_pipeline():
    """Fire an IRQ between real LLM hops — verify it's recorded and delivered."""
    conductor = Conductor()

    node1 = LLMNodeHandler(
        model=MODEL,
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
        node_capabilities=["research"],
        system_prompt="Summarize the topic in one sentence.",
        max_tokens=64,
        temperature=0.3,
    )

    node2 = LLMNodeHandler(
        model=MODEL,
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
        node_capabilities=["write"],
        system_prompt="Write a one-sentence conclusion based on the context provided.",
        max_tokens=64,
        temperature=0.3,
    )

    conductor.add_node("researcher", node1)
    conductor.add_node("writer", node2)

    task = conductor.create_task(
        payload="The SCION internet architecture",
        path=["researcher", "writer"],
    )

    # Execute first hop
    researcher_node = conductor._nodes["researcher"]
    task = await researcher_node.execute(task)
    report("first hop complete", len(task.hops) == 1 and task.hops[0].status.value == "complete")

    # Fire IRQ between hops
    irq_count = await conductor.fire_irq(
        source="researcher",
        irq_type=IRQType.CONTEXT_UPDATE,
        payload="SCION is already in production at SIX Swiss Exchange",
        reason="Additional context discovered",
        task=task,
        priority=IRQPriority.HIGH,
    )
    report("IRQ delivered", irq_count > 0, f"delivered to {irq_count}")
    report("IRQ in conductor log", len(conductor._irq_log) > 0)

    # Execute second hop
    writer_node = conductor._nodes["writer"]
    task = await writer_node.execute(task)
    report("second hop complete", len(task.hops) == 2)
    report("full pipeline complete", task.is_complete)


async def test_path_selection_with_real_nodes():
    """Register nodes with capabilities, let PathSelector choose the route."""
    conductor = Conductor()

    cheap = LLMNodeHandler(
        model=MODEL,
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
        node_capabilities=["summarize"],
        system_prompt="Summarize in one sentence.",
        max_tokens=64,
    )

    expensive = LLMNodeHandler(
        model=MODEL,
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
        node_capabilities=["summarize"],
        system_prompt="Provide a detailed summary.",
        max_tokens=128,
    )

    conductor.add_node("cheap_summarizer", cheap,
                       cost_per_call=0.001, avg_latency_ms=500)
    conductor.add_node("expensive_summarizer", expensive,
                       cost_per_call=0.05, avg_latency_ms=2000)

    # With prefer_low_cost, should pick cheap
    task = conductor.create_task(
        payload="Explain SCION in brief",
        required_capabilities=["summarize"],
        policy=PathPolicy(prefer_low_cost=True),
    )

    report("path auto-selected", len(task.path) == 1)
    report("cheap node selected", task.path[0] == "cheap_summarizer",
           f"got: {task.path}")

    # Actually execute it
    result = await conductor.execute(task)
    report("auto-selected path executed", result.is_complete)
    output = str(result.context.get("cheap_summarizer", ""))
    report("got real LLM output", len(output) > 0, f"got: {output!r}")


async def main():
    print("=" * 60)
    print("scion-graph integration tests (REAL API calls)")
    print(f"Model: {MODEL}")
    print(f"API: OpenRouter")
    print("=" * 60)

    tests = [
        ("Single node, real LLM", test_single_node_real_llm),
        ("Two-node pipeline", test_two_node_pipeline_real_llm),
        ("Multi-path parallel", test_multipath_real_llm),
        ("IRQ between hops", test_irq_during_pipeline),
        ("Auto path selection", test_path_selection_with_real_nodes),
    ]

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        start = time.time()
        try:
            await test_fn()
        except Exception as e:
            report(name, False, f"EXCEPTION: {e}")
        elapsed = time.time() - start
        print(f"  ({elapsed:.1f}s)")

    print(f"\n{'=' * 60}")
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 60}")
    sys.exit(1 if FAILED > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
