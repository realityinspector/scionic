#!/usr/bin/env python3
"""
Example: Multi-agent research pipeline using scionic.

Demonstrates:
- Path-aware routing through agent nodes
- Traceroute-style execution logging
- Peer messaging between agents
- IRQ interrupts (premise invalidation)
- Multi-path parallel execution

This example uses simple in-process handlers (no LLM calls)
to demonstrate the protocol. Swap in HermesNodeHandler or
LLMNodeHandler for real agent-backed nodes.
"""

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scionic import (
    Conductor,
    IRQPriority,
    IRQType,
    PathPolicy,
    PeerMessage,
    Task,
)
from scionic.node import NodeHandler
from scionic.types import Hop

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("demo")


# ── Simple handlers that simulate agent work ─────────────────────────

class ResearcherHandler:
    """Simulates a research agent that gathers information."""

    def capabilities(self) -> list[str]:
        return ["search", "rag"]

    async def process(self, task: Task, hop: Hop) -> str:
        await asyncio.sleep(0.1)  # Simulate work
        topic = task.payload
        return (
            f"Research findings on '{topic}':\n"
            f"1. SCION was developed at ETH Zurich starting in 2011\n"
            f"2. It provides path-aware, cryptographically secured routing\n"
            f"3. Production deployment: Secure Swiss Finance Network (SSFN)\n"
            f"4. Key innovation: endpoints choose their forwarding paths"
        )


class AnalystHandler:
    """Simulates an analyst that processes research into insights."""

    def __init__(self):
        self.peer_context: list[str] = []

    def capabilities(self) -> list[str]:
        return ["analyze", "summarize"]

    async def process(self, task: Task, hop: Hop) -> str:
        await asyncio.sleep(0.15)

        # Use context from previous hops
        research = task.context.get("researcher", "No research provided")

        # Include any peer messages received
        peer_info = ""
        if self.peer_context:
            peer_info = "\nPeer context: " + "; ".join(self.peer_context)

        return (
            f"Analysis of research:\n"
            f"- Core value proposition: path control + cryptographic verification\n"
            f"- Market position: only production-grade BGP alternative\n"
            f"- Key risk: requires bilateral adoption (chicken-and-egg)\n"
            f"- Recommendation: focus on high-security verticals first"
            f"{peer_info}"
        )


class WriterHandler:
    """Simulates a writer that produces the final output."""

    def capabilities(self) -> list[str]:
        return ["draft", "edit"]

    async def process(self, task: Task, hop: Hop) -> str:
        await asyncio.sleep(0.1)

        analysis = task.context.get("analyst", "No analysis provided")

        return (
            "# SCION Technology Assessment\n\n"
            "## Executive Summary\n"
            "SCION represents a fundamental improvement over BGP for inter-domain "
            "routing, offering path-aware networking with cryptographic verification.\n\n"
            "## Key Findings\n"
            "Based on our analysis, SCION's strongest value proposition is in "
            "high-security verticals (finance, government, critical infrastructure) "
            "where path control and verification justify the adoption cost.\n\n"
            "## Recommendation\n"
            "Pursue a phased adoption strategy starting with the Secure Swiss "
            "Finance Network as a reference deployment."
        )


class FactCheckerHandler:
    """Alternative analyst — used to demonstrate multi-path execution."""

    def capabilities(self) -> list[str]:
        return ["analyze", "verify"]

    async def process(self, task: Task, hop: Hop) -> str:
        await asyncio.sleep(0.2)
        return (
            "Fact-check results:\n"
            "- ETH Zurich origin: CONFIRMED\n"
            "- SSFN deployment: CONFIRMED (SIX Swiss Exchange)\n"
            "- BGP replacement claim: PARTIALLY TRUE (complement, not full replacement)\n"
            "- Path selection by endpoints: CONFIRMED"
        )


# ── Demo scenarios ───────────────────────────────────────────────────

async def demo_basic_pipeline():
    """Basic: researcher → analyst → writer with traceroute."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Pipeline with Traceroute")
    print("=" * 70)

    conductor = Conductor()

    conductor.add_node("researcher", ResearcherHandler(),
                       cost_per_call=0.01, avg_latency_ms=100)
    conductor.add_node("analyst", AnalystHandler(),
                       cost_per_call=0.02, avg_latency_ms=150)
    conductor.add_node("writer", WriterHandler(),
                       cost_per_call=0.03, avg_latency_ms=100)

    task = conductor.create_task(
        payload="SCION internet architecture",
        path=["researcher", "analyst", "writer"],
    )

    result = await conductor.execute(task)

    print("\n--- Traceroute ---")
    print(result.traceroute())
    print("\n--- Final Output ---")
    print(result.context.get("writer", "(no output)"))


async def demo_auto_path_selection():
    """Auto path selection based on required capabilities."""
    print("\n" + "=" * 70)
    print("DEMO 2: Automatic Path Selection")
    print("=" * 70)

    conductor = Conductor()

    conductor.add_node("researcher", ResearcherHandler(),
                       cost_per_call=0.01, avg_latency_ms=100)
    conductor.add_node("analyst", AnalystHandler(),
                       cost_per_call=0.02, avg_latency_ms=150)
    conductor.add_node("fact_checker", FactCheckerHandler(),
                       cost_per_call=0.015, avg_latency_ms=200)
    conductor.add_node("writer", WriterHandler(),
                       cost_per_call=0.03, avg_latency_ms=100)

    # Let the conductor pick the path
    task = conductor.create_task(
        payload="SCION internet architecture",
        required_capabilities=["search", "analyze", "draft"],
        policy=PathPolicy(prefer_low_cost=True),
    )

    print(f"Auto-selected path: {' → '.join(task.path)}")
    result = await conductor.execute(task)
    print("\n--- Traceroute ---")
    print(result.traceroute())


async def demo_peer_messaging():
    """Peer links: researcher sends context directly to writer."""
    print("\n" + "=" * 70)
    print("DEMO 3: Peer Messaging (Lateral Communication)")
    print("=" * 70)

    conductor = Conductor()

    conductor.add_node("researcher", ResearcherHandler(),
                       cost_per_call=0.01)
    analyst_handler = AnalystHandler()
    conductor.add_node("analyst", analyst_handler,
                       cost_per_call=0.02)
    conductor.add_node("writer", WriterHandler(),
                       cost_per_call=0.03)

    # Establish peering between researcher and analyst
    conductor.add_peer_link("researcher", "analyst")

    print(f"\n{conductor.topology_summary()}")

    # Send a peer message (simulating researcher sharing extra context)
    sent = await conductor.send_peer_message(
        source="researcher",
        target="analyst",
        payload="Note: Anapaya Systems is the commercial entity behind SCION",
        message_type="context",
    )
    print(f"\nPeer message delivered: {sent}")

    # Now run the pipeline — analyst has extra context from peer
    # (In a real system, the peer handler would inject this into the analyst's context)
    task = conductor.create_task(
        payload="SCION internet architecture",
        path=["researcher", "analyst", "writer"],
    )
    result = await conductor.execute(task)
    print("\n--- Traceroute ---")
    print(result.traceroute())


async def demo_irq_interrupt():
    """IRQ: analyst discovers premise is wrong, fires interrupt."""
    print("\n" + "=" * 70)
    print("DEMO 4: IRQ Interrupt Propagation")
    print("=" * 70)

    conductor = Conductor()

    conductor.add_node("researcher", ResearcherHandler())
    conductor.add_node("analyst", AnalystHandler())
    conductor.add_node("writer", WriterHandler())

    task = conductor.create_task(
        payload="SCION internet architecture",
        path=["researcher", "analyst", "writer"],
    )

    # Execute researcher hop first
    task = await conductor.forwarder.get_node("researcher").execute(task)
    print(f"After researcher: {task.traceroute()}")

    # Analyst discovers something and fires an IRQ
    print("\n--- Analyst fires IRQ: premise needs updating ---")
    delivered = await conductor.fire_irq(
        source="analyst",
        irq_type=IRQType.CONTEXT_UPDATE,
        payload={"correction": "SCION is not just academic — it's in production"},
        reason="Research missed the SSFN production deployment details",
        task=task,
        priority=IRQPriority.HIGH,
    )
    print(f"IRQ delivered to {delivered} handlers")

    # Continue execution
    task = await conductor.forwarder.get_node("analyst").execute(task)
    task = await conductor.forwarder.get_node("writer").execute(task)
    print(f"\n--- Final Traceroute ---")
    print(task.traceroute())


async def demo_multipath():
    """Multi-path: same task, two different analysis paths."""
    print("\n" + "=" * 70)
    print("DEMO 5: Multi-Path Parallel Execution")
    print("=" * 70)

    conductor = Conductor()

    conductor.add_node("researcher", ResearcherHandler(),
                       cost_per_call=0.01)
    conductor.add_node("analyst", AnalystHandler(),
                       cost_per_call=0.02)
    conductor.add_node("fact_checker", FactCheckerHandler(),
                       cost_per_call=0.015)
    conductor.add_node("writer", WriterHandler(),
                       cost_per_call=0.03)

    # Two paths: one through analyst, one through fact_checker
    task = Task(payload="SCION internet architecture")
    results = await conductor.forwarder.forward_multipath(
        task,
        alternate_paths=[
            ["researcher", "analyst", "writer"],
            ["researcher", "fact_checker", "writer"],
        ],
    )

    for i, result in enumerate(results):
        print(f"\n--- Path {i+1} Traceroute ---")
        print(result.traceroute())
        print(f"Final output preview: {str(result.context.get('writer', ''))[:100]}...")


# ── Main ─────────────────────────────────────────────────────────────

async def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  scionic: Path-Aware Agent Orchestration            ║")
    print("║  SCION-inspired middleware for multi-agent systems      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    await demo_basic_pipeline()
    await demo_auto_path_selection()
    await demo_peer_messaging()
    await demo_irq_interrupt()
    await demo_multipath()

    print("\n" + "=" * 70)
    print("All demos complete.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
