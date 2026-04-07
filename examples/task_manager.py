#!/usr/bin/env python3
"""
Task Manager v4 — scionic demo with every feature visible.

Routing rules are the thing being shown off. They're explicit, ordered,
non-overlapping, with a catch-all. Review runs concurrently with batch.
Trust domains restrict capabilities. Peer links share context laterally.
"""

import asyncio
import logging
from pathlib import Path

from scionic import TriageRouter, RoutingRule
from scionic.adapters.llm import LLMNodeHandler

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("task_manager")

MODEL = "anthropic/claude-haiku-4.5"
BASE = "https://openrouter.ai/api/v1"


def load_api_key() -> str:
    env_file = Path.home() / ".hermes" / ".env"
    for line in env_file.read_text().split("\n"):
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENROUTER_API_KEY not found in ~/.hermes/.env")


def llm(caps: list[str], prompt: str, max_tokens: int = 256, temp: float = 0.2) -> LLMNodeHandler:
    return LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        node_capabilities=caps, system_prompt=prompt,
        max_tokens=max_tokens, temperature=temp,
    )


API_KEY = load_api_key()


# ── Router setup ─────────────────────────────────────────────────────

def build_router() -> TriageRouter:
    r = TriageRouter(
        triage_node_id="triager",
        verify_signatures=True,
        auto_reroute=True,
        max_retries=1,
    )

    # Trust domains with real capability restrictions
    r.add_trust_domain("triage", "Triage",
                       allowed_capabilities=["triage", "classify", "answer", "quick"],
                       allowed_peers=["execution"])
    r.add_trust_domain("execution", "Execution",
                       allowed_capabilities=["plan", "research", "execute", "code"],
                       allowed_peers=["triage", "review"])
    r.add_trust_domain("review", "Review",
                       allowed_capabilities=["review", "approve"],
                       allowed_peers=["execution"])

    # ── Nodes ────────────────────────────────────────────────────────

    r.add_node("triager", llm(
        ["triage", "classify"],
        "Classify the task. Respond with JSON only:\n"
        '{"complexity":"trivial|simple|moderate|complex",'
        '"needs_code":bool,"needs_research":bool,"needs_planning":bool,'
        '"needs_terminal":bool,"reformulated":"clearer version or null"}\n'
        "Rules: trivial=factual lookup, simple=one-step, "
        "moderate=multi-step no research, complex=multi-step with research.",
        max_tokens=128, temp=0.0,
    ), trust_domain="triage", max_concurrency=4)

    r.add_node("quick_answer", llm(
        ["answer", "quick"],
        "Answer directly. 1-2 sentences. Be precise.",
        max_tokens=128, temp=0.0,
    ), trust_domain="triage", max_concurrency=4)

    r.add_node("planner", llm(
        ["plan"],
        "Break the task into 2-4 concrete steps. Be specific.",
        max_tokens=256, temp=0.2,
    ), trust_domain="execution", max_concurrency=2)

    r.add_node("researcher", llm(
        ["research"],
        "Research the topic. Provide 3-5 key facts with specifics.",
        max_tokens=256, temp=0.2,
    ), trust_domain="execution", max_concurrency=2)

    r.add_node("executor", llm(
        ["execute", "code"],
        "Execute the task using context from previous steps. "
        "If you see _feedback_for_executor, incorporate it. "
        "Check peer messages for hints. Be thorough.",
        max_tokens=512, temp=0.3,
    ), trust_domain="execution", max_concurrency=2)

    r.add_node("reviewer", llm(
        ["review", "approve"],
        'Review the work. Respond JSON only:\n'
        '{"approved":true|false,"feedback":"one sentence"}\n'
        "Approve if it adequately addresses the task. "
        "If rejecting, say specifically what to fix.",
        max_tokens=64, temp=0.0,
    ), trust_domain="review", max_concurrency=4)

    # Peer links — lateral context sharing
    r.add_peer_link("planner", "executor")
    r.add_peer_link("researcher", "executor")

    # ── Routing rules (explicit, non-overlapping, with catch-all) ────
    #
    # Evaluation order: lowest priority number first. First match wins.
    # Rules are mutually exclusive by condition nesting:
    #   trivial/simple → quick_answer  (exits before checking flags)
    #   complex (plan+research) → full pipeline  (checked before partials)
    #   code/terminal → code path  (checked before plan-only/research-only)
    #   planning-only → planner path
    #   research-only → researcher path
    #   catch-all → executor (nothing matched)

    r.add_routing_rule(RoutingRule(
        name="trivial",
        description="Trivial/simple complexity → quick_answer",
        priority=10,
        match=lambda t: (
            ["quick_answer"]
            if t.get("complexity") in ("trivial", "simple")
            else None
        ),
    ))

    r.add_routing_rule(RoutingRule(
        name="full_pipeline",
        description="Needs both planning AND research → planner → researcher → executor",
        priority=20,
        match=lambda t: (
            ["planner", "researcher", "executor"]
            if t.get("needs_planning") and t.get("needs_research")
            else None
        ),
    ))

    r.add_routing_rule(RoutingRule(
        name="code",
        description="Needs code/terminal (without research) → planner → executor",
        priority=30,
        match=lambda t: (
            ["planner", "executor"]
            if (t.get("needs_code") or t.get("needs_terminal"))
            and not t.get("needs_research")
            else None
        ),
    ))

    r.add_routing_rule(RoutingRule(
        name="planning_only",
        description="Needs planning (no code, no research) → planner → executor",
        priority=40,
        match=lambda t: (
            ["planner", "executor"]
            if t.get("needs_planning")
            and not t.get("needs_research")
            and not t.get("needs_code")
            else None
        ),
    ))

    r.add_routing_rule(RoutingRule(
        name="research_only",
        description="Needs research (no planning) → researcher → executor",
        priority=50,
        match=lambda t: (
            ["researcher", "executor"]
            if t.get("needs_research") and not t.get("needs_planning")
            else None
        ),
    ))

    r.add_routing_rule(RoutingRule(
        name="default",
        description="Catch-all → executor (nothing else matched)",
        priority=100,
        match=lambda t: ["executor"],
    ))

    return r


# ── Main ─────────────────────────────────────────────────────────────

async def main():
    router = build_router()

    print("=" * 70)
    print("  Task Manager v4 — scionic demo")
    print("=" * 70)
    print()
    print(router.rules_summary())
    print()
    print(router.topology_summary())
    print()

    # ── Create routed tasks ──────────────────────────────────────────

    payloads = [
        "What is the capital of Japan?",
        "Design a rate limiter for an API with token bucket algorithm, "
        "data structures, and Python implementation.",
        "List the files in /tmp and report how many there are.",
        "Write a Python function is_palindrome(s: str) -> bool that handles "
        "empty strings, case insensitivity, and punctuation. Include tests.",
    ]

    print("── Triaging ──")
    tasks = []
    for payload in payloads:
        task = await router.create_task_routed(payload=payload)
        rule = task.metadata.get("routing_rule", "default")
        path_str = " → ".join(task.path)
        reformulated = task.metadata.get("original_payload")
        print(f"  [{rule:15s}] {path_str}")
        if reformulated:
            print(f"                  reformulated from: {str(reformulated)[:60]}")
        tasks.append(task)

    # ── Execute concurrently, then review concurrently ───────────────

    print(f"\n── Executing ({len(tasks)} tasks, concurrency=4) ──")
    results = await router.execute_batch(tasks, concurrency=4, timeout_per_task=60.0)

    print(f"\n── Reviewing ({len(results)} results, concurrent) ──")
    review_coros = [
        router.execute_with_review(r, reviewer_node="reviewer", max_review_retries=1)
        for r in results
    ]
    final = await asyncio.gather(*review_coros)

    # ── Results ──────────────────────────────────────────────────────

    print(f"\n{'═' * 70}")
    print("  RESULTS")
    print(f"{'═' * 70}")

    for i, task in enumerate(final, 1):
        status = "DONE" if task.is_complete else "INCOMPLETE"
        rule = task.metadata.get("routing_rule", "?")
        path_str = " → ".join(task.path)
        hop_count = len(task.hops)

        print(f"\n{'─' * 70}")
        print(f"  Task {i}: {str(task.payload)[:65]}")
        print(f"  Rule: {rule} | Path: {path_str} | Status: {status} | Hops: {hop_count}")

        # Traceroute
        for line in task.traceroute().split("\n")[1:]:
            print(f"    {line}")

        # Show executor/quick_answer output (the deliverable)
        for node_id in ["executor", "quick_answer"]:
            output = task.context.get(node_id)
            if output:
                text = str(output)
                if len(text) > 300:
                    text = text[:300] + "..."
                print(f"\n  Output ({node_id}):")
                for line in text.split("\n")[:10]:
                    print(f"    {line}")
                if text.count("\n") > 10:
                    print(f"    ... ({text.count(chr(10))} lines total)")
                break

        # Show review verdict
        review = task.context.get("reviewer")
        if review:
            print(f"\n  Review: {str(review)[:120]}")

    # ── Flow summary ─────────────────────────────────────────────────

    print(f"\n{'═' * 70}")
    print(router.flow.flow_summary())
    print(f"{'═' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
