#!/usr/bin/env python3
"""
Task Manager v3 — TriageRouter with explicit routing + execute_batch + execute_with_review.

Demonstrates:
  - TriageRouter with EXPLICIT inline routing rules (no add_default_rules)
  - execute_batch for concurrent task execution
  - execute_with_review for reviewer IRQ retry
  - Trust domains with real capability restrictions
  - rules_summary() and flow_summary() reports
  - Model: anthropic/claude-haiku-4.5
  - 4 tasks: capital of Japan, rate limiter design, list /tmp, palindrome function
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from scionic import (
    TriageRouter,
    RoutingRule,
    TrustDomain,
    Task,
)
from scionic.adapters.llm import LLMNodeHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_api_key() -> str:
    """Load OPENROUTER_API_KEY from ~/.hermes/.env"""
    env_file = Path.home() / ".hermes" / ".env"
    if not env_file.exists():
        raise RuntimeError(f"Missing {env_file}")
    
    for line in env_file.read_text().split("\n"):
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    
    raise RuntimeError("OPENROUTER_API_KEY not found in ~/.hermes/.env")


async def setup_router() -> TriageRouter:
    """Build router with trust domains and explicit routing rules."""
    api_key = load_api_key()
    model = "anthropic/claude-haiku-4.5"
    
    router = TriageRouter(
        triage_node_id="triager",
        llm_model=model,
        llm_api_key=api_key,
        llm_base_url="https://openrouter.ai/api/v1",
    )
    
    # ── Trust domains ────────────────────────────────────────────────
    
    router.add_trust_domain(
        domain_id="triage",
        name="Triage Domain",
        description="Lightweight classification and routing",
        allowed_capabilities=["triage", "classify", "answer", "quick"],
    )
    
    router.add_trust_domain(
        domain_id="execution",
        name="Execution Domain",
        description="Code, terminal, research, planning",
        allowed_capabilities=["plan", "research", "execute", "code", "terminal"],
    )
    
    router.add_trust_domain(
        domain_id="review",
        name="Review Domain",
        description="Quality gates and approval",
        allowed_capabilities=["review", "approve"],
    )
    
    # ── Nodes ────────────────────────────────────────────────────────
    
    # Triager (classifies and reformulates)
    triager = LLMNodeHandler(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        node_capabilities=["triage", "classify"],
        system_prompt=(
            "You are a task classifier. Analyze the request and output "
            "JSON with: complexity (trivial/simple/moderate/complex), "
            "needs_code (bool), needs_research (bool), needs_planning (bool), "
            "needs_terminal (bool), needs_tools (bool), reformulated (str or null). "
            "Be concise."
        ),
        max_tokens=512,
        temperature=0.0,
    )
    router.add_node(
        "triager",
        triager,
        trust_domain="triage",
        max_concurrency=2,
    )
    
    # Quick answerer (trivial questions)
    quick = LLMNodeHandler(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        node_capabilities=["answer", "quick"],
        system_prompt="Answer the question directly and concisely.",
        max_tokens=256,
        temperature=0.7,
    )
    router.add_node(
        "quick_answer",
        quick,
        trust_domain="triage",
        max_concurrency=3,
    )
    
    # Planner (break down complex tasks)
    planner = LLMNodeHandler(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        node_capabilities=["plan"],
        system_prompt=(
            "Break down the task into steps. Output as a numbered list. "
            "Be thorough but concise."
        ),
        max_tokens=1024,
        temperature=0.7,
    )
    router.add_node(
        "planner",
        planner,
        trust_domain="execution",
        max_concurrency=2,
    )
    
    # Researcher (gather information)
    researcher = LLMNodeHandler(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        node_capabilities=["research"],
        system_prompt=(
            "Research and gather relevant information. "
            "Provide sources if possible. Be factual."
        ),
        max_tokens=1024,
        temperature=0.7,
    )
    router.add_node(
        "researcher",
        researcher,
        trust_domain="execution",
        max_concurrency=2,
    )
    
    # Executor (implement solutions)
    executor = LLMNodeHandler(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        node_capabilities=["execute", "code"],
        system_prompt=(
            "Execute the task. Provide code, scripts, or detailed solutions. "
            "Output should be ready to use."
        ),
        max_tokens=2048,
        temperature=0.7,
    )
    router.add_node(
        "executor",
        executor,
        trust_domain="execution",
        max_concurrency=1,
    )
    
    # Reviewer (quality gate)
    reviewer = LLMNodeHandler(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        node_capabilities=["review", "approve"],
        system_prompt=(
            "Review the work. Output JSON: {\"approved\": bool, \"feedback\": str}. "
            "Check for correctness, clarity, and completeness. "
            "Be constructive if you reject."
        ),
        max_tokens=512,
        temperature=0.0,
    )
    router.add_node(
        "reviewer",
        reviewer,
        trust_domain="review",
        max_concurrency=2,
    )
    
    # ── Explicit routing rules ───────────────────────────────────────
    
    def route_trivial(triage_data: dict):
        """Trivial/simple → quick_answer"""
        if triage_data.get("complexity") in ("trivial", "simple"):
            return ["quick_answer"]
        return None
    
    def route_code(triage_data: dict):
        """Code/terminal → planner → executor (stays in execution domain)"""
        if triage_data.get("needs_code") or triage_data.get("needs_terminal"):
            return ["planner", "executor"]
        return None
    
    def route_research_only(triage_data: dict):
        """Research without planning → researcher → executor"""
        if (triage_data.get("needs_research") and 
            not triage_data.get("needs_planning")):
            return ["researcher", "executor"]
        return None
    
    def route_planning_only(triage_data: dict):
        """Planning without research → planner → executor"""
        if (triage_data.get("needs_planning") and 
            not triage_data.get("needs_research")):
            return ["planner", "executor"]
        return None
    
    def route_complex(triage_data: dict):
        """Complex (plan + research) → planner → researcher → executor"""
        if (triage_data.get("needs_planning") and 
            triage_data.get("needs_research")):
            return ["planner", "researcher", "executor"]
        return None
    
    # Add rules in priority order
    router.add_routing_rule(RoutingRule(
        name="trivial",
        description="Simple questions → quick_answer",
        priority=10,
        match=route_trivial,
    ))
    
    router.add_routing_rule(RoutingRule(
        name="code",
        description="Code/terminal tasks → planner → executor",
        priority=20,
        match=route_code,
    ))
    
    router.add_routing_rule(RoutingRule(
        name="research_only",
        description="Research only → researcher → executor",
        priority=30,
        match=route_research_only,
    ))
    
    router.add_routing_rule(RoutingRule(
        name="planning_only",
        description="Planning only → planner → executor",
        priority=40,
        match=route_planning_only,
    ))
    
    router.add_routing_rule(RoutingRule(
        name="complex",
        description="Complex → planner → researcher → executor",
        priority=50,
        match=route_complex,
    ))
    
    return router


async def main():
    """Run task manager with 4 example tasks."""
    
    logger.info("=== Task Manager v3 ===")
    
    router = await setup_router()
    
    # Show configuration
    print("\n" + "=" * 70)
    print(router.rules_summary())
    print("\n" + "=" * 70)
    print(router.flow.flow_summary())
    print("=" * 70 + "\n")
    
    # ── Create 4 tasks ───────────────────────────────────────────────
    
    tasks = [
        {
            "id": "task_1",
            "payload": "What is the capital of Japan?",
        },
        {
            "id": "task_2",
            "payload": (
                "Design a rate limiter. Explain the concept, "
                "trade-offs, and provide a Python implementation."
            ),
        },
        {
            "id": "task_3",
            "payload": "List the contents of /tmp directory",
        },
        {
            "id": "task_4",
            "payload": (
                "Write a Python function that checks if a string "
                "is a palindrome. Include test cases."
            ),
        },
    ]
    
    # Create Task objects with routed paths
    routed_tasks = []
    for task_spec in tasks:
        task = await router.create_task_routed(
            payload=task_spec["payload"],
            trust_domain="default",
        )
        logger.info(
            f"Task {task.id[:8]}: routed via {' → '.join(task.path)} "
            f"(rule: {task.metadata.get('routing_rule', 'auto')})"
        )
        routed_tasks.append(task)
    
    # ── Execute with batch + review ──────────────────────────────────
    
    logger.info("\nExecuting batch (concurrency=2)...")
    print()
    
    # Execute first batch (tasks routed but not yet executed)
    results = await router.execute_batch(
        routed_tasks,
        concurrency=2,
        timeout_per_task=30.0,
    )
    
    # Run each result through review
    logger.info("\nApplying reviewer IRQ retry loop...")
    final_results = []
    for result in results:
        reviewed = await router.execute_with_review(
            result,
            reviewer_node="reviewer",
            max_review_retries=1,
        )
        final_results.append(reviewed)
    
    # ── Report ───────────────────────────────────────────────────────
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    for i, task in enumerate(final_results, 1):
        print(f"\nTask {i}: {task.payload[:60]}...")
        print(f"  Route: {' → '.join(task.path)}")
        print(f"  Routing: {task.metadata.get('routing', 'unknown')}")
        if task.metadata.get('routing_rule'):
            print(f"  Rule: {task.metadata.get('routing_rule')}")
        print(f"  Status: {len(task.hops)} hops")
        if task.is_complete:
            print(f"  ✓ Complete")
        else:
            print(f"  ✗ Incomplete (at hop {task.current_hop_index})")
        
        # Show traceroute
        for line in task.traceroute().split("\n")[1:]:
            print(f"    {line}")
    
    print("\n" + "=" * 70)
    print(router.flow.flow_summary())
    print("=" * 70)
    
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
