#!/usr/bin/env python3
"""
Task Manager — scionic v0.2.0 demo.

Processes four tasks concurrently through a TriageRouter pipeline.
Each task is triaged, deterministically routed, and executed via
execute_batch(). Results are displayed as a kanban board with
detailed traceroutes and a FlowController summary.

Nodes     : triager, quick_answer, planner, researcher,
            executor, hermes_executor, reviewer
Domains   : triage, execution, review (peered)
Peer links: planner↔executor, researcher↔executor
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from scionic import TriageRouter, HopStatus
from scionic.adapters.llm import LLMNodeHandler
from scionic.adapters.hermes import HermesNodeHandler


# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "anthropic/claude-haiku-4.5"
BASE  = "https://openrouter.ai/api/v1"


def load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_path):
        with open(env_path) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY=") and "=" in line:
                    return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENROUTER_API_KEY not found in env or ~/.hermes/.env")


# ── Task definitions ──────────────────────────────────────────────────────────

@dataclass
class TaskSpec:
    label: str
    payload: str
    trust_domain: str = "triage"


TASKS: list[TaskSpec] = [
    TaskSpec(
        label="Capital lookup",
        payload="What is the capital of Japan?",
    ),
    TaskSpec(
        label="Rate limiter design",
        payload="Design a rate limiter for an API",
    ),
    TaskSpec(
        label="List /tmp",
        payload="List files in /tmp",
    ),
    TaskSpec(
        label="Palindrome checker",
        payload="Write a Python function to check if a string is a palindrome",
    ),
]


# ── Pipeline builder ──────────────────────────────────────────────────────────

def build_router(api_key: str) -> TriageRouter:
    router = TriageRouter(
        triage_node_id="triager",
        circuit_failure_threshold=3,
        circuit_cooldown_seconds=30.0,
    )

    # Trust domains — triage feeds both execution and review
    router.add_trust_domain("triage",    "Triage",    allowed_peers=["execution", "review"])
    router.add_trust_domain("execution", "Execution", allowed_peers=["triage", "review"])
    router.add_trust_domain("review",    "Review",    allowed_peers=["triage", "execution"])

    # ── Triager ───────────────────────────────────────────────────────────────
    # Outputs a JSON classification that drives routing rules.
    triager = LLMNodeHandler(
        model=MODEL, api_key=api_key, base_url=BASE,
        node_capabilities=["triage"],
        system_prompt=(
            "You are a task triage agent. Classify the task and output ONLY a "
            "JSON object with these keys:\n"
            "  complexity   : 'trivial' | 'simple' | 'moderate' | 'complex'\n"
            "  needs_research  : true | false\n"
            "  needs_planning  : true | false\n"
            "  needs_code      : true | false\n"
            "  needs_tools     : true | false   (requires terminal/filesystem)\n"
            "  needs_terminal  : true | false   (requires shell commands)\n"
            "  reformulated    : <one clear sentence rephrasing the task>\n\n"
            "Rules:\n"
            "- 'trivial'/'simple' = factual lookup, one-liner, no design needed\n"
            "- needs_planning = architecture, design, multi-step solution\n"
            "- needs_research = requires factual knowledge retrieval\n"
            "- needs_code = must produce code\n"
            "- needs_tools/needs_terminal = must run shell commands or list files\n\n"
            "Output ONLY the JSON. No markdown fences, no explanation."
        ),
        max_tokens=256,
        temperature=0.0,
    )

    # ── Quick answer ──────────────────────────────────────────────────────────
    quick_answer = LLMNodeHandler(
        model=MODEL, api_key=api_key, base_url=BASE,
        node_capabilities=["answer", "quick"],
        system_prompt=(
            "You give concise, direct answers to simple factual questions. "
            "One to three sentences maximum. No preamble."
        ),
        max_tokens=128,
        temperature=0.0,
    )

    # ── Planner ───────────────────────────────────────────────────────────────
    planner = LLMNodeHandler(
        model=MODEL, api_key=api_key, base_url=BASE,
        node_capabilities=["plan", "design"],
        system_prompt=(
            "You are a technical planner. Given a task, produce a structured "
            "implementation plan: key components, data structures, edge cases, "
            "and a step-by-step approach. Be specific and concrete."
        ),
        max_tokens=512,
        temperature=0.3,
    )

    # ── Researcher ────────────────────────────────────────────────────────────
    researcher = LLMNodeHandler(
        model=MODEL, api_key=api_key, base_url=BASE,
        node_capabilities=["research", "knowledge"],
        system_prompt=(
            "You are a research specialist. Gather and summarize all relevant "
            "technical background, patterns, and prior art for the task. "
            "Be thorough and cite specific techniques where applicable."
        ),
        max_tokens=512,
        temperature=0.3,
    )

    # ── Executor (LLM) ────────────────────────────────────────────────────────
    executor = LLMNodeHandler(
        model=MODEL, api_key=api_key, base_url=BASE,
        node_capabilities=["execute", "implement", "code"],
        system_prompt=(
            "You are an implementation specialist. Given a plan and any research "
            "context, produce a complete, working implementation. For code tasks, "
            "write clean, type-annotated Python. For design tasks, write a "
            "detailed technical spec."
        ),
        max_tokens=1024,
        temperature=0.2,
    )

    # ── Hermes executor (with tools) ──────────────────────────────────────────
    hermes_executor = HermesNodeHandler(
        model=MODEL,
        node_capabilities=["execute", "terminal", "tools"],
        system_prompt=(
            "You are an execution agent with terminal access. "
            "Run the requested commands and report the output clearly."
        ),
        max_turns=3,
        toolsets=["terminal", "file"],
        timeout=60,
    )

    # ── Reviewer ──────────────────────────────────────────────────────────────
    reviewer = LLMNodeHandler(
        model=MODEL, api_key=api_key, base_url=BASE,
        node_capabilities=["review", "qa"],
        system_prompt=(
            "You are a senior reviewer. Review the implementation from previous "
            "steps. Check for correctness, completeness, edge cases, and clarity. "
            "Provide a brief verdict and any specific improvements needed."
        ),
        max_tokens=512,
        temperature=0.2,
    )

    # Register nodes with trust domains and cost hints
    router.add_node("triager",        triager,        trust_domain="triage",    cost_per_call=0.001, avg_latency_ms=800,  max_concurrency=4)
    router.add_node("quick_answer",   quick_answer,   trust_domain="execution", cost_per_call=0.001, avg_latency_ms=500,  max_concurrency=4)
    router.add_node("planner",        planner,        trust_domain="execution", cost_per_call=0.002, avg_latency_ms=1500, max_concurrency=2)
    router.add_node("researcher",     researcher,     trust_domain="execution", cost_per_call=0.002, avg_latency_ms=1500, max_concurrency=2)
    router.add_node("executor",       executor,       trust_domain="execution", cost_per_call=0.003, avg_latency_ms=2000, max_concurrency=2)
    router.add_node("hermes_executor",hermes_executor,trust_domain="execution", cost_per_call=0.005, avg_latency_ms=5000, max_concurrency=1)
    router.add_node("reviewer",       reviewer,       trust_domain="review",    cost_per_call=0.002, avg_latency_ms=1500, max_concurrency=3)

    # Peer links: planner and researcher share context with executor laterally
    router.add_peer_link("planner",    "executor")
    router.add_peer_link("researcher", "executor")

    # Load default deterministic routing rules
    router.add_default_rules()

    return router


# ── Display helpers ───────────────────────────────────────────────────────────

STATUS_ICON = {
    HopStatus.COMPLETE:    "+",
    HopStatus.FAILED:      "!",
    HopStatus.IN_PROGRESS: ">",
    HopStatus.SKIPPED:     "-",
    HopStatus.PENDING:     ".",
    HopStatus.RETRIED:     "R",
}

COLS = 72  # terminal width


def _sep(char: str = "─") -> str:
    return char * COLS


def _header(title: str, char: str = "═") -> str:
    pad = max(0, COLS - len(title) - 4)
    left = pad // 2
    right = pad - left
    return f"{char * left}  {title}  {char * right}"


def print_kanban(specs: list[TaskSpec], results, elapsed: float) -> None:
    """Compact kanban board: one row per task with status and route."""
    print()
    print(_header("KANBAN BOARD"))
    print(f"  {'#':<3}  {'Label':<22}  {'Route':<36}  {'Status':<8}  {'ms':>6}")
    print(f"  {_sep()}")

    for i, (spec, task) in enumerate(zip(specs, results), 1):
        route = " > ".join(task.path) if task.path else "(none)"
        if len(route) > 36:
            route = route[:33] + "..."

        failed = task.failed_hops
        ok_hops = [h for h in task.hops if h.status == HopStatus.COMPLETE]

        if failed:
            status = "FAILED"
        elif task.is_complete or len(ok_hops) == len(task.path):
            status = "DONE"
        else:
            status = "PARTIAL"

        total_ms = sum(h.duration_ms for h in task.hops)
        print(f"  {i:<3}  {spec.label:<22}  {route:<36}  {status:<8}  {total_ms:>6.0f}")

    print(f"  {_sep()}")
    print(f"  {len(specs)} tasks  |  wall time: {elapsed:.1f}s")
    print()


def print_detail(spec: TaskSpec, task, index: int) -> None:
    """Full detail view for one task."""
    print()
    print(_header(f"TASK {index}: {spec.label}", "─"))

    # Original payload
    print(f"  Payload : {spec.payload}")

    # Triage classification
    triage = task.metadata.get("triage")
    if triage:
        flags = [k for k in ("needs_research", "needs_planning", "needs_code",
                              "needs_tools", "needs_terminal") if triage.get(k)]
        print(f"  Triage  : complexity={triage.get('complexity','?')}  flags=[{', '.join(flags) or 'none'}]")
        if triage.get("reformulated"):
            print(f"  Reframe : {triage['reformulated']}")

    routing = task.metadata.get("routing", "unknown")
    print(f"  Routing : {routing}  |  path: {' > '.join(task.path)}")

    # Traceroute
    print()
    print("  Traceroute:")
    for hop in task.hops:
        icon = STATUS_ICON.get(hop.status, "?")
        extra = ""
        if hop.duration_ms:
            extra += f"  {hop.duration_ms:.0f}ms"
        if hop.tokens_used:
            extra += f"  {hop.tokens_used}tok"
        if hop.error:
            extra += f"  ERROR: {hop.error[:60]}"
        peer_note = f"  [+{hop.peer_messages_received} peer]" if hop.peer_messages_received else ""
        print(f"    [{icon}] {hop.node_id}{peer_note}{extra}")

    # Outputs per node
    print()
    print("  Outputs:")
    for node_id in task.path:
        raw = task.context.get(node_id, "")
        if not raw:
            continue
        text = str(raw).strip()
        # Truncate long outputs for display
        if len(text) > 400:
            text = text[:400] + " ...(truncated)"
        print(f"    [{node_id}]")
        for line in text.split("\n"):
            print(f"      {line}")

    # Failed hops
    for hop in task.failed_hops:
        print(f"  FAILED at {hop.node_id}: {hop.error}")


def print_flow_summary(router: TriageRouter) -> None:
    print()
    print(_header("FLOW SUMMARY"))
    for line in router.flow.flow_summary().split("\n"):
        print(f"  {line}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print()
    print(_header("scionic v0.2.0  ·  Task Manager", "═"))
    print(f"  Model    : {MODEL}")
    print(f"  Tasks    : {len(TASKS)}")
    print(f"  Strategy : TriageRouter  (deterministic-first)")
    print(f"  Dispatch : execute_batch (fully concurrent)")
    print()

    api_key = load_api_key()
    router = build_router(api_key)

    # ── Triage all tasks first to get routed Task objects ─────────────────────
    print("  [1/3] Triaging tasks ...")
    triage_start = time.time()

    triage_coros = [
        router.create_task_routed(
            payload=spec.payload,
            trust_domain=spec.trust_domain,
        )
        for spec in TASKS
    ]
    routed_tasks = await asyncio.gather(*triage_coros)

    triage_elapsed = time.time() - triage_start
    print(f"        done in {triage_elapsed:.1f}s")
    for spec, task in zip(TASKS, routed_tasks):
        routing = task.metadata.get("routing", "?")
        print(f"        {spec.label:<22}  {' > '.join(task.path)}  [{routing}]")

    # ── Execute all tasks concurrently ────────────────────────────────────────
    print()
    print("  [2/3] Executing all tasks concurrently ...")
    exec_start = time.time()

    results = await router.execute_batch(
        tasks=list(routed_tasks),
        concurrency=4,
        timeout_per_task=120.0,
    )

    exec_elapsed = time.time() - exec_start
    print(f"        done in {exec_elapsed:.1f}s")

    # ── Display ───────────────────────────────────────────────────────────────
    print()
    print("  [3/3] Rendering output ...")

    total_wall = triage_elapsed + exec_elapsed

    print_kanban(TASKS, results, total_wall)

    for i, (spec, task) in enumerate(zip(TASKS, results), 1):
        print_detail(spec, task, i)

    print_flow_summary(router)

    print(_header("done"))
    print()


if __name__ == "__main__":
    asyncio.run(main())
