#!/usr/bin/env python3
"""
Task Manager — built on scionic.

Not a toy. A task management system where:
- SmartConductor decides how to route each task (LLM picks the path)
- Hermes nodes can use tools (terminal, file, web) for real work
- LLM nodes handle classification, planning, review
- Reviewer can reject and fire IRQ to retry with feedback
- Trust domains separate triage (cheap/fast) from execution (expensive/capable)
- Peer messaging lets planner hint to executor directly
- Every task gets a cryptographically signed traceroute

Run: python examples/task_manager.py
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

from scionic import SmartConductor, IRQPriority, IRQType, HopStatus
from scionic.adapters.llm import LLMNodeHandler
from scionic.adapters.hermes import HermesNodeHandler
from scionic.transport import task_to_json

MODEL = "anthropic/claude-haiku-4.5"
BASE = "https://openrouter.ai/api/v1"


# ── Data model ───────────────────────────────────────────────────────

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Status(str, Enum):
    NEW = "new"
    ROUTED = "routed"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    DONE = "done"
    REJECTED = "rejected"


@dataclass
class UserTask:
    id: int
    title: str
    description: str
    priority: Priority = Priority.MEDIUM
    status: Status = Status.NEW
    route: list[str] = field(default_factory=list)  # scionic path chosen
    subtasks: list[str] = field(default_factory=list)
    result: str = ""
    review: str = ""
    trace: str = ""
    trace_json: str = ""  # serialized via transport layer
    retries: int = 0
    created_at: float = field(default_factory=time.time)
    elapsed_ms: float = 0.0


class TaskStore:
    def __init__(self):
        self._tasks: list[UserTask] = []
        self._next_id = 1

    def create(self, title: str, description: str) -> UserTask:
        t = UserTask(id=self._next_id, title=title, description=description)
        self._tasks.append(t)
        self._next_id += 1
        return t

    def get(self, task_id: int) -> Optional[UserTask]:
        return next((t for t in self._tasks if t.id == task_id), None)

    def list_all(self) -> list[UserTask]:
        return list(self._tasks)


def _extract_json(text: str) -> str:
    """Extract JSON from text that may be wrapped in markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines[1:] if l.strip() != "```"]
        text = "\n".join(lines)
    return text.strip()


# ── Scionic graph ────────────────────────────────────────────────────

def build_conductor() -> SmartConductor:
    """Build the task management agent graph."""
    c = SmartConductor(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        verify_signatures=True,
        auto_reroute=True,
        max_retries=1,
    )

    # Trust domains
    c.add_trust_domain("triage", "Triage", allowed_peers=["execution", "review"])
    c.add_trust_domain("execution", "Execution", allowed_peers=["triage", "review"])
    c.add_trust_domain("review", "Review", allowed_peers=["execution", "triage"])

    # ── Triage tier (cheap, fast) ────────────────────────────────────

    c.add_node("triager", LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        node_capabilities=["triage", "classify"],
        system_prompt=(
            "You are a task triager. Given a task, respond with JSON:\n"
            '{"priority": "low|medium|high|critical", '
            '"needs_planning": true|false, '
            '"needs_research": true|false, '
            '"needs_code": true|false, '
            '"reason": "one sentence"}\n'
            "ONLY the JSON, no markdown."
        ),
        max_tokens=128, temperature=0.0,
    ), trust_domain="triage", cost_per_call=0.001, avg_latency_ms=1000)

    # ── Execution tier ───────────────────────────────────────────────

    c.add_node("planner", LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        node_capabilities=["plan", "decompose"],
        system_prompt=(
            "You are a task planner. Break the task into 2-4 concrete subtasks.\n"
            "Check context from previous steps for triage info.\n"
            "Respond with JSON:\n"
            '{"subtasks": ["step 1", "step 2", ...], "approach": "one sentence"}\n'
            "ONLY the JSON, no markdown."
        ),
        max_tokens=256, temperature=0.3,
    ), trust_domain="execution", cost_per_call=0.002, avg_latency_ms=1500)

    c.add_node("researcher", LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        node_capabilities=["research", "search", "rag"],
        system_prompt=(
            "You are a researcher. Investigate the topic and provide "
            "key facts and findings. 3-5 bullet points. Be specific."
        ),
        max_tokens=256, temperature=0.3,
    ), trust_domain="execution", cost_per_call=0.002, avg_latency_ms=2000)

    c.add_node("executor", LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        node_capabilities=["execute", "write", "produce"],
        system_prompt=(
            "You are a task executor. Using context from previous steps "
            "(triage, planning, research), produce the deliverable.\n"
            "Check for peer messages — they contain direct hints.\n"
            "Check for _feedback_for_executor — it means you're retrying.\n"
            "Be thorough but concise. 3-5 sentences."
        ),
        max_tokens=384, temperature=0.5,
    ), trust_domain="execution", cost_per_call=0.003, avg_latency_ms=2500)

    # Hermes executor — for tasks that need tools
    c.add_node("hermes_executor", HermesNodeHandler(
        model=MODEL,
        node_capabilities=["execute", "code", "terminal"],
        system_prompt=(
            "You are a task executor with tool access. "
            "Use your tools if needed (terminal, file, web). "
            "Produce a clear, concise result."
        ),
        max_turns=3,
        toolsets=["terminal", "file"],
    ), trust_domain="execution", cost_per_call=0.005, avg_latency_ms=5000)

    c.add_node("quick_answer", LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        node_capabilities=["execute", "answer", "quick"],
        system_prompt="Answer the question directly. 1-2 sentences.",
        max_tokens=64, temperature=0.0,
    ), trust_domain="execution", cost_per_call=0.001, avg_latency_ms=800)

    # ── Review tier ──────────────────────────────────────────────────

    c.add_node("reviewer", LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        node_capabilities=["review", "approve"],
        system_prompt=(
            "You are a task reviewer. Evaluate the execution result.\n"
            "Respond with JSON:\n"
            '{"approved": true|false, '
            '"quality": "poor|acceptable|good|excellent", '
            '"feedback": "one sentence — if rejecting, say what to fix"}\n'
            "ONLY the JSON, no markdown."
        ),
        max_tokens=128, temperature=0.0,
    ), trust_domain="review", cost_per_call=0.001, avg_latency_ms=1000)

    # ── Peer links ───────────────────────────────────────────────────

    c.add_peer_link("planner", "executor")
    c.add_peer_link("planner", "hermes_executor")
    c.add_peer_link("researcher", "executor")

    return c


# ── Task Manager ─────────────────────────────────────────────────────

class TaskManager:
    def __init__(self):
        self.store = TaskStore()
        self.conductor = build_conductor()

    async def add(self, title: str, description: str) -> UserTask:
        """Create and process a task through the scionic graph."""
        ut = self.store.create(title, description)
        start = time.time()
        print(f"\n  [{ut.id}] Created: {title}")

        # SmartConductor picks the path based on available nodes
        try:
            scionic_task = self.conductor.create_task(
                payload=f"Title: {title}\nDescription: {description}",
            )
            ut.route = list(scionic_task.path)
            ut.status = Status.ROUTED
            print(f"  [{ut.id}] Routed: {' → '.join(ut.route)}")

        except Exception as e:
            # Fallback: manual path
            print(f"  [{ut.id}] Smart routing failed ({e}), using fallback")
            scionic_task = self.conductor.create_task(
                payload=f"Title: {title}\nDescription: {description}",
                path=["triager", "executor", "reviewer"],
            )
            ut.route = list(scionic_task.path)
            ut.status = Status.ROUTED
            print(f"  [{ut.id}] Fallback route: {' → '.join(ut.route)}")

        # Execute with retry support
        ut.status = Status.IN_PROGRESS
        result = await self.conductor.execute_with_retry(scionic_task)

        # Extract results from context
        self._extract_results(ut, result)
        ut.trace = result.traceroute()
        ut.trace_json = task_to_json(result)
        ut.elapsed_ms = (time.time() - start) * 1000
        ut.retries = result.retry_count

        # Check review
        if "reviewer" in result.context:
            self._apply_review(ut, result)
        else:
            ut.status = Status.DONE

        pri = ut.priority.value[0].upper()
        print(f"  [{ut.id}] {ut.status.value} [{pri}] ({ut.elapsed_ms:.0f}ms, {len(result.hops)} hops)")

        return ut

    def _extract_results(self, ut: UserTask, result) -> None:
        """Pull structured data from scionic context."""
        # Triage
        triage_out = result.context.get("triager", "")
        if triage_out:
            try:
                td = json.loads(_extract_json(str(triage_out)))
                ut.priority = Priority(td.get("priority", "medium"))
            except (json.JSONDecodeError, ValueError):
                pass

        # Plan
        plan_out = result.context.get("planner", "")
        if plan_out:
            try:
                pd = json.loads(_extract_json(str(plan_out)))
                ut.subtasks = pd.get("subtasks", [])
            except (json.JSONDecodeError, ValueError):
                pass

        # Execution result — take from whichever executor ran
        for exec_node in ["executor", "hermes_executor", "quick_answer"]:
            if exec_node in result.context:
                ut.result = str(result.context[exec_node])
                break

        # Review
        if "reviewer" in result.context:
            ut.review = str(result.context["reviewer"])

    def _apply_review(self, ut: UserTask, result) -> None:
        """Apply reviewer verdict. If rejected, request retry via IRQ."""
        try:
            rd = json.loads(_extract_json(ut.review))
            if rd.get("approved", True):
                ut.status = Status.DONE
            else:
                ut.status = Status.REJECTED
                # Could fire IRQ retry here for a second pass
        except json.JSONDecodeError:
            ut.status = Status.DONE

    def print_task(self, task_id: int) -> None:
        t = self.store.get(task_id)
        if not t:
            print(f"Task {task_id} not found.")
            return

        print(f"\n{'─' * 60}")
        print(f"  Task #{t.id}: {t.title}")
        print(f"{'─' * 60}")
        print(f"  Status:   {t.status.value}")
        print(f"  Priority: {t.priority.value}")
        print(f"  Route:    {' → '.join(t.route)}")
        print(f"  Time:     {t.elapsed_ms:.0f}ms")
        if t.retries:
            print(f"  Retries:  {t.retries}")
        if t.subtasks:
            print(f"  Subtasks:")
            for i, st in enumerate(t.subtasks, 1):
                print(f"    {i}. {st}")
        if t.result:
            print(f"  Result:")
            for line in t.result.split("\n"):
                print(f"    {line}")
        if t.review:
            print(f"  Review:   {t.review}")
        if t.trace:
            print(f"\n  Traceroute:")
            for line in t.trace.split("\n"):
                print(f"    {line}")
        print()

    def print_board(self) -> None:
        print(f"\n{'═' * 60}")
        print(f"  TASK BOARD")
        print(f"{'═' * 60}")
        for status in Status:
            tasks = [t for t in self.store.list_all() if t.status == status]
            if tasks:
                print(f"\n  [{status.value.upper()}]")
                for t in tasks:
                    pri = t.priority.value[0].upper()
                    route_summary = f"{t.route[0]}→...→{t.route[-1]}" if len(t.route) > 2 else "→".join(t.route)
                    print(f"    #{t.id} [{pri}] {t.title}  ({route_summary}, {t.elapsed_ms:.0f}ms)")
        print()


# ── Main ─────────────────────────────────────────────────────────────

async def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Task Manager — powered by scionic                      ║")
    print("║  SmartConductor + Hermes + trust domains + IRQ retry    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    tm = TaskManager()

    print(f"\n  Topology:")
    print(f"  {tm.conductor.topology_summary()}")

    tasks = [
        ("What is the capital of Japan?",
         "Simple factual question."),
        ("Design a rate limiter for an API",
         "Design a token bucket rate limiter with sliding window. Include data structures and algorithm."),
        ("List files in /tmp",
         "Use the terminal to list files in /tmp and report the count."),
        ("Write a Python function to check if a string is a palindrome",
         "Write the function with type hints. Handle edge cases like empty string and case sensitivity."),
    ]

    print("\n── Processing tasks (SmartConductor picks routes) ──")
    for title, desc in tasks:
        await tm.add(title, desc)

    tm.print_board()

    print("── Detailed Views ──")
    for t in tm.store.list_all():
        tm.print_task(t.id)


if __name__ == "__main__":
    asyncio.run(main())
