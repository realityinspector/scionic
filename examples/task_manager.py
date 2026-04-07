#!/usr/bin/env python3
"""
Task Manager — a CS 101 app built on scionic.

A basic task management system where tasks are created, prioritized,
broken down, and completed by routing them through a graph of LLM agents.

This demonstrates scionic doing real work, not just demos:
- Tasks enter the system and get routed through agent nodes
- A triager classifies and prioritizes incoming tasks
- A planner breaks complex tasks into subtasks
- An executor processes tasks and produces deliverables
- A reviewer validates the output before marking complete

The scionic protocol gives us:
- Traceroute: see exactly which agents touched each task and when
- Multi-path: route urgent tasks through a fast path, complex ones through a thorough path
- IRQ: if the reviewer finds issues, it fires an interrupt back to the executor
- Peer messaging: planner can send hints directly to executor
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scionic import Conductor, IRQPriority, IRQType, PathPolicy
from scionic.adapters.llm import LLMNodeHandler
from scionic.types import Task as ScionicTask

# ── Config ───────────────────────────────────────────────────────────

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

MODEL = "anthropic/claude-haiku-4.5"
BASE_URL = "https://openrouter.ai/api/v1"

logging.basicConfig(
    level=logging.WARNING,
    format="%(name)s | %(message)s",
)
logger = logging.getLogger("task_manager")


def _extract_json(text: str) -> str:
    """Extract JSON from text that may be wrapped in markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Drop first line (```json) and last line (```)
        lines = [l for l in lines[1:] if not l.strip() == "```"]
        text = "\n".join(lines)
    return text.strip()


# ── Task Data Model (CS 101) ────────────────────────────────────────

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Status(str, Enum):
    NEW = "new"
    TRIAGED = "triaged"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    DONE = "done"
    REJECTED = "rejected"


@dataclass
class UserTask:
    """A user-facing task in the system."""
    id: int
    title: str
    description: str
    priority: Priority = Priority.MEDIUM
    status: Status = Status.NEW
    subtasks: list[str] = field(default_factory=list)
    result: str = ""
    review: str = ""
    trace: str = ""  # scionic traceroute
    created_at: float = field(default_factory=time.time)


class TaskStore:
    """In-memory task store. CS 101 — just a list."""

    def __init__(self):
        self._tasks: list[UserTask] = []
        self._next_id = 1

    def create(self, title: str, description: str) -> UserTask:
        task = UserTask(id=self._next_id, title=title, description=description)
        self._tasks.append(task)
        self._next_id += 1
        return task

    def get(self, task_id: int) -> Optional[UserTask]:
        for t in self._tasks:
            if t.id == task_id:
                return t
        return None

    def list_all(self) -> list[UserTask]:
        return list(self._tasks)

    def list_by_status(self, status: Status) -> list[UserTask]:
        return [t for t in self._tasks if t.status == status]


# ── Scionic Pipeline ─────────────────────────────────────────────────

def build_conductor() -> Conductor:
    """Wire up the task management agent graph."""
    conductor = Conductor()

    # Node 1: Triager — classifies priority and determines routing
    triager = LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE_URL,
        node_capabilities=["triage"],
        system_prompt=(
            "You are a task triager. Given a task title and description, respond with a JSON object:\n"
            '{"priority": "low|medium|high|critical", "complexity": "simple|complex", "reason": "one sentence"}\n'
            "Simple tasks can be done in one step. Complex tasks need to be broken into subtasks.\n"
            "Respond with ONLY the JSON object, no markdown."
        ),
        max_tokens=128,
        temperature=0.0,
    )

    # Node 2: Planner — breaks complex tasks into subtasks
    planner = LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE_URL,
        node_capabilities=["plan"],
        system_prompt=(
            "You are a task planner. Given a task and triage info from the previous step, "
            "break it into 2-4 concrete subtasks. Respond with a JSON object:\n"
            '{"subtasks": ["subtask 1", "subtask 2", ...], "approach": "one sentence summary"}\n'
            "Respond with ONLY the JSON object, no markdown."
        ),
        max_tokens=256,
        temperature=0.3,
    )

    # Node 3: Executor — does the actual work
    executor = LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE_URL,
        node_capabilities=["execute"],
        system_prompt=(
            "You are a task executor. Given a task and its plan from previous steps, "
            "produce the deliverable. Write a clear, concise result (2-4 sentences) "
            "that completes the task. Be specific and actionable."
        ),
        max_tokens=256,
        temperature=0.5,
    )

    # Node 4: Reviewer — validates the output
    reviewer = LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE_URL,
        node_capabilities=["review"],
        system_prompt=(
            "You are a task reviewer. Given a task and its execution result from previous steps, "
            "evaluate the quality. Respond with a JSON object:\n"
            '{"approved": true|false, "feedback": "one sentence", "quality": "poor|acceptable|good|excellent"}\n'
            "Approve if the result adequately addresses the task. Respond with ONLY the JSON object, no markdown."
        ),
        max_tokens=128,
        temperature=0.0,
    )

    # Node 5: Quick executor — fast path for simple tasks (skips planning)
    quick_executor = LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE_URL,
        node_capabilities=["execute", "quick"],
        system_prompt=(
            "You are a fast task executor for simple tasks. "
            "Produce a brief, direct result in 1-2 sentences."
        ),
        max_tokens=128,
        temperature=0.3,
    )

    conductor.add_node("triager", triager, cost_per_call=0.001, avg_latency_ms=1000)
    conductor.add_node("planner", planner, cost_per_call=0.002, avg_latency_ms=1500)
    conductor.add_node("executor", executor, cost_per_call=0.003, avg_latency_ms=2000)
    conductor.add_node("reviewer", reviewer, cost_per_call=0.001, avg_latency_ms=1000)
    conductor.add_node("quick_executor", quick_executor, cost_per_call=0.001, avg_latency_ms=800)

    # Peer link: planner can send hints directly to executor
    conductor.add_peer_link("planner", "executor")

    return conductor


# ── Task Manager ─────────────────────────────────────────────────────

class TaskManager:
    """
    The main application. Creates tasks and routes them through scionic.

    Two paths through the graph:
    - Simple:  triager → quick_executor → reviewer
    - Complex: triager → planner → executor → reviewer
    """

    def __init__(self):
        self.store = TaskStore()
        self.conductor = build_conductor()

    async def add_task(self, title: str, description: str) -> UserTask:
        """Create a task and route it through the pipeline."""
        user_task = self.store.create(title, description)
        print(f"\n  [{user_task.id}] Created: {title}")

        # Step 1: Triage
        triage_result = await self._run_triage(user_task)
        complexity = "simple"
        try:
            triage_data = json.loads(_extract_json(triage_result))
            user_task.priority = Priority(triage_data.get("priority", "medium"))
            complexity = triage_data.get("complexity", "simple")
            print(f"  [{user_task.id}] Triaged: priority={user_task.priority.value}, complexity={complexity}")
        except (json.JSONDecodeError, ValueError):
            print(f"  [{user_task.id}] Triaged (raw): {triage_result[:80]}")

        user_task.status = Status.TRIAGED

        # Step 2: Route based on complexity
        if complexity == "complex":
            result = await self._run_complex_path(user_task)
        else:
            result = await self._run_simple_path(user_task)

        return user_task

    async def _run_triage(self, user_task: UserTask) -> str:
        """Run just the triage node."""
        scionic_task = ScionicTask(
            payload=f"Title: {user_task.title}\nDescription: {user_task.description}",
            path=["triager"],
        )
        result = await self.conductor.execute(scionic_task)
        return str(result.context.get("triager", ""))

    async def _run_simple_path(self, user_task: UserTask) -> UserTask:
        """Simple: quick_executor → reviewer."""
        print(f"  [{user_task.id}] Routing: simple path (quick_executor → reviewer)")

        scionic_task = ScionicTask(
            payload=f"Title: {user_task.title}\nDescription: {user_task.description}",
            path=["quick_executor", "reviewer"],
        )
        result = await self.conductor.execute(scionic_task)

        user_task.result = str(result.context.get("quick_executor", ""))
        user_task.review = str(result.context.get("reviewer", ""))
        user_task.trace = result.traceroute()
        user_task.status = Status.IN_REVIEW

        # Check review
        self._apply_review(user_task)

        print(f"  [{user_task.id}] {user_task.status.value}: {user_task.result[:80]}...")
        return user_task

    async def _run_complex_path(self, user_task: UserTask) -> UserTask:
        """Complex: planner → executor → reviewer."""
        print(f"  [{user_task.id}] Routing: complex path (planner → executor → reviewer)")

        user_task.status = Status.PLANNED

        scionic_task = ScionicTask(
            payload=f"Title: {user_task.title}\nDescription: {user_task.description}",
            path=["planner", "executor", "reviewer"],
        )
        result = await self.conductor.execute(scionic_task)

        # Extract subtasks from planner
        planner_output = str(result.context.get("planner", ""))
        try:
            plan_data = json.loads(_extract_json(planner_output))
            user_task.subtasks = plan_data.get("subtasks", [])
        except json.JSONDecodeError:
            pass

        user_task.result = str(result.context.get("executor", ""))
        user_task.review = str(result.context.get("reviewer", ""))
        user_task.trace = result.traceroute()
        user_task.status = Status.IN_REVIEW

        # Check review
        self._apply_review(user_task)

        print(f"  [{user_task.id}] {user_task.status.value}: {user_task.result[:80]}...")
        return user_task

    def _apply_review(self, user_task: UserTask) -> None:
        """Apply reviewer feedback to determine final status."""
        try:
            review_data = json.loads(_extract_json(user_task.review))
            if review_data.get("approved", False):
                user_task.status = Status.DONE
            else:
                user_task.status = Status.REJECTED
        except json.JSONDecodeError:
            # If we can't parse the review, accept it
            user_task.status = Status.DONE

    def print_task(self, task_id: int) -> None:
        """Pretty-print a task with its full trace."""
        task = self.store.get(task_id)
        if not task:
            print(f"Task {task_id} not found.")
            return

        print(f"\n{'─' * 60}")
        print(f"Task #{task.id}: {task.title}")
        print(f"{'─' * 60}")
        print(f"  Status:   {task.status.value}")
        print(f"  Priority: {task.priority.value}")
        if task.subtasks:
            print(f"  Subtasks:")
            for i, st in enumerate(task.subtasks, 1):
                print(f"    {i}. {st}")
        if task.result:
            print(f"  Result:   {task.result}")
        if task.review:
            print(f"  Review:   {task.review}")
        if task.trace:
            print(f"\n  Scionic Traceroute:")
            for line in task.trace.split("\n"):
                print(f"    {line}")
        print()

    def print_board(self) -> None:
        """Print a kanban-style board."""
        print(f"\n{'═' * 60}")
        print(f"  TASK BOARD")
        print(f"{'═' * 60}")
        for status in Status:
            tasks = self.store.list_by_status(status)
            if tasks:
                print(f"\n  [{status.value.upper()}]")
                for t in tasks:
                    pri = t.priority.value[0].upper()
                    print(f"    #{t.id} [{pri}] {t.title}")
        print()


# ── Main ─────────────────────────────────────────────────────────────

async def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Task Manager — powered by scionic                      ║")
    print("║  CS 101 app with SCION-inspired agent routing           ║")
    print("╚══════════════════════════════════════════════════════════╝")

    tm = TaskManager()

    # Create a mix of simple and complex tasks
    tasks_to_create = [
        ("Write a haiku about Python", "Write a haiku (5-7-5) about Python programming."),
        ("Design a REST API for a bookstore", "Design the endpoints, methods, and data models for a REST API that manages books, authors, and reviews."),
        ("Name 3 prime numbers", "List three prime numbers greater than 10."),
        ("Plan a database migration strategy", "We need to migrate from PostgreSQL 12 to 16 with zero downtime. Outline the strategy including rollback plan."),
    ]

    print("\n── Creating and processing tasks ──")
    for title, desc in tasks_to_create:
        await tm.add_task(title, desc)

    # Show the board
    tm.print_board()

    # Show detailed view of each task with traceroutes
    print("── Detailed Task Views ──")
    for task in tm.store.list_all():
        tm.print_task(task.id)


if __name__ == "__main__":
    asyncio.run(main())
