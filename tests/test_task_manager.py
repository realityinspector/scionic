#!/usr/bin/env python3
"""
Integration tests for the scionic task manager.

Real API calls to OpenRouter. No mocks.
"""

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load API key
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

# Import task manager after key is available (it loads key at import)
os.environ["OPENROUTER_API_KEY"] = API_KEY
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))
from task_manager import TaskManager, Priority, Status

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


async def test_simple_task_routes_through_two_hops():
    """A simple task should go: quick_executor → reviewer (2 hops)."""
    tm = TaskManager()
    task = await tm.add_task(
        "What is 2+2?",
        "Answer this arithmetic question."
    )

    report("task created", task.id == 1)
    report("task has status", task.status in (Status.DONE, Status.REJECTED),
           f"status={task.status.value}")
    report("task has result", len(task.result) > 0, "empty result")
    report("result contains 4", "4" in task.result, f"result={task.result!r}")
    report("task has review", len(task.review) > 0, "empty review")
    report("task has traceroute", len(task.trace) > 0, "empty trace")
    report("traceroute shows quick_executor", "quick_executor" in task.trace, task.trace)
    report("traceroute shows reviewer", "reviewer" in task.trace, task.trace)
    # Should NOT have planner or executor (those are for complex path)
    report("no planner in trace", "planner" not in task.trace, task.trace)


async def test_complex_task_routes_through_three_hops():
    """A complex task should go: planner → executor → reviewer (3 hops)."""
    tm = TaskManager()
    task = await tm.add_task(
        "Design a microservices architecture",
        "Design a microservices architecture for an e-commerce platform with user auth, product catalog, cart, and payment services. Include inter-service communication patterns."
    )

    report("task created", task.id == 1)
    report("task is done or rejected", task.status in (Status.DONE, Status.REJECTED),
           f"status={task.status.value}")
    report("priority is high or critical",
           task.priority in (Priority.HIGH, Priority.CRITICAL),
           f"priority={task.priority.value}")
    report("has subtasks", len(task.subtasks) > 0, "no subtasks generated")
    report("subtasks are meaningful",
           any(len(s) > 10 for s in task.subtasks) if task.subtasks else False,
           f"subtasks={task.subtasks}")
    report("has result", len(task.result) > 0, "empty result")
    report("has review", len(task.review) > 0, "empty review")
    report("traceroute shows planner", "planner" in task.trace, task.trace)
    report("traceroute shows executor", "executor" in task.trace, task.trace)
    report("traceroute shows reviewer", "reviewer" in task.trace, task.trace)


async def test_task_store_operations():
    """Basic CRUD operations on the task store."""
    tm = TaskManager()

    # Create two tasks with concrete descriptions
    t1 = await tm.add_task("Capital of France", "What is the capital of France?")
    t2 = await tm.add_task("Largest planet", "What is the largest planet in our solar system?")

    report("two tasks in store", len(tm.store.list_all()) == 2,
           f"got {len(tm.store.list_all())}")
    report("task 1 retrievable", tm.store.get(1) is not None)
    report("task 2 retrievable", tm.store.get(2) is not None)
    report("task 3 returns None", tm.store.get(3) is None)
    report("both tasks are done",
           all(t.status == Status.DONE for t in tm.store.list_all()),
           f"statuses={[t.status.value for t in tm.store.list_all()]}")


async def test_traceroute_has_timing_and_tokens():
    """Verify traceroute captures real timing and token data."""
    tm = TaskManager()
    task = await tm.add_task("Say hello", "Just say hello world.")

    # Parse traceroute for timing info
    report("trace has ms timing", "ms" in task.trace, task.trace)
    report("trace has token count", "tokens" in task.trace, task.trace)
    report("trace shows completion markers", "[+]" in task.trace, task.trace)


async def main():
    print("=" * 60)
    print("Task Manager integration tests (REAL API calls)")
    print("=" * 60)

    tests = [
        ("Simple task routing", test_simple_task_routes_through_two_hops),
        ("Complex task routing", test_complex_task_routes_through_three_hops),
        ("Task store CRUD", test_task_store_operations),
        ("Traceroute timing/tokens", test_traceroute_has_timing_and_tokens),
    ]

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        start = time.time()
        try:
            await test_fn()
        except Exception as e:
            report(name, False, f"EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
        elapsed = time.time() - start
        print(f"  ({elapsed:.1f}s)")

    print(f"\n{'=' * 60}")
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 60}")
    sys.exit(1 if FAILED > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
