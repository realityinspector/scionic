#!/usr/bin/env python3
"""
Task Manager — a multi-agent task routing app built on scionic.

Routes incoming tasks through a graph of specialized nodes:
  triager       → determines task complexity and type
  quick_answer  → handles simple factual questions directly
  planner       → breaks complex tasks into subtasks
  researcher    → gathers information / context
  executor      → implements / executes the plan
  hermes_executor → executor variant with real terminal tools (Hermes)
  reviewer      → validates and polishes output

Trust domains:
  triage  — triager, quick_answer          (low-trust input zone)
  ops     — planner, researcher, executor  (operation zone, peers triage)
  review  — reviewer                       (isolated review zone, peers ops)

Peer links:
  planner    <-> executor       (plan flows directly to executor)
  researcher <-> executor       (research context flows to executor)

IRQ retry:
  reviewer can request a retry of executor with feedback if output is weak.

Demonstrates:
  SmartConductor (LLM picks path per task)
  Hermes node (hermes_executor with terminal toolset)
  LLM nodes (all other nodes)
  Trust domains with cross-domain peering
  Peer links: planner→executor, researcher→executor
  IRQ retry: reviewer requests executor re-run
  Kanban board + detailed traceroute views
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Load API key from ~/.hermes/.env ─────────────────────────────────────────

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY=") and not line.startswith("#"):
                    API_KEY = line.split("=", 1)[1].strip()
                    break

if not API_KEY:
    print("ERROR: OPENROUTER_API_KEY not found in environment or ~/.hermes/.env")
    sys.exit(1)

from scionic import SmartConductor, IRQPriority, IRQType
from scionic.adapters.llm import LLMNodeHandler
from scionic.adapters.hermes import HermesNodeHandler

MODEL = "anthropic/claude-haiku-4.5"
BASE  = "https://openrouter.ai/api/v1"


# ── Task record ───────────────────────────────────────────────────────────────

@dataclass
class ManagedTask:
    id: str
    title: str
    description: str
    status: str = "pending"    # pending | running | done | failed
    result_task: Optional[object] = None
    started_at: float = 0.0
    finished_at: float = 0.0
    routed_path: Optional[list] = None


# ── Build the conductor ───────────────────────────────────────────────────────

def build_conductor() -> SmartConductor:
    c = SmartConductor(
        model=MODEL,
        api_key=API_KEY,
        base_url=BASE,
        verify_signatures=True,
        auto_reroute=True,
        max_retries=1,
    )

    # ── Trust domains ────────────────────────────────────────────────────────
    #
    #   triage  peers with ops     → triager can hand off to planner/executor
    #   ops     peers with triage  → bidirectional
    #   ops     peers with review  → executor output flows to reviewer
    #   review  peers with ops     → reviewer can request retry of executor
    #
    c.add_trust_domain("triage", "Triage",
                       description="Low-trust input zone. Accepts raw tasks.",
                       allowed_peers=["ops"])
    c.add_trust_domain("ops", "Operations",
                       description="Planning, research, and execution zone.",
                       allowed_peers=["triage", "review"])
    c.add_trust_domain("review", "Review",
                       description="Quality gate. Validates final output.",
                       allowed_peers=["ops"])

    # ── Node 1: triager ───────────────────────────────────────────────────────
    # Classifies the task and annotates with type/complexity.
    triager = LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        node_capabilities=["triage", "classify"],
        system_prompt=(
            "You are a task triager. Analyze the task and produce a brief triage report.\n\n"
            "Output JSON with keys:\n"
            "  type: one of [factual, design, terminal, coding, analysis]\n"
            "  complexity: one of [simple, moderate, complex]\n"
            "  requires_terminal: true|false\n"
            "  summary: one sentence description of what needs to be done\n\n"
            "ONLY output the JSON object, no markdown fences."
        ),
        max_tokens=256,
        temperature=0.0,
    )

    # ── Node 2: quick_answer ──────────────────────────────────────────────────
    # Handles simple factual questions directly — no planning or research needed.
    quick_answer = LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        node_capabilities=["quick_answer", "factual"],
        system_prompt=(
            "You are a fast factual answering agent. "
            "Answer the question concisely and accurately. "
            "Keep answers under 3 sentences for simple facts. "
            "You may see triage context — use it to frame your answer."
        ),
        max_tokens=256,
        temperature=0.1,
    )

    # ── Node 3: planner ───────────────────────────────────────────────────────
    # Breaks complex tasks into a numbered execution plan.
    planner = LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        node_capabilities=["plan", "decompose"],
        system_prompt=(
            "You are a task planner. Given a task and triage classification,\n"
            "produce a clear, numbered execution plan.\n\n"
            "Rules:\n"
            "- Be specific and actionable\n"
            "- Each step should be independently executable\n"
            "- 3-6 steps maximum\n"
            "- Include any key considerations or edge cases\n\n"
            "Format: numbered list, each step on its own line."
        ),
        max_tokens=512,
        temperature=0.3,
    )

    # ── Node 4: researcher ────────────────────────────────────────────────────
    # Gathers relevant context, patterns, and background.
    researcher = LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        node_capabilities=["research", "context"],
        system_prompt=(
            "You are a research specialist. For the given task:\n"
            "1. Identify the key concepts and patterns involved\n"
            "2. Recall relevant best practices, algorithms, or approaches\n"
            "3. Note any common pitfalls or edge cases\n"
            "4. Provide a concise knowledge brief that will help the executor\n\n"
            "Keep your research focused and directly applicable. "
            "This goes directly to the executor via peer link."
        ),
        max_tokens=512,
        temperature=0.3,
    )

    # ── Node 5: executor ──────────────────────────────────────────────────────
    # Implements the solution based on plan + research context.
    executor = LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        node_capabilities=["execute", "implement", "code"],
        system_prompt=(
            "You are a skilled implementer. You receive:\n"
            "- The original task\n"
            "- A triage classification\n"
            "- An execution plan (from planner)\n"
            "- Research context (from researcher, via peer link)\n"
            "- Any retry feedback in context\n\n"
            "Execute the plan and deliver a complete, high-quality result.\n"
            "For coding tasks: include working code with explanations.\n"
            "For design tasks: include concrete specifications and rationale.\n"
            "For analysis tasks: include findings with supporting reasoning.\n\n"
            "Check context for _feedback_for_executor — if present, incorporate it."
        ),
        max_tokens=1024,
        temperature=0.4,
    )

    # ── Node 6: hermes_executor ───────────────────────────────────────────────
    # Like executor but with real terminal tools via Hermes.
    # Used when requires_terminal=true (e.g., 'list files in /tmp').
    hermes_executor = HermesNodeHandler(
        model=MODEL,
        node_capabilities=["execute", "terminal", "file_ops"],
        system_prompt=(
            "You are an executor with terminal access. "
            "Complete the task using available tools. "
            "For file/directory operations, use the terminal tool. "
            "Report exactly what you find or do, including command outputs."
        ),
        max_turns=3,
        toolsets=["terminal", "file"],
        timeout=60,
    )

    # ── Node 7: reviewer ──────────────────────────────────────────────────────
    # Validates and polishes the final output.
    reviewer = LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        node_capabilities=["review", "validate", "quality"],
        system_prompt=(
            "You are a quality reviewer. Review the executor's output against "
            "the original task.\n\n"
            "Evaluate:\n"
            "- Correctness: does it solve what was asked?\n"
            "- Completeness: are there gaps or missing pieces?\n"
            "- Quality: is it well-structured and clear?\n\n"
            "Output JSON:\n"
            "  approved: true|false\n"
            "  score: 1-10\n"
            "  feedback: specific improvement notes (empty string if approved)\n"
            "  final_answer: the polished final answer (copy or improve from executor)\n\n"
            "ONLY output the JSON object, no markdown fences."
        ),
        max_tokens=1024,
        temperature=0.1,
    )

    # ── Register nodes ────────────────────────────────────────────────────────
    c.add_node("triager",         triager,         trust_domain="triage",  cost_per_call=0.001, avg_latency_ms=800)
    c.add_node("quick_answer",    quick_answer,     trust_domain="triage",  cost_per_call=0.001, avg_latency_ms=600)
    c.add_node("planner",         planner,          trust_domain="ops",     cost_per_call=0.002, avg_latency_ms=1200)
    c.add_node("researcher",      researcher,       trust_domain="ops",     cost_per_call=0.002, avg_latency_ms=1500)
    c.add_node("executor",        executor,         trust_domain="ops",     cost_per_call=0.004, avg_latency_ms=2000)
    c.add_node("hermes_executor", hermes_executor,  trust_domain="ops",     cost_per_call=0.003, avg_latency_ms=5000)
    c.add_node("reviewer",        reviewer,         trust_domain="review",  cost_per_call=0.002, avg_latency_ms=1000)

    # ── Peer links ────────────────────────────────────────────────────────────
    # planner  → executor  : plan details flow directly, no conductor hop
    # researcher → executor : research brief arrives at executor before it runs
    c.add_peer_link("planner",    "executor")
    c.add_peer_link("researcher", "executor")
    # reviewer ↔ executor  : reviewer can signal executor for retry context
    c.add_peer_link("reviewer",   "executor")

    return c


# ── Path selection heuristic ──────────────────────────────────────────────────
# SmartConductor uses the LLM to pick the path, but we also have a fallback
# heuristic in case the LLM routing produces an incomplete path (e.g. skips
# reviewer).  We force a minimum sensible path based on triage output.

def derive_fallback_path(triage_json: dict) -> list[str]:
    """Derive a path from triage metadata."""
    task_type  = triage_json.get("type", "coding")
    complexity = triage_json.get("complexity", "moderate")
    needs_term = triage_json.get("requires_terminal", False)

    if complexity == "simple" and task_type == "factual":
        return ["triager", "quick_answer", "reviewer"]

    if needs_term:
        return ["triager", "planner", "researcher", "hermes_executor", "reviewer"]

    return ["triager", "planner", "researcher", "executor", "reviewer"]


# ── Display helpers ───────────────────────────────────────────────────────────

COL_WIDTH = 22

STATUS_ICON = {
    "pending": "◇",
    "running": "◎",
    "done":    "✔",
    "failed":  "✖",
}

def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n-1] + "…"


def print_kanban(tasks: list[ManagedTask]) -> None:
    cols = {
        "pending": [],
        "running": [],
        "done":    [],
        "failed":  [],
    }
    for t in tasks:
        cols[t.status].append(t)

    w = COL_WIDTH
    sep = "─" * w
    print()
    print("╔" + ("═" * w + "╦") * 3 + "═" * w + "╗")
    headers = ["  PENDING", "  RUNNING", "  DONE", "  FAILED"]
    row = "║"
    for h in headers:
        row += h.ljust(w) + "║"
    print(row)
    print("╠" + (sep + "╬") * 3 + sep + "╣")

    max_rows = max(len(v) for v in cols.values()) if cols else 0
    for i in range(max_rows):
        row = "║"
        for status in ("pending", "running", "done", "failed"):
            items = cols[status]
            if i < len(items):
                t = items[i]
                icon = STATUS_ICON[t.status]
                cell = f" {icon} {_truncate(t.title, w - 4)}"
                row += cell.ljust(w) + "║"
            else:
                row += " " * w + "║"
        print(row)

    if max_rows == 0:
        print("║" + " " * w + "║" + " " * w + "║" + " " * w + "║" + " " * w + "║")

    print("╚" + (sep + "╩") * 3 + sep + "╝")
    print()


def print_task_detail(mt: ManagedTask) -> None:
    w = 64
    print()
    print("╔" + "═" * w + "╗")
    print(f"║  Task: {_truncate(mt.title, w - 8).ljust(w - 8)}  ║")
    print(f"║  Status: {mt.status.upper().ljust(w - 10)}  ║")
    print("╠" + "═" * w + "╣")

    # Description
    desc_lines = [mt.description[i:i+w-4] for i in range(0, len(mt.description), w-4)]
    for line in desc_lines[:3]:
        print(f"║  {line.ljust(w - 4)}  ║")

    rt = mt.result_task
    if rt is None:
        print("╚" + "═" * w + "╝")
        return

    # Timing
    elapsed = mt.finished_at - mt.started_at if mt.finished_at else 0
    print("╠" + "═" * w + "╣")
    print(f"║  Task ID : {rt.id[:16]}{'…'.ljust(w - 29)}  ║")
    print(f"║  Elapsed : {elapsed:.1f}s{' ' * (w - 14)}  ║")

    # Path
    path_str = " → ".join(rt.path) if rt.path else "—"
    print(f"║  Path    : {_truncate(path_str, w - 14).ljust(w - 14)}  ║")

    # Traceroute
    print("╠" + "═" * w + "╣")
    print(f"║  {'TRACEROUTE'.ljust(w - 2)}  ║")
    print("╠" + "─" * w + "╣")
    for line in rt.traceroute().split("\n"):
        print(f"║  {_truncate(line, w - 4).ljust(w - 4)}  ║")

    # Node outputs — print triage and final answer
    print("╠" + "═" * w + "╣")
    print(f"║  {'NODE OUTPUTS'.ljust(w - 2)}  ║")
    print("╠" + "─" * w + "╣")

    interesting = ["triager", "quick_answer", "executor", "hermes_executor", "reviewer"]
    for node_id in interesting:
        output = rt.context.get(node_id)
        if output is None:
            continue
        out_str = str(output).strip()
        print(f"║  [{node_id}]".ljust(w + 2) + "  ║")
        # Print up to 5 lines of output
        for line in out_str.split("\n")[:5]:
            print(f"║    {_truncate(line, w - 6).ljust(w - 6)}  ║")
        if len(out_str.split("\n")) > 5:
            print(f"║    … ({len(out_str.split(chr(10)))} lines total){' ' * (w - 26)}  ║")

    # Final answer from reviewer if available
    import json as _json
    reviewer_out = rt.context.get("reviewer", "")
    if reviewer_out:
        try:
            rev = _json.loads(str(reviewer_out))
            final = rev.get("final_answer", "")
            approved = rev.get("approved", False)
            score = rev.get("score", "?")
            feedback = rev.get("feedback", "")
            print("╠" + "═" * w + "╣")
            verdict = "APPROVED" if approved else "NEEDS REVISION"
            print(f"║  REVIEW: {verdict} (score: {score}/10){''.ljust(w - 33)}  ║")
            if feedback:
                for line in feedback.split("\n")[:3]:
                    print(f"║    {_truncate(line, w - 6).ljust(w - 6)}  ║")
            if final:
                print("╠" + "─" * w + "╣")
                print(f"║  {'FINAL ANSWER'.ljust(w - 2)}  ║")
                for line in final.split("\n")[:8]:
                    print(f"║    {_truncate(line, w - 6).ljust(w - 6)}  ║")
        except Exception:
            pass

    print("╚" + "═" * w + "╝")


# ── Main execution ────────────────────────────────────────────────────────────

TASKS = [
    ("What is the capital of Japan?",
     "Simple factual geography question — should route to quick_answer."),

    ("Design a rate limiter for an API",
     "Design a production-grade rate limiter: algorithm choices (token bucket, "
     "sliding window, etc.), data structures, Redis patterns, edge cases, "
     "and a Python implementation sketch."),

    ("List files in /tmp",
     "Use terminal tools to list all files and directories inside /tmp "
     "and report what you find."),

    ("Write a Python function to check if a string is a palindrome",
     "Implement is_palindrome(s: str) -> bool. Handle edge cases: "
     "empty string, single character, case insensitivity, spaces/punctuation."),
]


async def process_task(
    conductor: SmartConductor,
    mt: ManagedTask,
    kanban: list[ManagedTask],
) -> None:
    mt.status = "running"
    mt.started_at = time.time()

    print(f"\n  ◎ Starting: {mt.title}")
    print(f"    {mt.description[:80]}{'…' if len(mt.description) > 80 else ''}")

    try:
        # SmartConductor decides the path via LLM routing.
        # We pass the full description as payload for richer context.
        payload = f"Task: {mt.title}\n\nDetails: {mt.description}"

        task = conductor.create_task(
            payload=payload,
            trust_domain="triage",
            max_retries=1,
        )

        mt.routed_path = list(task.path)
        print(f"    Routed: {' → '.join(task.path)}")

        # Execute with IRQ retry support.
        # The reviewer can request executor to retry via context flag.
        result = await conductor.execute_with_retry(task)

        # Post-execution: if reviewer flagged low quality, fire retry
        import json as _json
        reviewer_out = result.context.get("reviewer", "")
        if reviewer_out and "executor" in result.path:
            try:
                rev = _json.loads(str(reviewer_out))
                if not rev.get("approved", True) and rev.get("feedback"):
                    print(f"    ↻ Reviewer requested retry: {rev['feedback'][:60]}…")
                    await conductor.request_retry(
                        task=result,
                        source="reviewer",
                        target="executor",
                        feedback=rev["feedback"],
                    )
                    result = await conductor.execute_with_retry(result)
            except Exception:
                pass

        mt.result_task = result
        mt.status = "done"

    except Exception as e:
        mt.status = "failed"
        print(f"    ERROR: {e}")

    finally:
        mt.finished_at = time.time()

    elapsed = mt.finished_at - mt.started_at
    icon = "✔" if mt.status == "done" else "✖"
    print(f"    {icon} Finished in {elapsed:.1f}s")


async def main() -> None:
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Task Manager — powered by scionic + SmartConductor             ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Model    : {MODEL}")
    print(f"  API key  : {API_KEY[:12]}…")
    print()

    conductor = build_conductor()
    print(conductor.topology_summary())

    # Build managed task list
    managed = [
        ManagedTask(
            id=f"T{i+1:02d}",
            title=title,
            description=desc,
        )
        for i, (title, desc) in enumerate(TASKS)
    ]

    # Initial kanban
    print_kanban(managed)

    # Process tasks sequentially (one at a time for readable output)
    for mt in managed:
        await process_task(conductor, mt, managed)
        print_kanban(managed)

    # Detailed view of each completed task
    print("\n" + "═" * 68)
    print("  DETAILED RESULTS WITH TRACEROUTES")
    print("═" * 68)

    for mt in managed:
        print_task_detail(mt)

    # Summary
    done    = sum(1 for t in managed if t.status == "done")
    failed  = sum(1 for t in managed if t.status == "failed")
    total_t = sum(t.finished_at - t.started_at for t in managed if t.finished_at)

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  Complete: {done}/{len(managed)} tasks  |  Failed: {failed}  |  Wall time: {total_t:.1f}s{''.ljust(18)}║")
    print("╚══════════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    asyncio.run(main())
