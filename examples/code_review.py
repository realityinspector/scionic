#!/usr/bin/env python3
"""
Code Review Pipeline — a real app built on scionic.

Routes code through a multi-agent review pipeline:
1. Linter (Hermes node with terminal access — runs actual linting)
2. Security scanner (LLM — checks for OWASP issues)
3. Code reviewer (LLM — reviews logic, style, correctness)
4. Approval gate (LLM — synthesizes findings, approve/reject)

Demonstrates:
- Hermes nodes (terminal tools) mixed with LLM nodes
- Peer messaging: linter shares findings directly with reviewer
- IRQ: if security scanner finds critical issue, halts pipeline
- Trust domains: linter/scanner in "tooling" domain, reviewer in "analysis"
- Full traceroute of the review process
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# API key
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

from scionic import Conductor, IRQPriority, IRQType
from scionic.adapters.llm import LLMNodeHandler
from scionic.adapters.hermes import HermesNodeHandler

MODEL = "anthropic/claude-haiku-4.5"
BASE = "https://openrouter.ai/api/v1"


def build_pipeline() -> Conductor:
    """Wire up the code review pipeline."""
    c = Conductor(verify_signatures=True, auto_reroute=True)

    # Trust domains
    c.add_trust_domain("tooling", "Tooling", allowed_peers=["analysis"])
    c.add_trust_domain("analysis", "Analysis", allowed_peers=["tooling"])

    # Node 1: Linter — uses Hermes with terminal access to run real tools
    linter = HermesNodeHandler(
        model=MODEL,
        node_capabilities=["lint"],
        system_prompt=(
            "You are a code linter. Analyze the code provided. "
            "Check for: syntax errors, unused variables, missing type hints, "
            "inconsistent naming. List each issue on its own line. "
            "If the code is clean, say 'No lint issues found.'"
        ),
        max_turns=1,  # Don't need tools for this — just analyze
        toolsets=[],
    )

    # Node 2: Security scanner — LLM checks for vulnerabilities
    security = LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        node_capabilities=["security"],
        system_prompt=(
            "You are a security scanner. Check the code for:\n"
            "- SQL injection\n- XSS\n- Command injection\n"
            "- Hardcoded secrets\n- Insecure deserialization\n"
            "- Path traversal\n\n"
            "Respond with JSON: "
            '{"severity": "none|low|medium|high|critical", '
            '"issues": ["issue 1", ...], '
            '"recommendation": "one sentence"}. '
            "ONLY the JSON, no markdown."
        ),
        max_tokens=256,
        temperature=0.0,
    )

    # Node 3: Code reviewer — reviews logic and style
    reviewer = LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        node_capabilities=["review"],
        system_prompt=(
            "You are a senior code reviewer. Review the code for:\n"
            "- Correctness and edge cases\n"
            "- Code clarity and readability\n"
            "- Performance concerns\n"
            "- Test coverage gaps\n\n"
            "You will see lint findings and security scan results from "
            "previous steps in the context. Incorporate them into your review.\n"
            "Also check for any peer messages with additional hints.\n\n"
            "Provide a concise review with specific, actionable feedback."
        ),
        max_tokens=512,
        temperature=0.3,
    )

    # Node 4: Approval gate — synthesizes everything
    approver = LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        node_capabilities=["approve"],
        system_prompt=(
            "You are the final approval gate for code review. "
            "You see all previous findings: lint, security, and review.\n\n"
            "Respond with JSON: "
            '{"approved": true|false, '
            '"summary": "one sentence overall assessment", '
            '"blocking_issues": ["issue 1", ...] or []}. '
            "ONLY the JSON, no markdown."
        ),
        max_tokens=256,
        temperature=0.0,
    )

    c.add_node("linter", linter, trust_domain="tooling",
               cost_per_call=0.002, avg_latency_ms=3000)
    c.add_node("security", security, trust_domain="tooling",
               cost_per_call=0.001, avg_latency_ms=1500)
    c.add_node("reviewer", reviewer, trust_domain="analysis",
               cost_per_call=0.003, avg_latency_ms=2000)
    c.add_node("approver", approver, trust_domain="analysis",
               cost_per_call=0.001, avg_latency_ms=1000)

    # Peer link: linter shares findings directly with reviewer
    c.add_peer_link("linter", "reviewer")
    # Peer link: security shares findings directly with approver
    c.add_peer_link("security", "approver")

    return c


SAMPLE_CODE_GOOD = '''
def fibonacci(n: int) -> list[int]:
    """Return the first n Fibonacci numbers."""
    if n <= 0:
        return []
    if n == 1:
        return [0]

    fibs = [0, 1]
    for _ in range(2, n):
        fibs.append(fibs[-1] + fibs[-2])
    return fibs
'''

SAMPLE_CODE_BAD = '''
import os
import subprocess

DB_PASSWORD = "admin123"  # TODO: move to env

def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = subprocess.run(f"psql -c '{query}'", shell=True, capture_output=True)
    return result.stdout

def process_file(filename):
    path = "/data/" + filename
    with open(path) as f:
        return eval(f.read())
'''


async def review_code(code: str, description: str):
    """Run a piece of code through the review pipeline."""
    c = build_pipeline()

    print(f"\n{'═' * 60}")
    print(f"  Code Review: {description}")
    print(f"{'═' * 60}")

    task = c.create_task(
        payload=f"Review this code:\n```python\n{code}\n```",
        path=["linter", "security", "reviewer", "approver"],
    )

    result = await c.execute(task)

    # Print results
    print(f"\n  Traceroute:")
    print(f"  {result.traceroute()}")

    for node_id in ["linter", "security", "reviewer", "approver"]:
        output = str(result.context.get(node_id, ""))
        print(f"\n  [{node_id}]:")
        for line in output.split("\n"):
            print(f"    {line}")

    return result


async def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Code Review Pipeline — powered by scionic              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Review clean code
    await review_code(SAMPLE_CODE_GOOD, "Clean Fibonacci function")

    # Review vulnerable code
    await review_code(SAMPLE_CODE_BAD, "Code with security issues")

    print(f"\n{'═' * 60}")
    print("  Reviews complete.")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
