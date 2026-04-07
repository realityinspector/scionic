"""
Hermes Adapter — maps Hermes Agent to scionic nodes.

Unlike the LLM adapter (raw API calls), Hermes nodes have tools:
terminal, file I/O, web search, code execution. A Hermes node can
*do things*, not just *say things*.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any, Optional

from ..types import Hop, Task

logger = logging.getLogger(__name__)


class HermesNodeHandler:
    """
    A scionic NodeHandler backed by a Hermes agent invocation.

    Each call to `process()` spawns a Hermes chat session via
    `hermes chat -q <prompt> -Q --max-turns N`. The -Q flag gives
    clean output for programmatic use.

    Hermes nodes can use tools (terminal, file, web) which makes them
    fundamentally different from raw LLM nodes — they can execute code,
    read files, search the web, etc.
    """

    def __init__(
        self,
        model: str = "anthropic/claude-haiku-4.5",
        hermes_bin: str = "hermes",
        node_capabilities: Optional[list[str]] = None,
        system_prompt: str = "",
        max_turns: int = 5,
        toolsets: Optional[list[str]] = None,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.hermes_bin = hermes_bin
        self._capabilities = node_capabilities or []
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.toolsets = toolsets or ["terminal", "file", "web"]
        self.timeout = timeout

    def capabilities(self) -> list[str]:
        return list(self._capabilities)

    async def process(self, task: Task, hop: Hop) -> Any:
        """
        Process a task by invoking Hermes as a subprocess.

        Uses -q for single query, -Q for quiet/programmatic output,
        --max-turns to limit tool-call loops.
        """
        prompt = self._build_prompt(task)

        logger.info(f"Hermes node ({self.model}): {prompt[:80]}...")

        cmd = [
            self.hermes_bin, "chat",
            "-q", prompt,
            "-Q",  # Quiet mode: no banner/spinner, clean output
            "-m", self.model,
            "--max-turns", str(self.max_turns),
        ]
        if self.toolsets:
            cmd.extend(["-t", ",".join(self.toolsets)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            output = result.stdout.strip()

            # Strip the session_id line that hermes appends
            lines = output.split("\n")
            clean_lines = [
                l for l in lines
                if not l.startswith("session_id:")
            ]
            output = "\n".join(clean_lines).strip()

            if result.returncode != 0 and not output:
                stderr = result.stderr.strip()
                raise RuntimeError(
                    f"Hermes exited {result.returncode}: {stderr[:200]}"
                )

            return output

        except FileNotFoundError:
            raise RuntimeError(
                f"Hermes binary not found at {self.hermes_bin}. "
                f"Install from https://github.com/NousResearch/hermes-agent"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Hermes agent timed out after {self.timeout}s"
            )

    def _build_prompt(self, task: Task) -> str:
        """Build a focused prompt from task payload and hop context."""
        parts = []

        if self.system_prompt:
            parts.append(f"ROLE: {self.system_prompt}")

        parts.append(f"TASK: {task.payload}")

        # Include context from previous hops
        if task.context:
            context_parts = []
            for node_id, output in task.context.items():
                if node_id.startswith("_"):
                    continue  # Skip internal keys
                output_str = str(output)
                if len(output_str) > 2000:
                    output_str = output_str[:2000] + "... (truncated)"
                context_parts.append(f"[{node_id}]: {output_str}")
            if context_parts:
                parts.append("\nCONTEXT FROM PREVIOUS STEPS:")
                parts.extend(context_parts)

        # Include feedback if retrying
        feedback_key = f"_feedback_for_{task.current_node}"
        if feedback_key in task.context:
            parts.append(f"\nFEEDBACK (incorporate this): {task.context[feedback_key]}")

        parts.append(
            "\nProvide a clear, concise result. "
            "Your output will be passed to the next step in the pipeline."
        )

        return "\n".join(parts)


class HermesAdapter:
    """
    High-level adapter for wiring Hermes agents into a Conductor.

    Usage:
        conductor = Conductor()
        adapter = HermesAdapter(conductor)
        adapter.add_agent("researcher", capabilities=["search", "rag"],
                          system_prompt="You are a research specialist.")
        adapter.add_agent("coder", capabilities=["code", "terminal"],
                          toolsets=["terminal", "file"])
    """

    def __init__(self, conductor, hermes_bin: str = "hermes") -> None:
        self.conductor = conductor
        self.hermes_bin = hermes_bin

    def add_agent(
        self,
        node_id: str,
        capabilities: list[str],
        model: str = "anthropic/claude-haiku-4.5",
        system_prompt: str = "",
        trust_domain: str = "default",
        cost_per_call: float = 0.0,
        avg_latency_ms: float = 5000.0,
        max_turns: int = 5,
        toolsets: Optional[list[str]] = None,
    ) -> None:
        """Add a Hermes-backed agent as a graph node."""
        handler = HermesNodeHandler(
            model=model,
            hermes_bin=self.hermes_bin,
            node_capabilities=capabilities,
            system_prompt=system_prompt,
            max_turns=max_turns,
            toolsets=toolsets,
        )
        self.conductor.add_node(
            node_id=node_id,
            handler=handler,
            trust_domain=trust_domain,
            cost_per_call=cost_per_call,
            avg_latency_ms=avg_latency_ms,
        )
