"""
Hermes Adapter — maps Hermes Agent to scionic nodes.

Wraps Hermes's delegate_tool (subagent spawning) as node instantiation,
maps Hermes's tool registry to beacon capabilities, and uses Hermes's
gateway as the inter-node transport layer.

This adapter allows a Conductor to orchestrate Hermes agents as
graph nodes with full SCION-style path selection, hop signing,
and IRQ propagation.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from typing import Any, Optional

from ..types import Hop, Task

logger = logging.getLogger(__name__)


class HermesNodeHandler:
    """
    A scionic NodeHandler backed by a Hermes agent invocation.

    Each call to `process()` spawns a Hermes chat session with the
    task payload as the prompt, using the configured model.

    This maps directly to Hermes's delegate_tool pattern, but instead of
    a tree-shaped parent→child delegation, it's a graph node that
    receives tasks from any direction via the forwarder.
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-6",
        hermes_bin: str = "hermes",
        node_capabilities: Optional[list[str]] = None,
        system_prompt: str = "",
        max_iterations: int = 20,
        toolsets: Optional[list[str]] = None,
    ) -> None:
        self.model = model
        self.hermes_bin = hermes_bin
        self._capabilities = node_capabilities or []
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.toolsets = toolsets or ["terminal", "file", "web"]

    def capabilities(self) -> list[str]:
        return list(self._capabilities)

    async def process(self, task: Task, hop: Hop) -> Any:
        """
        Process a task by invoking Hermes as a subprocess.

        Constructs a focused prompt from the task payload and
        accumulated context from previous hops.
        """
        prompt = self._build_prompt(task)

        logger.info(
            f"Hermes node processing with model={self.model}: "
            f"{prompt[:100]}..."
        )

        try:
            result = subprocess.run(
                [
                    self.hermes_bin,
                    "chat",
                    "--model", self.model,
                    "--non-interactive",
                    "--message", prompt,
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = result.stdout.strip()
            if result.returncode != 0:
                logger.error(f"Hermes error: {result.stderr}")
                raise RuntimeError(f"Hermes exited with code {result.returncode}: {result.stderr[:200]}")
            return output

        except FileNotFoundError:
            raise RuntimeError(
                f"Hermes binary not found at {self.hermes_bin}. "
                f"Install from https://github.com/NousResearch/hermes-agent"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Hermes agent timed out after 300s")

    def _build_prompt(self, task: Task) -> str:
        """Build a focused prompt from task payload and hop context."""
        parts = []

        if self.system_prompt:
            parts.append(f"ROLE: {self.system_prompt}")

        parts.append(f"TASK: {task.payload}")

        # Include context from previous hops (traceroute-style)
        if task.context:
            parts.append("\nCONTEXT FROM PREVIOUS STEPS:")
            for node_id, output in task.context.items():
                # Truncate long outputs
                output_str = str(output)
                if len(output_str) > 2000:
                    output_str = output_str[:2000] + "... (truncated)"
                parts.append(f"\n[{node_id}]:\n{output_str}")

        parts.append(
            "\nProvide a clear, concise result. "
            "Your output will be passed to the next step in the pipeline."
        )

        return "\n".join(parts)


class HermesAdapter:
    """
    High-level adapter for wiring Hermes agents into a Conductor.

    Usage:
        from scionic import Conductor
        from scionic.adapters import HermesAdapter

        conductor = Conductor()
        adapter = HermesAdapter(conductor)

        adapter.add_agent(
            "researcher",
            capabilities=["search", "rag"],
            model="anthropic/claude-sonnet-4-6",
            system_prompt="You are a research specialist.",
        )
        adapter.add_agent(
            "writer",
            capabilities=["draft", "edit"],
            model="anthropic/claude-opus-4-6",
            system_prompt="You are a technical writer.",
        )

        task = conductor.create_task(
            "Write a report on SCION",
            required_capabilities=["search", "draft"],
        )
        result = await conductor.execute(task)
    """

    def __init__(self, conductor, hermes_bin: str = "hermes") -> None:
        self.conductor = conductor
        self.hermes_bin = hermes_bin

    def add_agent(
        self,
        node_id: str,
        capabilities: list[str],
        model: str = "anthropic/claude-sonnet-4-6",
        system_prompt: str = "",
        trust_domain: str = "default",
        cost_per_call: float = 0.0,
        avg_latency_ms: float = 5000.0,
    ) -> None:
        """Add a Hermes-backed agent as a graph node."""
        handler = HermesNodeHandler(
            model=model,
            hermes_bin=self.hermes_bin,
            node_capabilities=capabilities,
            system_prompt=system_prompt,
        )
        self.conductor.add_node(
            node_id=node_id,
            handler=handler,
            trust_domain=trust_domain,
            cost_per_call=cost_per_call,
            avg_latency_ms=avg_latency_ms,
        )
