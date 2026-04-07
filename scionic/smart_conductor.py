"""
SmartConductor — LLM-driven routing.

Instead of deterministic path selection (cheapest, fastest), the
SmartConductor uses an LLM to read available beacons and reason
about which path the task needs.

"This looks like a security-sensitive task, route through the
security scanner AND the code reviewer, not just the linter."
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .conductor import Conductor
from .registry import BeaconRegistry
from .types import NodeID, PathPolicy, Task

logger = logging.getLogger(__name__)


class SmartConductor(Conductor):
    """
    A Conductor that uses an LLM to decide routing.

    Overrides create_task to call an LLM with the beacon registry
    summary and the task payload, letting the model assemble the path.

    Falls back to deterministic PathSelector if the LLM fails.
    """

    def __init__(
        self,
        model: str = "anthropic/claude-haiku-4.5",
        api_key: str = "",
        base_url: str = "https://openrouter.ai/api/v1",
        conductor_id: str = "smart-conductor",
        **kwargs,
    ) -> None:
        super().__init__(conductor_id=conductor_id, **kwargs)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def create_task(
        self,
        payload: Any,
        required_capabilities: Optional[list[str]] = None,
        path: Optional[list[NodeID]] = None,
        policy: Optional[PathPolicy] = None,
        trust_domain: str = "default",
        max_retries: Optional[int] = None,
    ) -> Task:
        """
        Create a task. If no explicit path given, ask the LLM to route it.

        Falls back to deterministic selection if LLM routing fails.
        """
        if path:
            return super().create_task(
                payload=payload, path=path, policy=policy,
                trust_domain=trust_domain, max_retries=max_retries,
            )

        # Try LLM routing
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context — use a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    llm_path = pool.submit(
                        asyncio.run, self._llm_route(payload)
                    ).result(timeout=30)
            else:
                llm_path = loop.run_until_complete(self._llm_route(payload))

            if llm_path:
                logger.info(f"SmartConductor LLM-routed: {' → '.join(llm_path)}")
                return super().create_task(
                    payload=payload, path=llm_path, policy=policy,
                    trust_domain=trust_domain, max_retries=max_retries,
                )
        except Exception as e:
            logger.warning(f"LLM routing failed, falling back: {e}")

        # Fallback to deterministic
        if required_capabilities:
            return super().create_task(
                payload=payload, required_capabilities=required_capabilities,
                policy=policy, trust_domain=trust_domain,
                max_retries=max_retries,
            )

        raise ValueError("SmartConductor: LLM routing failed and no required_capabilities provided")

    async def _llm_route(self, payload: Any) -> Optional[list[NodeID]]:
        """Ask the LLM to pick a path based on available nodes."""
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx required for SmartConductor")

        self.refresh_beacons()
        beacons = self.registry.all_active()
        if not beacons:
            return None

        # Build the beacon summary for the LLM
        node_descriptions = []
        for b in sorted(beacons, key=lambda x: x.node_id):
            caps = ", ".join(b.capabilities)
            desc = (
                f"- {b.node_id}: capabilities=[{caps}], "
                f"cost=${b.cost_per_call:.3f}, "
                f"latency={b.avg_latency_ms:.0f}ms, "
                f"trust_domain={b.trust_domain}"
            )
            node_descriptions.append(desc)

        nodes_text = "\n".join(node_descriptions)
        node_ids = [b.node_id for b in beacons]

        system_prompt = (
            "You are a routing conductor. Given a task and available nodes, "
            "select the best path (ordered list of node IDs) for the task.\n\n"
            "Rules:\n"
            "- Pick nodes whose capabilities match what the task needs\n"
            "- Order them logically (research before writing, analysis before review)\n"
            "- Prefer cheaper nodes when quality is equivalent\n"
            "- Only use nodes from the available list\n\n"
            f"Available nodes:\n{nodes_text}\n\n"
            "Respond with ONLY a JSON array of node IDs, e.g. [\"node_a\", \"node_b\"]. "
            "No markdown, no explanation."
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Route this task: {payload}"},
                    ],
                    "max_tokens": 128,
                    "temperature": 0.0,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"].strip()

        # Parse — handle markdown fences
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines[1:] if l.strip() != "```"]
            content = "\n".join(lines).strip()

        path = json.loads(content)

        # Validate all nodes exist
        valid_path = [n for n in path if n in node_ids]
        if not valid_path:
            return None

        return valid_path

    async def route_and_execute(self, payload: Any, **kwargs) -> Task:
        """Convenience: create + execute in one call."""
        task = self.create_task(payload=payload, **kwargs)
        return await self.execute(task)
