"""
Generic LLM Adapter — maps any OpenAI-compatible API to a scionic node.

Uses the OpenAI client format (works with OpenRouter, Ollama, vLLM, etc.)
so any model can be a graph node without requiring Hermes.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from ..types import Hop, Task

logger = logging.getLogger(__name__)


class LLMNodeHandler:
    """
    A scionic NodeHandler backed by a direct LLM API call.

    Uses the OpenAI-compatible chat completions format, which works with:
    - OpenRouter (any model)
    - Ollama (local models)
    - vLLM, TGI, etc.
    - OpenAI itself

    This is the lightweight alternative to HermesNodeHandler when you
    don't need Hermes's tool-calling, memory, or session management.
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-6",
        api_key: str = "",
        base_url: str = "https://openrouter.ai/api/v1",
        node_capabilities: Optional[list[str]] = None,
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._capabilities = node_capabilities or []
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    def capabilities(self) -> list[str]:
        return list(self._capabilities)

    async def process(self, task: Task, hop: Hop) -> Any:
        """Process a task by calling the LLM API."""
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx required for LLMNodeHandler: pip install httpx")

        messages = self._build_messages(task)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        hop.tokens_used = usage.get("total_tokens", 0)

        return content

    def _build_messages(self, task: Task) -> list[dict[str, str]]:
        """Build chat messages from task payload and context."""
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Build user message with context from previous hops
        parts = [str(task.payload)]

        if task.context:
            parts.append("\n\nContext from previous steps:")
            for node_id, output in task.context.items():
                output_str = str(output)
                if len(output_str) > 2000:
                    output_str = output_str[:2000] + "..."
                parts.append(f"\n[{node_id}]: {output_str}")

        messages.append({"role": "user", "content": "\n".join(parts)})
        return messages
