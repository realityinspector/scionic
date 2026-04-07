"""
IRQBus — async interrupt propagation across the graph.

Any node can fire an IRQ. The bus delivers it to:
- A specific target node
- All nodes on a task's path (broadcast)
- The conductor (always receives all IRQs)

The conductor can mask interrupts by priority level per-task.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from typing import Callable, Optional

from .types import IRQ, IRQPriority, IRQType, NodeID, Task

logger = logging.getLogger(__name__)

# Type for IRQ handler callbacks
IRQHandler = Callable[[IRQ], None]


class IRQBus:
    """
    Pub/sub interrupt bus for the graph.

    Nodes subscribe to receive IRQs. When an IRQ fires, the bus
    delivers it based on target, priority, and mask settings.
    """

    def __init__(self) -> None:
        # Per-node handlers
        self._handlers: dict[NodeID, list[IRQHandler]] = defaultdict(list)
        # Global handlers (conductor)
        self._global_handlers: list[IRQHandler] = []
        # Per-task IRQ masks: task_id -> set of masked priorities
        self._masks: dict[str, set[IRQPriority]] = {}
        # IRQ history for debugging
        self._history: list[IRQ] = []
        self._lock = asyncio.Lock()

    def subscribe(self, node_id: NodeID, handler: IRQHandler) -> None:
        """Subscribe a node to receive IRQs."""
        self._handlers[node_id].append(handler)

    def subscribe_global(self, handler: IRQHandler) -> None:
        """Subscribe to ALL IRQs (used by conductor)."""
        self._global_handlers.append(handler)

    def unsubscribe(self, node_id: NodeID) -> None:
        """Remove all handlers for a node."""
        self._handlers.pop(node_id, None)

    def mask(self, task_id: str, priority: IRQPriority) -> None:
        """Mask (suppress) IRQs at a priority level for a task."""
        if task_id not in self._masks:
            self._masks[task_id] = set()
        self._masks[task_id].add(priority)

    def unmask(self, task_id: str, priority: IRQPriority) -> None:
        """Unmask IRQs at a priority level for a task."""
        if task_id in self._masks:
            self._masks[task_id].discard(priority)

    async def fire(
        self,
        irq: IRQ,
        task: Optional[Task] = None,
    ) -> int:
        """
        Fire an IRQ and deliver to appropriate handlers.

        Returns the number of handlers that received it.
        """
        async with self._lock:
            self._history.append(irq)

        delivered = 0

        # Check task-level mask
        if task and irq.task_id:
            masked = self._masks.get(irq.task_id, set())
            if irq.priority in masked and irq.priority != IRQPriority.CRITICAL:
                logger.debug(f"IRQ {irq.id[:8]} masked for task {irq.task_id[:8]}")
                return 0

        # Deliver to global handlers (conductor) — always
        for handler in self._global_handlers:
            await self._deliver(handler, irq)
            delivered += 1

        # Deliver to specific target
        if irq.target:
            for handler in self._handlers.get(irq.target, []):
                await self._deliver(handler, irq)
                delivered += 1
        elif task:
            # Broadcast to all nodes on the task's path
            for node_id in task.path:
                if node_id == irq.source:
                    continue  # Don't deliver to sender
                for handler in self._handlers.get(node_id, []):
                    await self._deliver(handler, irq)
                    delivered += 1

        logger.info(
            f"IRQ {irq.irq_type.value} from {irq.source} "
            f"delivered to {delivered} handlers"
        )
        return delivered

    async def _deliver(self, handler: IRQHandler, irq: IRQ) -> None:
        """Deliver an IRQ to a handler, handling both sync and async."""
        try:
            if inspect.iscoroutinefunction(handler):
                await handler(irq)
            else:
                handler(irq)
        except Exception as e:
            logger.error(f"IRQ handler error: {e}")

    def history(self, task_id: Optional[str] = None) -> list[IRQ]:
        """Get IRQ history, optionally filtered by task."""
        if task_id:
            return [i for i in self._history if i.task_id == task_id]
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()
