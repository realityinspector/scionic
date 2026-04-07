"""
Node — a processing unit in the graph.

Each node has a handler function that processes tasks.
Nodes broadcast beacons, sign hops, and forward tasks.
Adapters (Hermes, HTTP, etc.) implement NodeHandler.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import time
from typing import Any, Protocol

from .types import Beacon, Hop, HopStatus, IRQ, NodeID, PeerMessage, Task

logger = logging.getLogger(__name__)


class NodeHandler(Protocol):
    """
    Interface that adapters implement.

    A NodeHandler does the actual work — calling an LLM, running a search,
    processing data. The Node wraps it with hop signing and forwarding.
    """

    async def process(self, task: Task, hop: Hop) -> Any:
        """Process a task and return the output."""
        ...

    def capabilities(self) -> list[str]:
        """Return this node's capability list for beaconing."""
        ...


class Node:
    """
    A graph node that processes tasks, signs hops, and manages beacons.

    Wraps a NodeHandler with the scionic protocol:
    - Validates it's the correct hop
    - Injects peer context into the task before processing
    - Hashes input, calls handler, hashes output
    - Signs the hop
    - Forwards to the next node (via the forwarder)
    """

    def __init__(
        self,
        node_id: NodeID,
        handler: NodeHandler,
        trust_domain: str = "default",
        secret: str = "",
        cost_per_call: float = 0.0,
        avg_latency_ms: float = 0.0,
        max_concurrency: int = 1,
    ) -> None:
        self.node_id = node_id
        self.handler = handler
        self.trust_domain = trust_domain
        self.secret = secret or node_id
        self.cost_per_call = cost_per_call
        self.avg_latency_ms = avg_latency_ms
        self.max_concurrency = max_concurrency
        self.current_load = 0
        self._irq_handlers: list = []
        self._peer_handlers: list = []
        # Peer context buffer: messages received from peers, injected on next process()
        self._peer_context: list[PeerMessage] = []

    def beacon(self) -> Beacon:
        """Generate this node's capability beacon."""
        return Beacon(
            node_id=self.node_id,
            capabilities=self.handler.capabilities(),
            trust_domain=self.trust_domain,
            cost_per_call=self.cost_per_call,
            avg_latency_ms=self.avg_latency_ms,
            max_concurrency=self.max_concurrency,
            current_load=self.current_load,
        )

    async def execute(self, task: Task) -> Task:
        """
        Process a task at this hop.

        1. Validate we're the current hop
        2. Inject any buffered peer context into the task
        3. Create the hop record, hash input
        4. Call the handler
        5. Hash output, sign the hop
        6. Append to task's traceroute, advance
        """
        if task.current_node != self.node_id:
            raise ValueError(
                f"Node {self.node_id} received task meant for {task.current_node}"
            )

        self.current_load += 1

        # Inject buffered peer messages into task context
        peer_count = 0
        if self._peer_context:
            peer_data = []
            for msg in self._peer_context:
                peer_data.append({
                    "from": msg.source,
                    "type": msg.message_type,
                    "payload": msg.payload,
                })
            task.context[f"_peer_messages_for_{self.node_id}"] = peer_data
            peer_count = len(self._peer_context)
            self._peer_context.clear()

        hop = Hop(node_id=self.node_id, status=HopStatus.IN_PROGRESS)
        hop.peer_messages_received = peer_count
        hop.started_at = time.time()

        # Check if this is a retry
        retried_nodes = [h.node_id for h in task.hops if h.status == HopStatus.FAILED]
        if self.node_id in retried_nodes:
            hop.retry_of = self.node_id

        # Hash input
        input_data = json.dumps(
            {"payload": str(task.payload), "context": task.context},
            sort_keys=True, default=str
        )
        hop.input_hash = hashlib.sha256(input_data.encode()).hexdigest()[:16]

        try:
            output = await self.handler.process(task, hop)
            hop.output = output
            hop.status = HopStatus.COMPLETE
            task.context[self.node_id] = output

        except Exception as e:
            hop.status = HopStatus.FAILED
            hop.error = str(e)
            hop.output = None

        hop.completed_at = time.time()

        # Hash output and sign
        output_data = json.dumps(str(hop.output), default=str)
        hop.output_hash = hashlib.sha256(output_data.encode()).hexdigest()[:16]
        hop.sign(self.secret)

        task.hops.append(hop)
        task.advance()
        self.current_load -= 1

        return task

    def on_irq(self, handler) -> None:
        """Register an IRQ handler for this node."""
        self._irq_handlers.append(handler)

    async def receive_irq(self, irq: IRQ) -> None:
        """Handle an incoming interrupt."""
        for handler in self._irq_handlers:
            if inspect.iscoroutinefunction(handler):
                await handler(irq)
            else:
                handler(irq)

    def on_peer_message(self, handler) -> None:
        """Register a peer message handler."""
        self._peer_handlers.append(handler)

    async def receive_peer_message(self, msg: PeerMessage) -> None:
        """
        Handle a direct peer message.

        Default behavior: buffer the message so it's injected into the
        task context on the next process() call. Custom handlers can
        override this by registering via on_peer_message().
        """
        # Always buffer for injection
        self._peer_context.append(msg)
        logger.debug(f"Node {self.node_id} buffered peer message from {msg.source}")

        # Also call custom handlers
        for handler in self._peer_handlers:
            if inspect.iscoroutinefunction(handler):
                await handler(msg)
            else:
                handler(msg)
