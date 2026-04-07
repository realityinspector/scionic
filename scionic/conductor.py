"""
Conductor — the graph-aware orchestrator.

Special node that sits at the top of the topology (like a SCION core AS).
It runs beaconing, path selection, IRQ routing, and monitors execution.
Can itself be backed by an LLM (Hermes) or run as a deterministic process.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from .forwarder import TaskForwarder
from .irq_bus import IRQBus
from .node import Node, NodeHandler
from .path_selector import PathSelector
from .peer import PeerNetwork
from .registry import BeaconRegistry
from .types import (
    IRQ,
    IRQPriority,
    IRQType,
    NodeID,
    PathPolicy,
    PeerMessage,
    Task,
)

logger = logging.getLogger(__name__)


class Conductor:
    """
    Top-level graph orchestrator.

    Wires together all SCION-graph components:
    - BeaconRegistry for capability discovery
    - PathSelector for route assembly
    - TaskForwarder for hop-by-hop execution
    - IRQBus for interrupt propagation
    - PeerNetwork for lateral messaging

    Usage:
        conductor = Conductor()
        conductor.add_node("researcher", handler, capabilities=["search", "rag"])
        conductor.add_node("writer", handler, capabilities=["draft", "edit"])
        conductor.add_peer_link("researcher", "writer")

        task = conductor.create_task("Write a report on X")
        result = await conductor.execute(task)
        print(result.traceroute())
    """

    def __init__(self, conductor_id: str = "conductor") -> None:
        self.conductor_id = conductor_id
        self.registry = BeaconRegistry()
        self.path_selector = PathSelector(self.registry)
        self.forwarder = TaskForwarder()
        self.irq_bus = IRQBus()
        self.peer_network = PeerNetwork()
        self._nodes: dict[NodeID, Node] = {}
        self._irq_log: list[IRQ] = []

        # Conductor subscribes to all IRQs
        self.irq_bus.subscribe_global(self._handle_irq)

    def add_node(
        self,
        node_id: NodeID,
        handler: NodeHandler,
        trust_domain: str = "default",
        secret: str = "",
        cost_per_call: float = 0.0,
        avg_latency_ms: float = 0.0,
        max_concurrency: int = 1,
    ) -> Node:
        """Register a node in the graph."""
        node = Node(
            node_id=node_id,
            handler=handler,
            trust_domain=trust_domain,
            secret=secret,
            cost_per_call=cost_per_call,
            avg_latency_ms=avg_latency_ms,
            max_concurrency=max_concurrency,
        )
        self._nodes[node_id] = node
        self.forwarder.register_node(node)
        self.registry.register(node.beacon())

        # Wire up IRQ and peer handlers
        self.irq_bus.subscribe(node_id, lambda irq, n=node: asyncio.ensure_future(n.receive_irq(irq)))
        self.peer_network.subscribe(node_id, lambda msg, n=node: asyncio.ensure_future(n.receive_peer_message(msg)))

        logger.info(f"Node {node_id} added to graph")
        return node

    def remove_node(self, node_id: NodeID) -> None:
        """Remove a node from the graph."""
        self._nodes.pop(node_id, None)
        self.forwarder.deregister_node(node_id)
        self.registry.deregister(node_id)
        self.irq_bus.unsubscribe(node_id)
        self.peer_network.unsubscribe(node_id)

    def add_peer_link(self, node_a: NodeID, node_b: NodeID) -> None:
        """Establish a peering relationship between two nodes."""
        self.peer_network.add_link(node_a, node_b)

    def refresh_beacons(self) -> None:
        """Re-broadcast all node beacons (update load, etc.)."""
        for node in self._nodes.values():
            self.registry.register(node.beacon())

    def create_task(
        self,
        payload: Any,
        required_capabilities: Optional[list[str]] = None,
        path: Optional[list[NodeID]] = None,
        policy: Optional[PathPolicy] = None,
        trust_domain: str = "default",
    ) -> Task:
        """
        Create a task with automatic or manual path selection.

        If `path` is provided, use it directly.
        If `required_capabilities` is provided, use PathSelector to find a path.
        """
        task = Task(
            payload=payload,
            trust_domain=trust_domain,
            policy=policy.__dict__ if policy else {},
        )

        if path:
            task.path = path
        elif required_capabilities:
            self.refresh_beacons()
            scored_paths = self.path_selector.select(
                required_capabilities=required_capabilities,
                policy=policy,
            )
            if not scored_paths:
                raise ValueError(
                    f"No path found for capabilities: {required_capabilities}"
                )
            task.path = scored_paths[0].path
            logger.info(f"Auto-selected path: {' → '.join(task.path)}")
        else:
            raise ValueError("Provide either `path` or `required_capabilities`")

        return task

    async def execute(self, task: Task) -> Task:
        """Execute a task through its path."""
        logger.info(
            f"Executing task {task.id[:8]}: "
            f"{' → '.join(task.path)}"
        )
        result = await self.forwarder.forward(task)
        logger.info(f"Task {task.id[:8]} complete:\n{result.traceroute()}")
        return result

    async def execute_multipath(
        self,
        payload: Any,
        required_capabilities: list[str],
        policy: Optional[PathPolicy] = None,
        num_paths: int = 2,
    ) -> list[Task]:
        """
        Execute a task across multiple paths simultaneously.

        Returns all results — caller picks the best.
        """
        self.refresh_beacons()
        scored_paths = self.path_selector.select(
            required_capabilities=required_capabilities,
            policy=policy,
            max_paths=num_paths,
        )
        if not scored_paths:
            raise ValueError(
                f"No paths found for capabilities: {required_capabilities}"
            )

        task = Task(payload=payload, policy=policy.__dict__ if policy else {})
        alternate_paths = [sp.path for sp in scored_paths]

        logger.info(
            f"Multi-path execution with {len(alternate_paths)} paths"
        )
        return await self.forwarder.forward_multipath(task, alternate_paths)

    async def fire_irq(
        self,
        source: NodeID,
        irq_type: IRQType,
        payload: Any = None,
        reason: str = "",
        task: Optional[Task] = None,
        target: Optional[NodeID] = None,
        priority: IRQPriority = IRQPriority.NORMAL,
    ) -> int:
        """Fire an interrupt from a node."""
        irq = IRQ(
            source=source,
            target=target,
            task_id=task.id if task else "",
            irq_type=irq_type,
            priority=priority,
            payload=payload,
            reason=reason,
        )
        return await self.irq_bus.fire(irq, task)

    async def send_peer_message(
        self,
        source: NodeID,
        target: NodeID,
        payload: Any,
        task_id: str = "",
        message_type: str = "context",
    ) -> bool:
        """Send a direct peer message between nodes."""
        msg = PeerMessage(
            source=source,
            target=target,
            task_id=task_id,
            payload=payload,
            message_type=message_type,
        )
        return await self.peer_network.send(msg)

    def _handle_irq(self, irq: IRQ) -> None:
        """Conductor's own IRQ handler — logs and can take action."""
        self._irq_log.append(irq)
        logger.info(
            f"Conductor received IRQ: {irq.irq_type.value} from {irq.source} "
            f"(priority={irq.priority.name}): {irq.reason}"
        )
        # Critical IRQs could trigger path re-selection, halt downstream, etc.
        if irq.is_halt:
            logger.warning(f"HALT IRQ received from {irq.source}: {irq.reason}")

    def topology_summary(self) -> str:
        """Human-readable graph summary."""
        lines = [
            f"Conductor: {self.conductor_id}",
            f"Nodes: {len(self._nodes)}",
            "",
            self.registry.summary(),
            "",
            "Peer links:",
        ]
        for link in self.peer_network._links:
            lines.append(f"  {link}")
        if not self.peer_network._links:
            lines.append("  (none)")
        return "\n".join(lines)
