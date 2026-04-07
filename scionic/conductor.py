"""
Conductor — the graph-aware orchestrator.

Special node that sits at the top of the topology (like a SCION core AS).
It runs beaconing, path selection, IRQ routing, and monitors execution.
Can itself be backed by an LLM (Hermes) or run as a deterministic process.

Key behaviors:
- IRQ retry: when a RETRY_REQUESTED IRQ fires, re-execute the failed hop
  with feedback injected into the task context
- Reroute on failure: when a hop fails, find an alternate node with the
  same capability and reroute
- Trust domain enforcement: register domains and let the forwarder enforce
"""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import Any, Optional

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
    Hop,
    HopStatus,
    NodeID,
    PathPolicy,
    PeerMessage,
    Task,
    TrustDomain,
)

logger = logging.getLogger(__name__)


class Conductor:
    """
    Top-level graph orchestrator.

    Wires together all scionic components:
    - BeaconRegistry for capability discovery
    - PathSelector for route assembly
    - TaskForwarder for hop-by-hop execution with signature verification
    - IRQBus for interrupt propagation
    - PeerNetwork for lateral messaging
    - Trust domains for isolation boundaries
    """

    def __init__(
        self,
        conductor_id: str = "conductor",
        verify_signatures: bool = True,
        auto_reroute: bool = True,
        max_retries: int = 1,
    ) -> None:
        self.conductor_id = conductor_id
        self.registry = BeaconRegistry()
        self.path_selector = PathSelector(self.registry)
        self.forwarder = TaskForwarder(verify_signatures=verify_signatures)
        self.irq_bus = IRQBus()
        self.peer_network = PeerNetwork()
        self._nodes: dict[NodeID, Node] = {}
        self._trust_domains: dict[str, TrustDomain] = {}
        self._irq_log: list[IRQ] = []
        self.auto_reroute = auto_reroute
        self.max_retries = max_retries

        # Conductor subscribes to all IRQs
        self.irq_bus.subscribe_global(self._handle_irq)

    # ── Node management ──────────────────────────────────────────────

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

        # Wire up IRQ and peer handlers (async — awaited by bus/network)
        self.irq_bus.subscribe(
            node_id,
            lambda irq, n=node: n.receive_irq(irq),
        )
        self.peer_network.subscribe(
            node_id,
            lambda msg, n=node: n.receive_peer_message(msg),
        )

        logger.info(f"Node {node_id} added to graph (trust_domain={trust_domain})")
        return node

    def remove_node(self, node_id: NodeID) -> None:
        self._nodes.pop(node_id, None)
        self.forwarder.deregister_node(node_id)
        self.registry.deregister(node_id)
        self.irq_bus.unsubscribe(node_id)
        self.peer_network.unsubscribe(node_id)

    # ── Trust domain management ──────────────────────────────────────

    def add_trust_domain(
        self,
        domain_id: str,
        name: str,
        description: str = "",
        allowed_peers: Optional[list[str]] = None,
    ) -> TrustDomain:
        """Register a trust domain (ISD)."""
        domain = TrustDomain(
            id=domain_id,
            name=name,
            description=description,
            allowed_peers=allowed_peers or [],
        )
        self._trust_domains[domain_id] = domain
        self.forwarder.register_trust_domain(domain)
        logger.info(f"Trust domain '{name}' ({domain_id}) registered")
        return domain

    # ── Peer links ───────────────────────────────────────────────────

    def add_peer_link(self, node_a: NodeID, node_b: NodeID) -> None:
        self.peer_network.add_link(node_a, node_b)

    # ── Beaconing ────────────────────────────────────────────────────

    def refresh_beacons(self) -> None:
        for node in self._nodes.values():
            self.registry.register(node.beacon())

    # ── Task creation ────────────────────────────────────────────────

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
        Create a task with automatic or manual path selection.

        If `path` is provided, use it directly.
        If `required_capabilities` is provided, use PathSelector to find a path.
        """
        task = Task(
            payload=payload,
            trust_domain=trust_domain,
            policy=policy.__dict__ if policy else {},
            max_retries=max_retries if max_retries is not None else self.max_retries,
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

    # ── Execution ────────────────────────────────────────────────────

    async def execute(self, task: Task) -> Task:
        """
        Execute a task through its path.

        If a hop fails and auto_reroute is enabled, attempts to find
        an alternate node with the same capability and retry.
        """
        logger.info(
            f"Executing task {task.id[:8]}: "
            f"{' → '.join(task.path)}"
        )

        result = await self.forwarder.forward(task)

        # Check for failures and attempt reroute
        if self.auto_reroute and result.failed_hops:
            result = await self._attempt_reroute(result)

        logger.info(f"Task {result.id[:8]} complete:\n{result.traceroute()}")
        return result

    async def _attempt_reroute(self, task: Task) -> Task:
        """
        When a hop fails, find an alternate node with the same capability
        and retry from that point.
        """
        if not task.failed_hops:
            return task

        last_failure = task.failed_hops[-1]
        failed_node_id = last_failure.node_id
        failed_node = self._nodes.get(failed_node_id)

        if failed_node is None:
            return task

        # Find the capability of the failed node
        capabilities = failed_node.handler.capabilities()
        if not capabilities:
            return task

        # Look for alternate nodes with the same capability
        self.refresh_beacons()
        for cap in capabilities:
            alternates = self.registry.find_by_capability(cap)
            for beacon in alternates:
                if beacon.node_id == failed_node_id:
                    continue
                if beacon.available_capacity <= 0:
                    continue

                # Found an alternate — reroute
                alt_node = self._nodes.get(beacon.node_id)
                if alt_node is None:
                    continue

                logger.info(
                    f"Rerouting from failed {failed_node_id} to {beacon.node_id}"
                )

                # Mark the failed hop
                last_failure.status = HopStatus.RETRIED

                # Replace the failed node in the path and rewind
                path_idx = task.path.index(failed_node_id)
                task.path[path_idx] = beacon.node_id
                task.rewind(1)

                # Continue execution from the rerouted node
                return await self.forwarder.forward(task)

        return task

    async def execute_with_retry(
        self,
        task: Task,
        feedback_fn: Optional[Any] = None,
    ) -> Task:
        """
        Execute a task with IRQ-based retry support.

        After execution, if the task has a _retry_requested flag in context
        (set by an IRQ handler), rewind and re-execute the failed hop with
        feedback injected.
        """
        result = await self.execute(task)

        retries = 0
        while (
            result.context.get("_retry_requested")
            and retries < result.max_retries
        ):
            retry_info = result.context.pop("_retry_requested")
            target_node = retry_info.get("target", "")
            feedback = retry_info.get("feedback", "")

            if target_node and feedback:
                logger.info(
                    f"Retrying {target_node} with feedback: {feedback[:80]}..."
                )
                result.inject_feedback(target_node, feedback)

                # Find the target node's position and rewind to it
                if target_node in result.path:
                    target_idx = result.path.index(target_node)
                    result.current_hop_index = target_idx

                    # Mark previous attempt
                    for hop in result.hops:
                        if hop.node_id == target_node and hop.status == HopStatus.COMPLETE:
                            hop.status = HopStatus.RETRIED

                    result = await self.forwarder.forward(result)

            retries += 1

        return result

    # ── Batch execution ────────────────────────────────────────────

    async def execute_batch(
        self,
        tasks: list[Task],
        concurrency: int = 3,
        timeout_per_task: Optional[float] = None,
    ) -> list[Task]:
        """
        Execute multiple tasks concurrently with a semaphore.

        Args:
            tasks: List of tasks to execute
            concurrency: Max tasks running simultaneously
            timeout_per_task: Per-task timeout in seconds (None = no timeout)
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def run_one(task: Task) -> Task:
            async with semaphore:
                if timeout_per_task:
                    try:
                        return await asyncio.wait_for(
                            self.execute(task), timeout=timeout_per_task
                        )
                    except asyncio.TimeoutError:
                        from .types import Hop
                        timeout_hop = Hop(
                            node_id=task.current_node or "unknown",
                            status=HopStatus.FAILED,
                            error=f"Pipeline timeout after {timeout_per_task}s",
                        )
                        task.hops.append(timeout_hop)
                        return task
                return await self.execute(task)

        results = await asyncio.gather(
            *(run_one(t) for t in tasks),
            return_exceptions=True,
        )

        completed = []
        for r in results:
            if isinstance(r, Task):
                completed.append(r)
            elif isinstance(r, Exception):
                logger.error(f"Batch task exception: {r}")
                # Create a failed task placeholder
                failed = Task(payload="(exception)")
                failed.hops.append(Hop(
                    node_id="batch",
                    status=HopStatus.FAILED,
                    error=str(r),
                ))
                completed.append(failed)
        return completed

    # ── Multi-path ───────────────────────────────────────────────────

    async def execute_multipath(
        self,
        payload: Any,
        required_capabilities: list[str],
        policy: Optional[PathPolicy] = None,
        num_paths: int = 2,
    ) -> list[Task]:
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

        logger.info(f"Multi-path execution with {len(alternate_paths)} paths")
        return await self.forwarder.forward_multipath(task, alternate_paths)

    # ── IRQ ──────────────────────────────────────────────────────────

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

    async def request_retry(
        self,
        task: Task,
        source: NodeID,
        target: NodeID,
        feedback: str,
    ) -> None:
        """
        Request a retry of a specific node with feedback.

        Injects the retry request into the task context so execute_with_retry
        can pick it up.
        """
        task.context["_retry_requested"] = {
            "target": target,
            "feedback": feedback,
            "source": source,
        }
        await self.fire_irq(
            source=source,
            irq_type=IRQType.RETRY_REQUESTED,
            payload={"target": target, "feedback": feedback},
            reason=f"Retry requested for {target}: {feedback}",
            task=task,
            target=target,
            priority=IRQPriority.HIGH,
        )

    # ── Peer messaging ───────────────────────────────────────────────

    async def send_peer_message(
        self,
        source: NodeID,
        target: NodeID,
        payload: Any,
        task_id: str = "",
        message_type: str = "context",
    ) -> bool:
        msg = PeerMessage(
            source=source,
            target=target,
            task_id=task_id,
            payload=payload,
            message_type=message_type,
        )
        return await self.peer_network.send(msg)

    # ── IRQ handling ─────────────────────────────────────────────────

    def _handle_irq(self, irq: IRQ) -> None:
        """Conductor's own IRQ handler."""
        self._irq_log.append(irq)
        logger.info(
            f"Conductor received IRQ: {irq.irq_type.value} from {irq.source} "
            f"(priority={irq.priority.name}): {irq.reason}"
        )
        if irq.is_halt:
            logger.warning(f"HALT IRQ received from {irq.source}: {irq.reason}")

    # ── Introspection ────────────────────────────────────────────────

    def topology_summary(self) -> str:
        lines = [
            f"Conductor: {self.conductor_id}",
            f"Nodes: {len(self._nodes)}",
        ]
        if self._trust_domains:
            lines.append(f"Trust domains: {', '.join(self._trust_domains.keys())}")
        lines.append("")
        lines.append(self.registry.summary())
        lines.append("")
        lines.append("Peer links:")
        for link in self.peer_network._links:
            lines.append(f"  {link}")
        if not self.peer_network._links:
            lines.append("  (none)")
        return "\n".join(lines)
