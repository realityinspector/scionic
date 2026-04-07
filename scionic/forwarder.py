"""
TaskForwarder — moves tasks through the graph along their path.

Each task carries its own route (packet-carried forwarding state).
The forwarder reads the route, finds the next node, and delivers.
Verifies hop signatures as tasks move through the chain.
Enforces trust domain boundaries.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import Optional

from .node import Node
from .types import Hop, HopStatus, NodeID, Task, TrustDomain

logger = logging.getLogger(__name__)


class TaskForwarder:
    """
    Routes tasks through registered nodes according to their path.

    Validates hop signatures (chain of trust) and enforces trust domain
    boundaries during forwarding.
    """

    def __init__(self, verify_signatures: bool = True) -> None:
        self._nodes: dict[NodeID, Node] = {}
        self._trust_domains: dict[str, TrustDomain] = {}
        self.verify_signatures = verify_signatures

    def register_node(self, node: Node) -> None:
        self._nodes[node.node_id] = node

    def deregister_node(self, node_id: NodeID) -> None:
        self._nodes.pop(node_id, None)

    def get_node(self, node_id: NodeID) -> Optional[Node]:
        return self._nodes.get(node_id)

    def register_trust_domain(self, domain: TrustDomain) -> None:
        self._trust_domains[domain.id] = domain

    def _check_trust_boundary(self, from_node: Optional[Node], to_node: Node) -> bool:
        """Check if forwarding across trust domain boundary is allowed."""
        if from_node is None:
            return True  # First hop, no boundary to check
        if from_node.trust_domain == to_node.trust_domain:
            return True  # Same domain

        # Check if the domains have a peering relationship
        from_domain = self._trust_domains.get(from_node.trust_domain)
        to_domain = self._trust_domains.get(to_node.trust_domain)

        if from_domain and from_domain.allows_peer(to_node.trust_domain):
            return True
        if to_domain and to_domain.allows_peer(from_node.trust_domain):
            return True

        # If no trust domains are registered, allow (backward compat)
        if not self._trust_domains:
            return True

        return False

    def _verify_last_hop(self, task: Task) -> bool:
        """Verify the signature of the most recent hop."""
        if not task.hops:
            return True
        last_hop = task.hops[-1]
        if last_hop.status != HopStatus.COMPLETE:
            return True  # Only verify completed hops

        node = self._nodes.get(last_hop.node_id)
        if node is None:
            logger.warning(f"Cannot verify hop from unknown node {last_hop.node_id}")
            return True  # Node may have been removed

        return last_hop.verify(node.secret)

    async def forward(self, task: Task) -> Task:
        """
        Forward a task through its entire path.

        At each hop:
        1. Verify the previous hop's signature (chain of trust)
        2. Check trust domain boundary
        3. Execute at the current node
        4. Check for failure
        """
        prev_node: Optional[Node] = None

        while not task.is_complete:
            current = task.current_node
            if current is None:
                break

            node = self._nodes.get(current)
            if node is None:
                logger.error(f"No node registered for {current}")
                error_hop = Hop(
                    node_id=current,
                    status=HopStatus.FAILED,
                    error=f"Node {current} not found in forwarder",
                )
                task.hops.append(error_hop)
                break

            # Verify previous hop signature
            if self.verify_signatures and not self._verify_last_hop(task):
                logger.error(f"Signature verification failed before {current}")
                error_hop = Hop(
                    node_id=current,
                    status=HopStatus.FAILED,
                    error=f"Previous hop signature verification failed",
                )
                task.hops.append(error_hop)
                break

            # Check trust domain boundary
            if not self._check_trust_boundary(prev_node, node):
                from_domain = prev_node.trust_domain if prev_node else "none"
                logger.error(
                    f"Trust domain boundary violation: "
                    f"{from_domain} → {node.trust_domain}"
                )
                error_hop = Hop(
                    node_id=current,
                    status=HopStatus.FAILED,
                    error=f"Trust domain boundary violation: {from_domain} → {node.trust_domain}",
                )
                task.hops.append(error_hop)
                break

            logger.info(f"Forwarding task {task.id[:8]} to {current}")
            task = await node.execute(task)
            prev_node = node

            # Check if the last hop failed
            if task.hops and task.hops[-1].status == HopStatus.FAILED:
                logger.error(
                    f"Task {task.id[:8]} failed at {current}: "
                    f"{task.hops[-1].error}"
                )
                break

        return task

    async def forward_multipath(
        self, task: Task, alternate_paths: list[list[NodeID]]
    ) -> list[Task]:
        """
        Execute a task across multiple paths simultaneously.

        Creates a copy of the task for each path and runs them in parallel.
        Returns all results — the caller (or conductor) decides which to use.
        """
        tasks = []
        for path in alternate_paths:
            t = copy.deepcopy(task)
            t.path = path
            t.current_hop_index = 0
            t.hops = []
            tasks.append(t)

        results = await asyncio.gather(
            *(self.forward(t) for t in tasks),
            return_exceptions=True,
        )

        completed = []
        for r in results:
            if isinstance(r, Task):
                completed.append(r)
            else:
                logger.error(f"Multipath execution error: {r}")

        return completed
