"""
TaskForwarder — moves tasks through the graph along their path.

Each task carries its own route (packet-carried forwarding state).
The forwarder reads the route, finds the next node, and delivers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .node import Node
from .types import HopStatus, NodeID, Task

logger = logging.getLogger(__name__)


class TaskForwarder:
    """
    Routes tasks through registered nodes according to their path.

    Nodes register with the forwarder. When a task arrives, the forwarder
    looks up the current hop's node and delivers the task to it.
    """

    def __init__(self) -> None:
        self._nodes: dict[NodeID, Node] = {}

    def register_node(self, node: Node) -> None:
        """Register a node that can receive tasks."""
        self._nodes[node.node_id] = node

    def deregister_node(self, node_id: NodeID) -> None:
        self._nodes.pop(node_id, None)

    def get_node(self, node_id: NodeID) -> Optional[Node]:
        return self._nodes.get(node_id)

    async def forward(self, task: Task) -> Task:
        """
        Forward a task through its entire path.

        Executes each hop sequentially. If a hop fails, the task
        stops and returns with the failure recorded in the traceroute.
        """
        while not task.is_complete:
            current = task.current_node
            if current is None:
                break

            node = self._nodes.get(current)
            if node is None:
                logger.error(f"No node registered for {current}")
                from .types import Hop
                error_hop = Hop(
                    node_id=current,
                    status=HopStatus.FAILED,
                    error=f"Node {current} not found in forwarder",
                )
                task.hops.append(error_hop)
                break

            logger.info(f"Forwarding task {task.id[:8]} to {current}")
            task = await node.execute(task)

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
        import copy

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
