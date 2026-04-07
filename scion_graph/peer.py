"""
PeerLink / PeerNetwork — direct node-to-node messaging.

Lateral channels that bypass the conductor. Nodes with explicit
peering relationships can exchange context directly — like SCION
peering links that shortcut the hierarchy.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from typing import Callable, Optional

from .types import NodeID, PeerMessage

logger = logging.getLogger(__name__)

PeerHandler = Callable[[PeerMessage], None]


class PeerLink:
    """A bidirectional link between two nodes."""

    def __init__(self, node_a: NodeID, node_b: NodeID) -> None:
        self.node_a = node_a
        self.node_b = node_b

    def connects(self, node_id: NodeID) -> bool:
        return node_id in (self.node_a, self.node_b)

    def other(self, node_id: NodeID) -> Optional[NodeID]:
        if node_id == self.node_a:
            return self.node_b
        if node_id == self.node_b:
            return self.node_a
        return None

    def __repr__(self) -> str:
        return f"PeerLink({self.node_a} <-> {self.node_b})"


class PeerNetwork:
    """
    Manages peer links and delivers peer messages.

    Nodes must have an explicit PeerLink to communicate directly.
    Messages are delivered without going through the conductor.
    """

    def __init__(self) -> None:
        self._links: list[PeerLink] = []
        self._handlers: dict[NodeID, list[PeerHandler]] = defaultdict(list)
        self._message_log: list[PeerMessage] = []

    def add_link(self, node_a: NodeID, node_b: NodeID) -> PeerLink:
        """Create a peering relationship between two nodes."""
        link = PeerLink(node_a, node_b)
        self._links.append(link)
        logger.info(f"Peer link established: {node_a} <-> {node_b}")
        return link

    def remove_link(self, node_a: NodeID, node_b: NodeID) -> None:
        """Remove a peering relationship."""
        self._links = [
            l for l in self._links
            if not (l.connects(node_a) and l.connects(node_b))
        ]

    def subscribe(self, node_id: NodeID, handler: PeerHandler) -> None:
        """Register a handler for incoming peer messages."""
        self._handlers[node_id].append(handler)

    def unsubscribe(self, node_id: NodeID) -> None:
        self._handlers.pop(node_id, None)

    def has_link(self, node_a: NodeID, node_b: NodeID) -> bool:
        """Check if two nodes have a peering relationship."""
        return any(
            l.connects(node_a) and l.connects(node_b)
            for l in self._links
        )

    def peers_of(self, node_id: NodeID) -> list[NodeID]:
        """Get all peers of a node."""
        peers = []
        for link in self._links:
            other = link.other(node_id)
            if other is not None:
                peers.append(other)
        return peers

    async def send(self, message: PeerMessage) -> bool:
        """
        Send a peer message. Only delivers if a link exists.

        Returns True if delivered, False if no link exists.
        """
        if not self.has_link(message.source, message.target):
            logger.warning(
                f"No peer link between {message.source} and {message.target}"
            )
            return False

        self._message_log.append(message)

        handlers = self._handlers.get(message.target, [])
        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Peer handler error: {e}")

        logger.debug(
            f"Peer message {message.source} -> {message.target}: "
            f"{message.message_type}"
        )
        return True

    async def broadcast_to_peers(
        self, source: NodeID, payload: any, task_id: str = "", message_type: str = "context"
    ) -> int:
        """Send a message to all peers of a node."""
        peers = self.peers_of(source)
        delivered = 0
        for peer in peers:
            msg = PeerMessage(
                source=source,
                target=peer,
                task_id=task_id,
                payload=payload,
                message_type=message_type,
            )
            if await self.send(msg):
                delivered += 1
        return delivered

    def message_log(self, task_id: Optional[str] = None) -> list[PeerMessage]:
        if task_id:
            return [m for m in self._message_log if m.task_id == task_id]
        return list(self._message_log)
