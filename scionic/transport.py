"""
Transport layer — enables distributed nodes.

Provides task serialization and transport abstractions so nodes
can run as separate processes, on separate machines, or in separate
containers.

Transports:
- LocalTransport: async queues (in-process, for testing)
- Future: Redis, WebSocket, NATS, gRPC adapters

The key insight: tasks already carry their own route (packet-carried
state), so the transport just needs to serialize/deserialize and
deliver to the next hop's queue.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Any, Optional, Protocol

from .types import (
    Hop,
    HopStatus,
    IRQ,
    IRQPriority,
    IRQType,
    NodeID,
    PeerMessage,
    Task,
)

logger = logging.getLogger(__name__)


# ── Serialization ────────────────────────────────────────────────────

def task_to_dict(task: Task) -> dict:
    """Serialize a Task to a JSON-compatible dict."""
    return {
        "id": task.id,
        "payload": task.payload,
        "path": task.path,
        "current_hop_index": task.current_hop_index,
        "hops": [_hop_to_dict(h) for h in task.hops],
        "trust_domain": task.trust_domain,
        "policy": task.policy,
        "context": {k: _safe_serialize(v) for k, v in task.context.items()},
        "irq_mask": [m.value for m in task.irq_mask],
        "created_at": task.created_at,
        "max_retries": task.max_retries,
        "metadata": task.metadata,
    }


def task_from_dict(d: dict) -> Task:
    """Deserialize a Task from a dict."""
    task = Task(
        id=d["id"],
        payload=d["payload"],
        path=d["path"],
        current_hop_index=d["current_hop_index"],
        trust_domain=d.get("trust_domain", ""),
        policy=d.get("policy", {}),
        context=d.get("context", {}),
        created_at=d.get("created_at", time.time()),
        max_retries=d.get("max_retries", 1),
        metadata=d.get("metadata", {}),
    )
    task.hops = [_hop_from_dict(h) for h in d.get("hops", [])]
    task.irq_mask = [IRQPriority(v) for v in d.get("irq_mask", [])]
    return task


def task_to_json(task: Task) -> str:
    """Serialize a Task to JSON string."""
    return json.dumps(task_to_dict(task), default=str)


def task_from_json(s: str) -> Task:
    """Deserialize a Task from JSON string."""
    return task_from_dict(json.loads(s))


def _hop_to_dict(hop: Hop) -> dict:
    return {
        "node_id": hop.node_id,
        "status": hop.status.value,
        "input_hash": hop.input_hash,
        "output_hash": hop.output_hash,
        "output": _safe_serialize(hop.output),
        "signature": hop.signature,
        "started_at": hop.started_at,
        "completed_at": hop.completed_at,
        "tokens_used": hop.tokens_used,
        "cost": hop.cost,
        "error": hop.error,
        "retry_of": hop.retry_of,
        "metadata": hop.metadata,
    }


def _hop_from_dict(d: dict) -> Hop:
    return Hop(
        node_id=d["node_id"],
        status=HopStatus(d["status"]),
        input_hash=d.get("input_hash", ""),
        output_hash=d.get("output_hash", ""),
        output=d.get("output"),
        signature=d.get("signature", ""),
        started_at=d.get("started_at", 0.0),
        completed_at=d.get("completed_at", 0.0),
        tokens_used=d.get("tokens_used", 0),
        cost=d.get("cost", 0.0),
        error=d.get("error", ""),
        retry_of=d.get("retry_of"),
        metadata=d.get("metadata", {}),
    )


def _safe_serialize(obj: Any) -> Any:
    """Make an object JSON-serializable."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    return str(obj)


# ── Transport Protocol ───────────────────────────────────────────────

class Transport(Protocol):
    """Interface for task delivery between nodes."""

    async def send(self, target: NodeID, task: Task) -> None:
        """Send a task to a target node."""
        ...

    async def receive(self, node_id: NodeID) -> Task:
        """Receive the next task for a node (blocks until available)."""
        ...

    async def send_irq(self, irq: IRQ) -> None:
        """Broadcast an IRQ."""
        ...

    async def send_peer(self, msg: PeerMessage) -> None:
        """Send a peer message."""
        ...


# ── Local Transport (async queues) ───────────────────────────────────

class LocalTransport:
    """
    In-process transport using asyncio queues.

    Each node gets its own queue. Tasks are serialized to JSON and
    back to prove the serialization path works (not just passing
    Python objects by reference).
    """

    def __init__(self, serialize: bool = True) -> None:
        self._queues: dict[NodeID, asyncio.Queue] = defaultdict(asyncio.Queue)
        self._irq_queue: asyncio.Queue = asyncio.Queue()
        self._peer_queues: dict[NodeID, asyncio.Queue] = defaultdict(asyncio.Queue)
        self.serialize = serialize  # If True, JSON round-trip to prove serialization

    async def send(self, target: NodeID, task: Task) -> None:
        """Send a task to a node's queue."""
        if self.serialize:
            data = task_to_json(task)
            logger.debug(f"Transport: {len(data)} bytes → {target}")
            await self._queues[target].put(data)
        else:
            await self._queues[target].put(task)

    async def receive(self, node_id: NodeID, timeout: float = 30.0) -> Task:
        """Receive the next task for a node."""
        try:
            item = await asyncio.wait_for(
                self._queues[node_id].get(), timeout=timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"No task received for {node_id} within {timeout}s")

        if self.serialize:
            return task_from_json(item)
        return item

    async def send_irq(self, irq: IRQ) -> None:
        await self._irq_queue.put(irq)

    async def receive_irq(self, timeout: float = 5.0) -> Optional[IRQ]:
        try:
            return await asyncio.wait_for(
                self._irq_queue.get(), timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    async def send_peer(self, msg: PeerMessage) -> None:
        await self._peer_queues[msg.target].put(msg)

    async def receive_peer(self, node_id: NodeID, timeout: float = 5.0) -> Optional[PeerMessage]:
        try:
            return await asyncio.wait_for(
                self._peer_queues[node_id].get(), timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    def pending_count(self, node_id: NodeID) -> int:
        return self._queues[node_id].qsize()
