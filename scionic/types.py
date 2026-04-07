"""
Core data structures for scionic.

These are the middleware primitives — transport-agnostic, adapter-agnostic.
Network adapters (SCION, TCP) and agent adapters (Hermes, Claude, generic)
both map to and from these types.
"""

from __future__ import annotations

import hashlib
import hmac
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# A node identifier — opaque string, unique within a topology
NodeID = str

# An ordered list of node IDs representing a route through the graph
Path = list[NodeID]


class HopStatus(str, Enum):
    """Status of a single hop in a task's execution trace."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRIED = "retried"


class IRQPriority(int, Enum):
    """Interrupt priority levels. Higher number = higher priority."""
    LOW = 1       # Informational — conductor may mask
    NORMAL = 5    # Default — delivered unless explicitly masked
    HIGH = 8      # Important — always delivered, may pause downstream
    CRITICAL = 10 # Halt — stops the pipeline


class IRQType(str, Enum):
    """Categories of interrupt signals."""
    CONTEXT_UPDATE = "context_update"   # New info relevant to other nodes
    PREMISE_INVALID = "premise_invalid" # The task's assumption is wrong
    HALT = "halt"                       # Stop all downstream processing
    CAPACITY_CHANGE = "capacity_change" # Node load/availability changed
    RESULT_READY = "result_ready"       # Async result available for pickup
    RETRY_REQUESTED = "retry_requested" # Ask conductor to retry a hop


@dataclass
class TrustDomain:
    """
    Isolation boundary — maps to SCION's ISD.

    Nodes in the same trust domain can peer freely.
    Cross-domain traffic requires explicit policy.
    Failures in one domain don't cascade to others.
    """
    id: str
    name: str
    description: str = ""
    allowed_peers: list[str] = field(default_factory=list)

    def allows_peer(self, other_domain_id: str) -> bool:
        return other_domain_id == self.id or other_domain_id in self.allowed_peers


@dataclass
class Beacon:
    """
    Capability advertisement — maps to SCION's PCB (Path Construction Beacon).

    Nodes broadcast beacons to announce what they can do.
    The BeaconRegistry collects these; the PathSelector uses them to build routes.
    Beacons expire after `ttl_seconds`.
    """
    node_id: NodeID
    capabilities: list[str]
    trust_domain: str
    cost_per_call: float = 0.0
    avg_latency_ms: float = 0.0
    max_concurrency: int = 1
    current_load: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    ttl_seconds: float = 60.0
    timestamp: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl_seconds

    @property
    def available_capacity(self) -> int:
        return max(0, self.max_concurrency - self.current_load)

    def matches_capability(self, required: str) -> bool:
        return required in self.capabilities


@dataclass
class Hop:
    """
    A single step in a task's execution trace — maps to SCION's hop field.

    Each node that processes a task appends a Hop to the task's `hops` list.
    This creates a traceroute-style execution log.
    The signature proves this node actually processed the task.
    """
    node_id: NodeID
    status: HopStatus = HopStatus.PENDING
    input_hash: str = ""
    output_hash: str = ""
    output: Any = None
    signature: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0
    error: str = ""
    retry_of: Optional[str] = None  # node_id of the hop this retried
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return 0.0

    def sign(self, secret: str) -> None:
        """Sign this hop with the node's secret key."""
        payload = f"{self.node_id}:{self.input_hash}:{self.output_hash}:{self.started_at}"
        self.signature = hmac.new(
            secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()

    def verify(self, secret: str) -> bool:
        """Verify this hop's signature."""
        payload = f"{self.node_id}:{self.input_hash}:{self.output_hash}:{self.started_at}"
        expected = hmac.new(
            secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(self.signature, expected)


@dataclass
class Task:
    """
    A unit of work moving through the graph — maps to a SCION packet.

    The task carries its own route (packet-carried forwarding state).
    Each node reads the path, processes if it's the current hop, and forwards.
    The hops list grows as the task traverses the graph — a live traceroute.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: Any = None
    path: Path = field(default_factory=list)
    current_hop_index: int = 0
    hops: list[Hop] = field(default_factory=list)
    trust_domain: str = ""
    policy: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    irq_mask: list[IRQPriority] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    max_retries: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def current_node(self) -> Optional[NodeID]:
        if 0 <= self.current_hop_index < len(self.path):
            return self.path[self.current_hop_index]
        return None

    @property
    def next_node(self) -> Optional[NodeID]:
        idx = self.current_hop_index + 1
        if 0 <= idx < len(self.path):
            return self.path[idx]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_hop_index >= len(self.path)

    @property
    def failed_hops(self) -> list[Hop]:
        return [h for h in self.hops if h.status == HopStatus.FAILED]

    @property
    def retry_count(self) -> int:
        return sum(1 for h in self.hops if h.retry_of is not None)

    def advance(self) -> None:
        """Move to the next hop."""
        self.current_hop_index += 1

    def rewind(self, steps: int = 1) -> None:
        """Move back N hops (for retry)."""
        self.current_hop_index = max(0, self.current_hop_index - steps)

    def inject_feedback(self, node_id: str, feedback: str) -> None:
        """Inject feedback into context for a node to see on retry."""
        key = f"_feedback_for_{node_id}"
        existing = self.context.get(key, "")
        if existing:
            self.context[key] = f"{existing}\n{feedback}"
        else:
            self.context[key] = feedback

    def traceroute(self) -> str:
        """Human-readable execution trace."""
        lines = [f"Task {self.id[:8]} traceroute:"]
        for i, hop in enumerate(self.hops):
            status_icon = {
                HopStatus.COMPLETE: "+",
                HopStatus.FAILED: "!",
                HopStatus.IN_PROGRESS: ">",
                HopStatus.SKIPPED: "-",
                HopStatus.PENDING: ".",
                HopStatus.RETRIED: "R",
            }.get(hop.status, "?")
            line = f"  [{status_icon}] {i+1}. {hop.node_id}"
            if hop.retry_of:
                line += " (retry)"
            if hop.duration_ms:
                line += f" ({hop.duration_ms:.0f}ms"
                if hop.tokens_used:
                    line += f", {hop.tokens_used} tokens"
                if hop.cost:
                    line += f", ${hop.cost:.4f}"
                line += ")"
            if hop.error:
                line += f" ERROR: {hop.error}"
            lines.append(line)
        if not self.is_complete and self.current_node:
            remaining_idx = self.current_hop_index
            for j, node_id in enumerate(self.path[remaining_idx:]):
                marker = ">" if j == 0 else "."
                lines.append(f"  [{marker}] {len(self.hops)+j+1}. {node_id} (pending)")
        return "\n".join(lines)


@dataclass
class IRQ:
    """
    Interrupt signal — async notification propagated across the graph.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: NodeID = ""
    target: Optional[NodeID] = None
    task_id: str = ""
    irq_type: IRQType = IRQType.CONTEXT_UPDATE
    priority: IRQPriority = IRQPriority.NORMAL
    payload: Any = None
    reason: str = ""
    suggested_action: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def is_halt(self) -> bool:
        return self.irq_type == IRQType.HALT or self.priority == IRQPriority.CRITICAL

    def should_deliver(self, mask: list[IRQPriority]) -> bool:
        return self.priority not in mask


@dataclass
class PeerMessage:
    """
    Direct node-to-node message that bypasses the conductor.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: NodeID = ""
    target: NodeID = ""
    task_id: str = ""
    payload: Any = None
    message_type: str = "context"
    timestamp: float = field(default_factory=time.time)


@dataclass
class PathPolicy:
    """
    Constraints for path selection.
    """
    max_cost: Optional[float] = None
    max_latency_ms: Optional[float] = None
    required_capabilities: list[str] = field(default_factory=list)
    preferred_trust_domains: list[str] = field(default_factory=list)
    excluded_nodes: list[NodeID] = field(default_factory=list)
    excluded_trust_domains: list[str] = field(default_factory=list)
    min_path_diversity: int = 1
    prefer_low_cost: bool = False
    prefer_low_latency: bool = True
    require_same_trust_domain: bool = False
