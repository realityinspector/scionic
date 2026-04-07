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
from typing import Any, Callable, Optional


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
    # Nodes are allowed to peer with these other domains
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
    capabilities: list[str]          # What this node can do: ["search", "summarize", "code"]
    trust_domain: str                # Which ISD this node belongs to
    cost_per_call: float = 0.0       # Estimated cost in dollars
    avg_latency_ms: float = 0.0      # Average processing time
    max_concurrency: int = 1         # How many tasks it can handle in parallel
    current_load: int = 0            # How many tasks it's currently processing
    metadata: dict[str, Any] = field(default_factory=dict)  # Adapter-specific data
    ttl_seconds: float = 60.0        # How long this beacon is valid
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
    input_hash: str = ""             # Hash of what this node received
    output_hash: str = ""            # Hash of what this node produced
    output: Any = None               # The actual output (can be large)
    signature: str = ""              # HMAC proving this node processed the task
    started_at: float = 0.0
    completed_at: float = 0.0
    tokens_used: int = 0             # LLM tokens consumed (if applicable)
    cost: float = 0.0                # Actual cost incurred
    error: str = ""                  # Error message if failed
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
    payload: Any = None              # The actual work to be done
    path: Path = field(default_factory=list)         # Ordered route through nodes
    current_hop_index: int = 0       # Which node should process next
    hops: list[Hop] = field(default_factory=list)    # Execution trace (traceroute)
    trust_domain: str = ""           # Which ISD this task belongs to
    policy: dict[str, Any] = field(default_factory=dict)  # Path selection constraints
    context: dict[str, Any] = field(default_factory=dict)  # Accumulated context from hops
    irq_mask: list[IRQPriority] = field(default_factory=list)  # Masked interrupt levels
    created_at: float = field(default_factory=time.time)
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

    def advance(self) -> None:
        """Move to the next hop."""
        self.current_hop_index += 1

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
            }.get(hop.status, "?")
            line = f"  [{status_icon}] {i+1}. {hop.node_id}"
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
            lines.append(f"  [>] {len(self.hops)+1}. {self.current_node} (next)")
            for node_id in self.path[self.current_hop_index + 1:]:
                lines.append(f"  [.] {self.path.index(node_id)+1}. {node_id} (pending)")
        return "\n".join(lines)


@dataclass
class IRQ:
    """
    Interrupt signal — async notification propagated across the graph.

    Any node can fire an IRQ. The conductor decides whether to mask it
    (based on priority) or propagate it to other nodes on the path.
    Maps to the creative "IRQ loops informing agents across the graph" concept.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: NodeID = ""              # Who fired the interrupt
    target: Optional[NodeID] = None  # Specific target, or None for broadcast
    task_id: str = ""                # Which task this relates to
    irq_type: IRQType = IRQType.CONTEXT_UPDATE
    priority: IRQPriority = IRQPriority.NORMAL
    payload: Any = None              # The interrupt data
    reason: str = ""                 # Human-readable explanation
    suggested_action: str = ""       # What the sender thinks should happen
    timestamp: float = field(default_factory=time.time)

    @property
    def is_halt(self) -> bool:
        return self.irq_type == IRQType.HALT or self.priority == IRQPriority.CRITICAL

    def should_deliver(self, mask: list[IRQPriority]) -> bool:
        """Check if this IRQ should be delivered given a priority mask."""
        return self.priority not in mask


@dataclass
class PeerMessage:
    """
    Direct node-to-node message that bypasses the conductor.

    Like SCION peering links — lateral communication between nodes
    that share a peering relationship. The conductor doesn't see these.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: NodeID = ""
    target: NodeID = ""
    task_id: str = ""                # Context: which task this relates to
    payload: Any = None
    message_type: str = "context"    # "context", "hint", "correction", "data"
    timestamp: float = field(default_factory=time.time)


@dataclass
class PathPolicy:
    """
    Constraints for path selection — what the caller cares about.

    The PathSelector uses these to rank candidate paths assembled from beacons.
    """
    max_cost: Optional[float] = None          # Total budget in dollars
    max_latency_ms: Optional[float] = None    # Max acceptable end-to-end latency
    required_capabilities: list[str] = field(default_factory=list)
    preferred_trust_domains: list[str] = field(default_factory=list)
    excluded_nodes: list[NodeID] = field(default_factory=list)
    min_path_diversity: int = 1               # Minimum distinct paths to return
    prefer_low_cost: bool = False
    prefer_low_latency: bool = True
