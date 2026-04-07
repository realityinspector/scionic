"""
FlowController — fluid dynamics for agent graphs.

Tasks are fluid. Nodes are pipes with capacity. Backpressure builds
when nodes are overloaded. Circuit breakers trip when nodes fail.
Flow diverts around obstacles automatically.

This replaces dumb sequential forwarding with pressure-aware routing:
- Nodes report capacity via beacons (already built)
- FlowController tracks real-time pressure per node
- When a node is at capacity, flow diverts to alternates
- When a node fails repeatedly, circuit breaker opens
- Metrics track throughput, latency percentiles, error rates

The fluid metaphor: tasks find the path of least resistance.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .types import NodeID

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"      # Normal — requests flow through
    OPEN = "open"          # Tripped — requests rejected immediately
    HALF_OPEN = "half_open"  # Testing — one request allowed through


@dataclass
class CircuitBreaker:
    """
    Per-node circuit breaker.

    Tracks consecutive failures. When threshold is reached, opens the
    circuit (rejects requests immediately). After cooldown, enters
    half-open state — one request allowed through as a probe. If it
    succeeds, circuit closes. If it fails, circuit re-opens.
    """
    failure_threshold: int = 3
    cooldown_seconds: float = 30.0
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    total_failures: int = 0
    total_successes: int = 0

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.total_successes += 1
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker closed (probe succeeded)")

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.total_failures += 1
        self.last_failure_time = time.time()

        if self.consecutive_failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker OPEN after {self.consecutive_failures} failures"
            )

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.cooldown_seconds:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker half-open (cooldown elapsed)")
                return True
            return False
        # HALF_OPEN — allow one probe
        return True

    @property
    def error_rate(self) -> float:
        total = self.total_successes + self.total_failures
        if total == 0:
            return 0.0
        return self.total_failures / total


@dataclass
class NodePressure:
    """
    Real-time pressure metrics for a node.

    Pressure = current_load / max_capacity. At 1.0, the node is full.
    Above 1.0 means requests are queuing (shouldn't happen with flow control).
    """
    node_id: NodeID
    max_capacity: int = 1
    current_load: int = 0
    # Latency tracking (rolling window)
    latency_window: deque = field(default_factory=lambda: deque(maxlen=20))
    # Throughput tracking
    completed_count: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def pressure(self) -> float:
        """0.0 = idle, 1.0 = full, >1.0 = overloaded."""
        if self.max_capacity <= 0:
            return float('inf')
        return self.current_load / self.max_capacity

    @property
    def available(self) -> bool:
        return self.current_load < self.max_capacity

    @property
    def avg_latency_ms(self) -> float:
        if not self.latency_window:
            return 0.0
        return sum(self.latency_window) / len(self.latency_window)

    @property
    def p95_latency_ms(self) -> float:
        if not self.latency_window:
            return 0.0
        sorted_latencies = sorted(self.latency_window)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def throughput_per_sec(self) -> float:
        elapsed = time.time() - self.started_at
        if elapsed <= 0:
            return 0.0
        return self.completed_count / elapsed

    def record_start(self) -> None:
        self.current_load += 1

    def record_complete(self, latency_ms: float) -> None:
        self.current_load = max(0, self.current_load - 1)
        self.latency_window.append(latency_ms)
        self.completed_count += 1

    def record_failure(self) -> None:
        self.current_load = max(0, self.current_load - 1)


class FlowController:
    """
    Manages flow across the graph.

    Tracks pressure per node, manages circuit breakers, and provides
    flow-aware node selection — pick the node with lowest pressure
    among those with matching capabilities.
    """

    def __init__(
        self,
        circuit_failure_threshold: int = 3,
        circuit_cooldown_seconds: float = 30.0,
    ) -> None:
        self._pressure: dict[NodeID, NodePressure] = {}
        self._breakers: dict[NodeID, CircuitBreaker] = {}
        self._failure_threshold = circuit_failure_threshold
        self._cooldown = circuit_cooldown_seconds

    def register_node(self, node_id: NodeID, max_capacity: int = 1) -> None:
        self._pressure[node_id] = NodePressure(
            node_id=node_id, max_capacity=max_capacity
        )
        self._breakers[node_id] = CircuitBreaker(
            failure_threshold=self._failure_threshold,
            cooldown_seconds=self._cooldown,
        )

    def deregister_node(self, node_id: NodeID) -> None:
        self._pressure.pop(node_id, None)
        self._breakers.pop(node_id, None)

    def can_accept(self, node_id: NodeID) -> bool:
        """Check if a node can accept a new task (capacity + circuit)."""
        pressure = self._pressure.get(node_id)
        breaker = self._breakers.get(node_id)
        if pressure is None or breaker is None:
            return False
        return pressure.available and breaker.allow_request()

    def record_start(self, node_id: NodeID) -> None:
        if node_id in self._pressure:
            self._pressure[node_id].record_start()

    def record_success(self, node_id: NodeID, latency_ms: float) -> None:
        if node_id in self._pressure:
            self._pressure[node_id].record_complete(latency_ms)
        if node_id in self._breakers:
            self._breakers[node_id].record_success()

    def record_failure(self, node_id: NodeID) -> None:
        if node_id in self._pressure:
            self._pressure[node_id].record_failure()
        if node_id in self._breakers:
            self._breakers[node_id].record_failure()

    def select_by_pressure(
        self, candidates: list[NodeID]
    ) -> Optional[NodeID]:
        """
        From a list of candidate nodes, pick the one with lowest pressure
        that can accept a request. Returns None if all are blocked.
        """
        available = []
        for nid in candidates:
            if self.can_accept(nid):
                pressure = self._pressure.get(nid)
                if pressure:
                    available.append((nid, pressure.pressure))

        if not available:
            return None

        # Sort by pressure (lowest first), break ties by node_id for determinism
        available.sort(key=lambda x: (x[1], x[0]))
        return available[0][0]

    def get_pressure(self, node_id: NodeID) -> Optional[NodePressure]:
        return self._pressure.get(node_id)

    def get_breaker(self, node_id: NodeID) -> Optional[CircuitBreaker]:
        return self._breakers.get(node_id)

    def flow_summary(self) -> str:
        lines = ["Flow status:"]
        for nid in sorted(self._pressure.keys()):
            p = self._pressure[nid]
            b = self._breakers[nid]
            status = b.state.value
            line = (
                f"  {nid}: pressure={p.pressure:.1%}, "
                f"load={p.current_load}/{p.max_capacity}, "
                f"circuit={status}, "
                f"avg_latency={p.avg_latency_ms:.0f}ms, "
                f"errors={b.total_failures}"
            )
            lines.append(line)
        return "\n".join(lines)
