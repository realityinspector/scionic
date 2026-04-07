"""
BeaconRegistry — capability discovery and indexing.

Nodes broadcast Beacons advertising their capabilities.
The registry collects, indexes, and expires them.
PathSelector queries the registry to find nodes that match requirements.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from .types import Beacon, NodeID


class BeaconRegistry:
    """
    Collects and indexes capability beacons from nodes.

    Thread-safe. Beacons auto-expire based on their TTL.
    Nodes re-broadcast periodically to stay registered.
    """

    def __init__(self) -> None:
        self._beacons: dict[NodeID, Beacon] = {}
        self._lock = threading.Lock()

    def register(self, beacon: Beacon) -> None:
        """Register or update a node's beacon."""
        with self._lock:
            self._beacons[beacon.node_id] = beacon

    def deregister(self, node_id: NodeID) -> None:
        """Remove a node's beacon."""
        with self._lock:
            self._beacons.pop(node_id, None)

    def get(self, node_id: NodeID) -> Optional[Beacon]:
        """Get a specific node's beacon, if active."""
        with self._lock:
            beacon = self._beacons.get(node_id)
            if beacon and not beacon.is_expired:
                return beacon
            return None

    def find_by_capability(self, capability: str) -> list[Beacon]:
        """Find all active nodes that advertise a capability."""
        self._prune_expired()
        with self._lock:
            return [
                b for b in self._beacons.values()
                if b.matches_capability(capability) and not b.is_expired
            ]

    def find_by_trust_domain(self, domain: str) -> list[Beacon]:
        """Find all active nodes in a trust domain."""
        self._prune_expired()
        with self._lock:
            return [
                b for b in self._beacons.values()
                if b.trust_domain == domain and not b.is_expired
            ]

    def find_available(self) -> list[Beacon]:
        """Find all active nodes with available capacity."""
        self._prune_expired()
        with self._lock:
            return [
                b for b in self._beacons.values()
                if b.available_capacity > 0 and not b.is_expired
            ]

    def all_active(self) -> list[Beacon]:
        """Return all active (non-expired) beacons."""
        self._prune_expired()
        with self._lock:
            return [b for b in self._beacons.values() if not b.is_expired]

    def _prune_expired(self) -> None:
        """Remove expired beacons."""
        now = time.time()
        with self._lock:
            expired = [
                nid for nid, b in self._beacons.items()
                if now - b.timestamp > b.ttl_seconds
            ]
            for nid in expired:
                del self._beacons[nid]

    def __len__(self) -> int:
        self._prune_expired()
        with self._lock:
            return len(self._beacons)

    def summary(self) -> str:
        """Human-readable summary of registered nodes."""
        beacons = self.all_active()
        if not beacons:
            return "No active nodes."
        lines = [f"Active nodes ({len(beacons)}):"]
        for b in sorted(beacons, key=lambda x: x.node_id):
            caps = ", ".join(b.capabilities)
            load = f"{b.current_load}/{b.max_concurrency}"
            lines.append(f"  {b.node_id} [{b.trust_domain}] ({caps}) load={load}")
        return "\n".join(lines)
