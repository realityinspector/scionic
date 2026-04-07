"""
PathSelector — route assembly from beacons.

Given a task's requirements and available beacons, assemble candidate paths.
Rank by policy (cost, latency, trust domain, capability match).
Return multiple paths for multi-path execution.
"""

from __future__ import annotations

from itertools import permutations
from typing import Optional

from .registry import BeaconRegistry
from .types import Beacon, NodeID, Path, PathPolicy


class ScoredPath:
    """A candidate path with its composite score."""

    def __init__(self, path: Path, beacons: list[Beacon], score: float) -> None:
        self.path = path
        self.beacons = beacons
        self.score = score

    @property
    def total_cost(self) -> float:
        return sum(b.cost_per_call for b in self.beacons)

    @property
    def total_latency_ms(self) -> float:
        return sum(b.avg_latency_ms for b in self.beacons)

    @property
    def hop_count(self) -> int:
        return len(self.path)

    def __repr__(self) -> str:
        nodes = " → ".join(self.path)
        return f"ScoredPath({nodes}, score={self.score:.2f}, cost=${self.total_cost:.4f}, latency={self.total_latency_ms:.0f}ms)"


class PathSelector:
    """
    Assembles and ranks paths through the node graph.

    Uses beacons from the registry to find nodes that satisfy
    required capabilities, then ranks candidate orderings by policy.
    """

    def __init__(self, registry: BeaconRegistry) -> None:
        self.registry = registry

    def select(
        self,
        required_capabilities: list[str],
        policy: Optional[PathPolicy] = None,
        source: Optional[NodeID] = None,
        max_paths: int = 3,
    ) -> list[ScoredPath]:
        """
        Find and rank paths that satisfy the required capabilities.

        Each capability maps to one node. The selector finds the best
        node for each capability, then assembles orderings.
        """
        if policy is None:
            policy = PathPolicy()

        # For each required capability, find candidate nodes
        candidates_per_cap: list[list[Beacon]] = []
        for cap in required_capabilities:
            matches = self.registry.find_by_capability(cap)
            # Apply policy filters
            matches = [
                b for b in matches
                if b.node_id not in policy.excluded_nodes
                and b.available_capacity > 0
            ]
            if policy.preferred_trust_domains:
                preferred = [b for b in matches if b.trust_domain in policy.preferred_trust_domains]
                if preferred:
                    matches = preferred
            if not matches:
                return []  # Can't satisfy this capability
            candidates_per_cap.append(matches)

        # Build candidate paths: pick best node per capability
        paths = self._assemble_paths(
            required_capabilities, candidates_per_cap, policy, max_paths
        )

        # Sort by score (higher is better)
        paths.sort(key=lambda p: p.score, reverse=True)
        return paths[:max_paths]

    def _assemble_paths(
        self,
        capabilities: list[str],
        candidates_per_cap: list[list[Beacon]],
        policy: PathPolicy,
        max_paths: int,
    ) -> list[ScoredPath]:
        """Assemble candidate paths from per-capability beacon lists."""
        # Simple greedy: for each capability, try each candidate
        # For small graphs this is fine; for large ones we'd use beam search
        results: list[ScoredPath] = []

        # Generate candidate selections (pick one beacon per capability)
        selections = self._cartesian_select(candidates_per_cap, max_paths * 2)

        for selection in selections:
            path = [b.node_id for b in selection]
            score = self._score_path(selection, policy)

            # Apply hard policy constraints
            if policy.max_cost is not None:
                total_cost = sum(b.cost_per_call for b in selection)
                if total_cost > policy.max_cost:
                    continue
            if policy.max_latency_ms is not None:
                total_latency = sum(b.avg_latency_ms for b in selection)
                if total_latency > policy.max_latency_ms:
                    continue

            results.append(ScoredPath(path, list(selection), score))

        return results

    def _cartesian_select(
        self,
        candidates_per_cap: list[list[Beacon]],
        max_combos: int,
    ) -> list[tuple[Beacon, ...]]:
        """Generate candidate selections, one beacon per capability slot."""
        if not candidates_per_cap:
            return []

        # Sort each candidate list by cost (cheapest first) as a heuristic
        sorted_caps = [sorted(cs, key=lambda b: b.cost_per_call) for cs in candidates_per_cap]

        # Simple: take top candidates and generate combos
        # Limit each slot to top-3 to avoid explosion
        trimmed = [cs[:3] for cs in sorted_caps]

        results: list[tuple[Beacon, ...]] = []
        self._gen_combos(trimmed, 0, [], results, max_combos)
        return results

    def _gen_combos(
        self,
        slots: list[list[Beacon]],
        idx: int,
        current: list[Beacon],
        results: list[tuple[Beacon, ...]],
        limit: int,
    ) -> None:
        if len(results) >= limit:
            return
        if idx >= len(slots):
            results.append(tuple(current))
            return
        for beacon in slots[idx]:
            self._gen_combos(slots, idx + 1, current + [beacon], results, limit)

    def _score_path(self, beacons: tuple[Beacon, ...] | list[Beacon], policy: PathPolicy) -> float:
        """Score a path. Higher = better."""
        score = 100.0

        total_cost = sum(b.cost_per_call for b in beacons)
        total_latency = sum(b.avg_latency_ms for b in beacons)
        avg_load_ratio = sum(b.current_load / max(b.max_concurrency, 1) for b in beacons) / max(len(beacons), 1)

        if policy.prefer_low_cost:
            score -= total_cost * 100
        if policy.prefer_low_latency:
            score -= total_latency * 0.01

        # Penalize loaded nodes
        score -= avg_load_ratio * 20

        # Bonus for preferred trust domains
        if policy.preferred_trust_domains:
            in_preferred = sum(1 for b in beacons if b.trust_domain in policy.preferred_trust_domains)
            score += in_preferred * 10

        # Fewer hops is generally better
        score -= len(beacons) * 2

        return score
