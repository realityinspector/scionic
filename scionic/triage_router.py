"""
TriageRouter — deterministic routing with intent reformulation.

The "telepathic" conductor. Instead of asking an LLM to pick a path
(SmartConductor's approach — expensive, slow), the TriageRouter:

1. Runs the triage node ONCE (cheap, fast)
2. Triage classifies AND reformulates the task (what did you actually need?)
3. Deterministic routing rules map triage output to paths
4. LLM routing only fires when triage says "I don't know"

The reformulation is the key insight. User says "make me a website."
Triage reformulates: "Design a single-page HTML/CSS/JS website. Needs:
structure planning, implementation, review." Then the path is obvious.

This is the fluid dynamics: the triage node is a pressure sensor at
the inlet. It measures what the task IS, not just what the user SAID.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .conductor import Conductor
from .flow import FlowController
from .types import HopStatus, NodeID, PathPolicy, Task

logger = logging.getLogger(__name__)

@dataclass
class RoutingRule:
    """
    A named routing rule with a match function.

    Rules are evaluated in priority order. First match wins.
    The match function takes triage_data (dict) and returns a
    path (list[NodeID]) if it matches, or None if it doesn't.
    """
    name: str
    description: str
    match: Callable[[dict], Optional[list[NodeID]]]
    priority: int = 0  # Lower = evaluated first

    def __repr__(self) -> str:
        return f"RoutingRule({self.name!r}, priority={self.priority})"


class TriageRouter(Conductor):
    """
    Deterministic-first conductor with fluid routing.

    Routing hierarchy:
    1. Explicit path (if caller provides one)
    2. Routing rules (fast, deterministic, from triage output)
    3. Capability matching (PathSelector, no LLM)
    4. LLM routing (SmartConductor-style, only as last resort)

    The triage node runs first. Its JSON output drives everything.
    """

    def __init__(
        self,
        triage_node_id: str = "triager",
        llm_model: str = "",
        llm_api_key: str = "",
        llm_base_url: str = "https://openrouter.ai/api/v1",
        circuit_failure_threshold: int = 3,
        circuit_cooldown_seconds: float = 30.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.triage_node_id = triage_node_id
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self._routing_rules: list[RoutingRule] = []
        self.flow = FlowController(
            circuit_failure_threshold=circuit_failure_threshold,
            circuit_cooldown_seconds=circuit_cooldown_seconds,
        )

    # ── Node management (with flow registration) ─────────────────────

    def add_node(self, node_id, handler, max_concurrency=1, **kwargs):
        node = super().add_node(
            node_id, handler, max_concurrency=max_concurrency, **kwargs
        )
        self.flow.register_node(node_id, max_capacity=max_concurrency)
        return node

    def remove_node(self, node_id):
        super().remove_node(node_id)
        self.flow.deregister_node(node_id)

    # ── Routing rules ────────────────────────────────────────────────

    def add_routing_rule(self, rule: RoutingRule) -> None:
        """
        Add a named routing rule.

        Rules are evaluated in priority order (lower = first).
        First match wins.

        Example:
            router.add_routing_rule(RoutingRule(
                name="simple_facts",
                description="Trivial questions → quick_answer",
                match=lambda t: ["quick_answer"] if t.get("complexity") == "trivial" else None,
                priority=10,
            ))
        """
        self._routing_rules.append(rule)
        self._routing_rules.sort(key=lambda r: r.priority)

    def add_default_rules(self) -> None:
        """
        Add sensible default routing rules.

        Rules and their match conditions (fully visible):

        1. trivial (p=10): complexity=trivial|simple → quick_answer
           (or hermes_executor if needs_tools)
        2. code (p=20): needs_code|needs_tools|needs_terminal
           → planner → hermes_executor → reviewer
        3. research_only (p=30): needs_research AND NOT needs_planning
           → researcher → executor → reviewer
        4. planning_only (p=40): needs_planning AND NOT needs_research
           → planner → executor|hermes_executor → reviewer
        5. full_pipeline (p=50): needs_planning AND needs_research
           → planner → researcher → executor|hermes_executor → reviewer
        """
        router = self

        self._routing_rules = [
            RoutingRule(
                name="trivial",
                description="Simple/trivial tasks → quick_answer (or hermes for tools)",
                priority=10,
                match=lambda t: (
                    router._pick_with_flow(["hermes_executor"])
                    if t.get("complexity") in ("trivial", "simple")
                    and (t.get("needs_tools") or t.get("needs_terminal"))
                    else router._pick_with_flow(["quick_answer"])
                    if t.get("complexity") in ("trivial", "simple")
                    else None
                ),
            ),
            RoutingRule(
                name="code",
                description="Code/terminal tasks → planner → hermes_executor → reviewer",
                priority=20,
                match=lambda t: (
                    router._pick_with_flow(["planner", "hermes_executor", "reviewer"])
                    if t.get("needs_code") or t.get("needs_tools") or t.get("needs_terminal")
                    else None
                ),
            ),
            RoutingRule(
                name="research_only",
                description="Research (no planning) → researcher → executor → reviewer",
                priority=30,
                match=lambda t: (
                    router._pick_with_flow(["researcher", "executor", "reviewer"])
                    if t.get("needs_research") and not t.get("needs_planning")
                    else None
                ),
            ),
            RoutingRule(
                name="planning_only",
                description="Planning (no research) → planner → executor → reviewer",
                priority=40,
                match=lambda t: (
                    router._pick_with_flow([
                        "planner",
                        "hermes_executor" if t.get("needs_tools") else "executor",
                        "reviewer",
                    ])
                    if t.get("needs_planning") and not t.get("needs_research")
                    else None
                ),
            ),
            RoutingRule(
                name="full_pipeline",
                description="Complex (plan+research) → planner → researcher → executor → reviewer",
                priority=50,
                match=lambda t: (
                    router._pick_with_flow([
                        "planner", "researcher",
                        "hermes_executor" if t.get("needs_tools") else "executor",
                        "reviewer",
                    ])
                    if t.get("needs_planning") and t.get("needs_research")
                    else None
                ),
            ),
        ]

    def rules_summary(self) -> str:
        """Human-readable summary of active routing rules."""
        if not self._routing_rules:
            return "No routing rules configured."
        lines = ["Routing rules (evaluated in priority order):"]
        for rule in self._routing_rules:
            lines.append(f"  [{rule.priority:2d}] {rule.name}: {rule.description}")
        return "\n".join(lines)

    def _pick_with_flow(self, path: list[NodeID]) -> Optional[list[NodeID]]:
        """
        Validate a candidate path against flow state.

        For each node in the path, check if it can accept. If not,
        try to find an alternate with the same capability.
        """
        result = []
        for nid in path:
            if nid not in self._nodes:
                # Node doesn't exist — try to find alternate by capability
                alt = self._find_alternate(nid)
                if alt:
                    result.append(alt)
                else:
                    return None  # Can't build this path
            elif self.flow.can_accept(nid):
                result.append(nid)
            else:
                # Node exists but can't accept — find alternate
                alt = self._find_alternate_by_capability(nid)
                if alt:
                    result.append(alt)
                else:
                    result.append(nid)  # Use it anyway, will queue
        return result

    def _find_alternate(self, node_id: NodeID) -> Optional[NodeID]:
        """Find any node that exists, for a missing node_id."""
        return None  # No magic guessing

    def _find_alternate_by_capability(self, node_id: NodeID) -> Optional[NodeID]:
        """Find another node with the same capabilities."""
        node = self._nodes.get(node_id)
        if not node:
            return None

        caps = node.handler.capabilities()
        if not caps:
            return None

        self.refresh_beacons()
        for cap in caps:
            candidates = self.registry.find_by_capability(cap)
            for beacon in candidates:
                if beacon.node_id == node_id:
                    continue
                if self.flow.can_accept(beacon.node_id):
                    return beacon.node_id

        return None

    # ── Task creation with triage-first routing ──────────────────────

    async def create_task_routed(
        self,
        payload: Any,
        policy: Optional[PathPolicy] = None,
        trust_domain: str = "default",
        max_retries: Optional[int] = None,
    ) -> Task:
        """
        Create a task with triage-first routing.

        1. Run triage node to classify and reformulate
        2. Apply routing rules to triage output
        3. Fall back to capability matching if no rule matches
        4. Fall back to LLM routing as last resort
        """
        task = Task(
            payload=payload,
            trust_domain=trust_domain,
            policy=policy.__dict__ if policy else {},
            max_retries=max_retries if max_retries is not None else self.max_retries,
        )

        # Step 1: Run triage
        triage_data = await self._run_triage(task)

        # Reformulate payload if triage suggests it
        if triage_data and triage_data.get("reformulated"):
            original = task.payload
            task.payload = triage_data["reformulated"]
            task.metadata["original_payload"] = str(original)
            logger.info(f"Reformulated: {str(original)[:50]} → {str(task.payload)[:50]}")

        # Step 2: Try routing rules
        if triage_data:
            for rule in self._routing_rules:
                path = rule.match(triage_data)
                if path:
                    valid = all(nid in self._nodes for nid in path)
                    if valid:
                        task.path = path
                        task.metadata["routing"] = "deterministic"
                        task.metadata["routing_rule"] = rule.name
                        task.metadata["routing_rule_desc"] = rule.description
                        task.metadata["triage"] = triage_data
                        logger.info(
                            f"Rule '{rule.name}' matched: {' → '.join(path)}"
                        )
                        return task

        # Step 3: Try capability matching
        if triage_data:
            caps = self._infer_capabilities(triage_data)
            if caps:
                try:
                    self.refresh_beacons()
                    scored = self.path_selector.select(
                        required_capabilities=caps, policy=policy
                    )
                    if scored:
                        task.path = scored[0].path
                        task.metadata["routing"] = "capability_match"
                        logger.info(f"Capability-routed: {' → '.join(task.path)}")
                        return task
                except Exception:
                    pass

        # Step 4: LLM routing as last resort
        if self.llm_model and self.llm_api_key:
            try:
                path = await self._llm_route(task.payload)
                if path:
                    task.path = path
                    task.metadata["routing"] = "llm_fallback"
                    logger.info(f"LLM-routed (fallback): {' → '.join(path)}")
                    return task
            except Exception as e:
                logger.warning(f"LLM routing failed: {e}")

        # Nothing worked — use a sensible default
        task.path = self._default_path()
        task.metadata["routing"] = "default_fallback"
        logger.info(f"Default-routed: {' → '.join(task.path)}")
        return task

    async def _run_triage(self, task: Task) -> Optional[dict]:
        """Run the triage node and parse its JSON output."""
        triage_node = self._nodes.get(self.triage_node_id)
        if not triage_node:
            logger.warning(f"No triage node '{self.triage_node_id}' registered")
            return None

        triage_task = Task(
            payload=task.payload,
            path=[self.triage_node_id],
        )

        try:
            result = await triage_node.execute(triage_task)
            raw = str(result.context.get(self.triage_node_id, ""))
            return self._parse_triage_json(raw)
        except Exception as e:
            logger.warning(f"Triage execution failed: {e}")
            return None

    def _parse_triage_json(self, raw: str) -> Optional[dict]:
        """Resilient JSON extraction from LLM output."""
        clean = raw.strip()

        # Strip markdown fences
        if clean.startswith("```"):
            lines = clean.split("\n")
            lines = [l for l in lines[1:] if l.strip() != "```"]
            clean = "\n".join(lines).strip()

        # Try direct parse first
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        import re
        match = re.search(r'\{[^{}]*\}', clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f"Triage parse failed, raw: {clean[:100]}")
        return None

    def _infer_capabilities(self, triage_data: dict) -> list[str]:
        """Infer required capabilities from triage output."""
        caps = []
        if triage_data.get("needs_research"):
            caps.append("research")
        if triage_data.get("needs_planning"):
            caps.append("plan")
        if triage_data.get("needs_code") or triage_data.get("needs_tools"):
            caps.append("execute")
        if not caps:
            caps.append("execute")
        caps.append("review")
        return caps

    def _default_path(self) -> list[NodeID]:
        """Sensible default when all routing fails."""
        if "executor" in self._nodes and "reviewer" in self._nodes:
            return ["executor", "reviewer"]
        # Just use whatever nodes exist
        return list(self._nodes.keys())[:2]

    async def _llm_route(self, payload: Any) -> Optional[list[NodeID]]:
        """LLM routing — identical to SmartConductor but only used as fallback."""
        try:
            import httpx
        except ImportError:
            return None

        self.refresh_beacons()
        beacons = self.registry.all_active()
        if not beacons:
            return None

        node_descriptions = []
        for b in sorted(beacons, key=lambda x: x.node_id):
            caps = ", ".join(b.capabilities)
            node_descriptions.append(f"- {b.node_id}: [{caps}]")

        nodes_text = "\n".join(node_descriptions)
        node_ids = [b.node_id for b in beacons]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.llm_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": (
                            "Pick the best path. Respond with ONLY a JSON array "
                            f"of node IDs.\nAvailable:\n{nodes_text}"
                        )},
                        {"role": "user", "content": str(payload)},
                    ],
                    "max_tokens": 128,
                    "temperature": 0.0,
                },
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines[1:] if l.strip() != "```"]
            content = "\n".join(lines).strip()

        path = json.loads(content)
        return [n for n in path if n in node_ids] or None

    # ── Execution with flow tracking ─────────────────────────────────

    async def execute(self, task: Task) -> Task:
        """Execute with flow tracking — record pressure and circuit state."""
        logger.info(f"Executing task {task.id[:8]}: {' → '.join(task.path)}")

        # Track flow for each hop
        for node_id in task.path:
            self.flow.record_start(node_id)

        result = await self.forwarder.forward(task)

        # Update flow metrics from hops
        for hop in result.hops:
            if hop.status == HopStatus.COMPLETE:
                self.flow.record_success(hop.node_id, hop.duration_ms)
            elif hop.status == HopStatus.FAILED:
                self.flow.record_failure(hop.node_id)
            # Release pressure for hops that didn't run
            # (because pipeline stopped early)
        for node_id in result.path[len(result.hops):]:
            if node_id in self.flow._pressure:
                self.flow._pressure[node_id].current_load = max(
                    0, self.flow._pressure[node_id].current_load - 1
                )

        # Auto-reroute on failure
        if self.auto_reroute and result.failed_hops:
            result = await self._attempt_reroute(result)

        logger.info(f"Task {result.id[:8]} complete:\n{result.traceroute()}")
        return result
