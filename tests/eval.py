#!/usr/bin/env python3
"""
scionic eval — written from scratch by someone who just read the docs.

Every test hits a real LLM. No mocks. If a feature is claimed in the
README, there's a test for it here.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Load API key
for line in (Path.home() / ".hermes" / ".env").read_text().split("\n"):
    if line.startswith("OPENROUTER_API_KEY="):
        API_KEY = line.split("=", 1)[1].strip()
        break
else:
    sys.exit("No OPENROUTER_API_KEY in ~/.hermes/.env")

from scionic import (
    Conductor, SmartConductor, TriageRouter, RoutingRule,
    FlowController, CircuitBreaker, CircuitState,
    IRQPriority, IRQType, PathPolicy,
    Task, Hop, HopStatus,
)
from scionic.adapters.llm import LLMNodeHandler
from scionic.adapters.hermes import HermesNodeHandler
from scionic.transport import LocalTransport, task_to_json, task_from_json

MODEL = "anthropic/claude-haiku-4.5"
BASE = "https://openrouter.ai/api/v1"
PASS = 0
FAIL = 0
T0 = time.time()


def ok(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}: {detail}")


def node(caps, prompt, max_tokens=64, temp=0.0):
    return LLMNodeHandler(
        model=MODEL, api_key=API_KEY, base_url=BASE,
        node_capabilities=caps, system_prompt=prompt,
        max_tokens=max_tokens, temperature=temp,
    )


# ═════════════════════════════════════════════════════════════════════
#  1. Can I send a task through one node and get a signed hop back?
# ═════════════════════════════════════════════════════════════════════

async def test_single_hop():
    c = Conductor()
    c.add_node("oracle", node(["answer"], "Reply ONLY with the number 42."))
    t = c.create_task(payload="What is the answer?", path=["oracle"])
    r = await c.execute(t)
    ok("completes", r.is_complete)
    ok("1 hop", len(r.hops) == 1, str(len(r.hops)))
    h = r.hops[0]
    ok("hop signed", h.signature != "")
    ok("sig verifies", h.verify(c._nodes["oracle"].secret))
    ok("output has 42", "42" in str(r.context.get("oracle", "")),
       str(r.context.get("oracle", ""))[:80])
    ok("timing > 0", h.duration_ms > 0)
    ok("tokens > 0", h.tokens_used > 0, str(h.tokens_used))


# ═════════════════════════════════════════════════════════════════════
#  2. Does context flow from hop 1 to hop 2?
# ═════════════════════════════════════════════════════════════════════

async def test_context_flows():
    c = Conductor()
    c.add_node("gen", node(["gen"],
        "List exactly 3 fruit names, comma-separated. Nothing else.",
        max_tokens=32, temp=0.9))
    c.add_node("count", node(["count"],
        "Count items from the previous step. Reply ONLY with the number.",
        max_tokens=8))
    t = c.create_task(payload="fruits", path=["gen", "count"])
    r = await c.execute(t)
    ok("pipeline done", r.is_complete)
    ok("2 hops", len(r.hops) == 2, str(len(r.hops)))
    ok("chain verifies", all(h.verify(c._nodes[h.node_id].secret) for h in r.hops))
    ok("count says 3", "3" in str(r.context.get("count", "")),
       str(r.context.get("count", ""))[:40])


# ═════════════════════════════════════════════════════════════════════
#  3. Does path selection pick the cheapest node?
# ═════════════════════════════════════════════════════════════════════

async def test_path_selection():
    c = Conductor()
    c.add_node("cheap", node(["summarize"], "Summarize."), cost_per_call=0.001)
    c.add_node("pricey", node(["summarize"], "Summarize."), cost_per_call=0.10)
    t = c.create_task(payload="test", required_capabilities=["summarize"],
                      policy=PathPolicy(prefer_low_cost=True))
    ok("picks cheap", t.path == ["cheap"], str(t.path))
    r = await c.execute(t)
    ok("executes", r.is_complete)


# ═════════════════════════════════════════════════════════════════════
#  4. Can I run the same task on two paths at once?
# ═════════════════════════════════════════════════════════════════════

async def test_multipath():
    c = Conductor()
    c.add_node("a", node(["answer"], "Reply with ONLY the word ALPHA. Nothing else."))
    c.add_node("b", node(["answer"], "Reply with ONLY the word BETA. Nothing else."))
    t = Task(payload="Identify yourself.")
    rs = await c.forwarder.forward_multipath(t, [["a"], ["b"]])
    ok("2 results", len(rs) == 2, str(len(rs)))
    ok("both done", all(r.is_complete for r in rs))
    a_out = str(rs[0].context.get("a", ""))
    b_out = str(rs[1].context.get("b", ""))
    ok("path a has output", "ALPHA" in a_out.upper(), a_out[:40])
    ok("path b has output", "BETA" in b_out.upper(), b_out[:40])


# ═════════════════════════════════════════════════════════════════════
#  5. Do IRQs propagate between hops?
# ═════════════════════════════════════════════════════════════════════

async def test_irq():
    c = Conductor()
    c.add_node("s1", node(["step"], "Say hello."))
    c.add_node("s2", node(["step"], "Say world."))
    t = c.create_task(payload="test", path=["s1", "s2"])
    t = await c._nodes["s1"].execute(t)
    ok("hop 1 done", t.hops[0].status == HopStatus.COMPLETE)
    n = await c.fire_irq(source="s1", irq_type=IRQType.CONTEXT_UPDATE,
                         reason="extra info", task=t, priority=IRQPriority.HIGH)
    ok("irq delivered", n > 0, str(n))
    ok("irq logged", len(c._irq_log) > 0)
    t = await c._nodes["s2"].execute(t)
    ok("hop 2 done", t.is_complete)


# ═════════════════════════════════════════════════════════════════════
#  6. Does masking suppress delivery to nodes but not the conductor?
# ═════════════════════════════════════════════════════════════════════

async def test_irq_mask():
    c = Conductor()
    c.add_node("w", node(["work"], "ok"))
    t = c.create_task(payload="x", path=["w"])
    c.irq_bus.mask(t.id, IRQPriority.LOW)
    n = await c.fire_irq(source="ext", irq_type=IRQType.CONTEXT_UPDATE,
                         task=t, priority=IRQPriority.LOW)
    ok("conductor got it", len(c._irq_log) > 0)
    ok("only conductor (masked)", n == 1, str(n))
    n2 = await c.fire_irq(source="ext", irq_type=IRQType.CONTEXT_UPDATE,
                          task=t, priority=IRQPriority.HIGH)
    ok("high not masked", n2 > 1, str(n2))


# ═════════════════════════════════════════════════════════════════════
#  7. Does a peer message get injected into node context?
# ═════════════════════════════════════════════════════════════════════

async def test_peer_inject():
    c = Conductor()
    c.add_node("sender", node(["send"], "ok"))
    c.add_node("receiver", node(["recv"],
        "If you see peer messages in context, say PEER_FOUND. Otherwise say NONE.",
        max_tokens=16))
    c.add_peer_link("sender", "receiver")
    await c.send_peer_message("sender", "receiver", payload="secret hint")
    t = c.create_task(payload="check peers", path=["receiver"])
    r = await c.execute(t)
    ok("task done", r.is_complete)
    ok("peer key in context", "_peer_messages_for_receiver" in r.context,
       str(list(r.context.keys())))
    ok("traceroute shows peer", "peer" in r.traceroute().lower())
    ok("hop records peer count", r.hops[0].peer_messages_received == 1 if r.hops else False)


# ═════════════════════════════════════════════════════════════════════
#  8. Do trust domains block cross-domain hops?
# ═════════════════════════════════════════════════════════════════════

async def test_trust_domains():
    c = Conductor(verify_signatures=False)
    c.add_trust_domain("safe", "Safe")
    c.add_trust_domain("unsafe", "Unsafe")
    c.add_node("a", node(["step"], "ok"), trust_domain="safe")
    c.add_node("b", node(["step"], "ok"), trust_domain="unsafe")
    t = c.create_task(payload="x", path=["a", "b"])
    r = await c.execute(t)
    ok("cross-domain blocked",
       any("Trust domain" in (h.error or "") for h in r.hops),
       str([h.error for h in r.hops if h.error]))

    # Now with peering
    c2 = Conductor(verify_signatures=False)
    c2.add_trust_domain("safe", "Safe", allowed_peers=["unsafe"])
    c2.add_trust_domain("unsafe", "Unsafe", allowed_peers=["safe"])
    c2.add_node("a", node(["step"], "Say A."), trust_domain="safe")
    c2.add_node("b", node(["step"], "Say B."), trust_domain="unsafe")
    t2 = c2.create_task(payload="x", path=["a", "b"])
    r2 = await c2.execute(t2)
    ok("peered domains pass", r2.is_complete)


# ═════════════════════════════════════════════════════════════════════
#  9. Do trust domains enforce capability restrictions?
# ═════════════════════════════════════════════════════════════════════

async def test_domain_capabilities():
    c = Conductor(verify_signatures=False)
    c.add_trust_domain("classify_only", "Classify",
                       allowed_capabilities=["triage", "classify"])
    c.add_node("wrong", node(["execute"], "do stuff"), trust_domain="classify_only")
    t = c.create_task(payload="x", path=["wrong"])
    r = await c.execute(t)
    ok("capability blocked",
       any("blocked" in (h.error or "").lower() for h in r.hops),
       str([h.error for h in r.hops if h.error]))


# ═════════════════════════════════════════════════════════════════════
# 10. Does a tampered signature halt forwarding?
# ═════════════════════════════════════════════════════════════════════

async def test_sig_tamper():
    c = Conductor(verify_signatures=True)
    c.add_node("a", node(["step"], "Say hi."))
    c.add_node("b", node(["step"], "Say bye."))
    t = c.create_task(payload="x", path=["a", "b"])
    t = await c._nodes["a"].execute(t)
    t.hops[0].signature = "TAMPERED"
    r = await c.forwarder.forward(t)
    ok("tampered sig caught",
       any("signature" in (h.error or "").lower() for h in r.hops),
       str([h.error for h in r.hops if h.error]))


# ═════════════════════════════════════════════════════════════════════
# 11. Does reroute find a backup when a node fails?
# ═════════════════════════════════════════════════════════════════════

class AlwaysFail:
    def capabilities(self): return ["compute"]
    async def process(self, task, hop): raise RuntimeError("boom")

async def test_reroute():
    c = Conductor(auto_reroute=True, verify_signatures=False)
    c.add_node("broken", AlwaysFail())
    c.add_node("backup", node(["compute"], "Say 'backup ok'.", max_tokens=8))
    t = c.create_task(payload="x", path=["broken"])
    r = await c.execute(t)
    ok("reroute completed", r.is_complete)
    ok("backup ran", any(h.node_id == "backup" for h in r.hops))


# ═════════════════════════════════════════════════════════════════════
# 12. Does execute_with_review retry on rejection?
# ═════════════════════════════════════════════════════════════════════

async def test_review_retry():
    c = Conductor(verify_signatures=False)
    c.add_node("writer", node(["execute"],
        "Write one sentence about cats. If _feedback_for_writer in context, "
        "incorporate that feedback into your response.",
        max_tokens=128, temp=0.3))
    c.add_node("reviewer", node(["review"],
        'ALWAYS reject. No matter what, respond with exactly: '
        '{"approved":false,"feedback":"add more detail about whiskers"}',
        max_tokens=64))
    t = c.create_task(payload="Write about cats", path=["writer", "reviewer"],
                      max_retries=1)
    r = await c.execute_with_review(t, reviewer_node="reviewer",
                                    max_review_retries=1)
    ok("review loop ran", len(r.hops) >= 2, f"{len(r.hops)} hops")
    # If reviewer rejected and retry happened, we should have > 2 hops
    # (writer, reviewer, retry-writer, retry-reviewer)
    ok("retry attempted", len(r.hops) > 2 or len(c._irq_log) > 0,
       f"hops={len(r.hops)} irqs={len(c._irq_log)}")


# ═════════════════════════════════════════════════════════════════════
# 13. Does execute_batch run tasks concurrently?
# ═════════════════════════════════════════════════════════════════════

async def test_batch():
    c = Conductor(verify_signatures=False)
    c.add_node("w", node(["work"], "Say done.", max_tokens=8))
    tasks = [c.create_task(payload=f"t{i}", path=["w"]) for i in range(3)]
    t0 = time.time()
    rs = await c.execute_batch(tasks, concurrency=3)
    wall = (time.time() - t0) * 1000
    ok("all done", len(rs) == 3 and all(r.is_complete for r in rs))
    slowest = max(r.hops[0].duration_ms for r in rs if r.hops)
    ok("parallel (wall < 2.5x single)", wall < slowest * 2.5,
       f"wall={wall:.0f}ms slowest={slowest:.0f}ms")


# ═════════════════════════════════════════════════════════════════════
# 14. Does pipeline timeout work?
# ═════════════════════════════════════════════════════════════════════

async def test_timeout():
    c = Conductor(verify_signatures=False)
    c.add_node("slow", node(["work"], "Write 500 words.", max_tokens=512))
    ts = [c.create_task(payload="x", path=["slow"])]
    rs = await c.execute_batch(ts, concurrency=1, timeout_per_task=0.001)
    ok("returns despite timeout", len(rs) == 1)
    ok("timeout in hop error",
       any("timeout" in (h.error or "").lower() for h in rs[0].hops),
       str([h.error for h in rs[0].hops if h.error]))


# ═════════════════════════════════════════════════════════════════════
# 15. Does FlowController track pressure and trip breakers?
# ═════════════════════════════════════════════════════════════════════

async def test_flow():
    fc = FlowController(circuit_failure_threshold=2, circuit_cooldown_seconds=0.3)
    fc.register_node("fast", 3)
    fc.register_node("slow", 1)
    ok("fast available", fc.can_accept("fast"))
    fc.record_start("slow")
    ok("slow full", not fc.can_accept("slow"))
    fc.record_success("slow", 100)
    ok("slow released", fc.can_accept("slow"))
    fc.record_start("fast")
    fc.record_start("slow")
    ok("picks lower pressure", fc.select_by_pressure(["fast", "slow"]) == "fast")
    fc.record_success("fast", 50)
    fc.record_success("slow", 200)
    fc.record_failure("slow")
    fc.record_failure("slow")
    ok("breaker opens", fc.get_breaker("slow").state == CircuitState.OPEN)
    ok("slow blocked", not fc.can_accept("slow"))
    await asyncio.sleep(0.4)
    ok("breaker half-open", fc.can_accept("slow"))
    fc.record_success("slow", 100)
    ok("breaker closes", fc.get_breaker("slow").state == CircuitState.CLOSED)


# ═════════════════════════════════════════════════════════════════════
# 16. Does TriageRouter route deterministically from triage output?
# ═════════════════════════════════════════════════════════════════════

async def test_triage_router():
    tr = TriageRouter(triage_node_id="triager", verify_signatures=False)
    tr.add_node("triager", node(["triage", "classify"],
        'Classify. JSON only: {"complexity":"trivial","needs_code":false,'
        '"needs_research":false,"needs_planning":false,"needs_terminal":false,'
        '"reformulated":"clearer version"}', max_tokens=128))
    tr.add_node("quick_answer", node(["answer", "quick"], "Answer directly."))
    tr.add_node("planner", node(["plan"], "Plan."))
    tr.add_node("executor", node(["execute"], "Execute."))
    tr.add_node("reviewer", node(["review"], "Review."))
    tr.add_default_rules()
    t = await tr.create_task_routed(payload="What is 2+2?")
    ok("trivial → quick_answer", t.path == ["quick_answer"], str(t.path))
    ok("routing = deterministic", t.metadata.get("routing") == "deterministic",
       str(t.metadata.get("routing")))
    ok("rule name recorded", t.metadata.get("routing_rule") == "trivial",
       str(t.metadata.get("routing_rule")))
    r = await tr.execute(t)
    ok("executes", r.is_complete)


# ═════════════════════════════════════════════════════════════════════
# 17. Are routing rules inspectable?
# ═════════════════════════════════════════════════════════════════════

async def test_rules_summary():
    tr = TriageRouter(verify_signatures=False)
    tr.add_node("triager", node(["triage"], "x"))
    tr.add_node("quick_answer", node(["answer"], "x"))
    tr.add_node("executor", node(["execute"], "x"))
    tr.add_node("planner", node(["plan"], "x"))
    tr.add_node("researcher", node(["research"], "x"))
    tr.add_node("hermes_executor", node(["execute", "terminal"], "x"))
    tr.add_node("reviewer", node(["review"], "x"))
    tr.add_default_rules()
    s = tr.rules_summary()
    ok("summary has rules", "trivial" in s and "full_pipeline" in s)
    ok("5 default rules", len(tr._routing_rules) == 5, str(len(tr._routing_rules)))
    ok("all rules named", all(r.name for r in tr._routing_rules))


# ═════════════════════════════════════════════════════════════════════
# 18. Does transport serialize and deserialize a real task?
# ═════════════════════════════════════════════════════════════════════

async def test_transport():
    c = Conductor()
    c.add_node("x", node(["step"], "Say ok."))
    t = c.create_task(payload="test", path=["x"])
    r = await c.execute(t)
    j = task_to_json(r)
    ok("serializes", len(j) > 0)
    back = task_from_json(j)
    ok("id preserved", back.id == r.id)
    ok("hops preserved", len(back.hops) == len(r.hops))
    ok("sigs preserved", all(h.signature != "" for h in back.hops))
    tr = LocalTransport(serialize=True)
    await tr.send("dest", r)
    got = await tr.receive("dest")
    ok("queue round-trip", got.id == r.id)


# ═════════════════════════════════════════════════════════════════════
# 19. Does the Hermes adapter actually call hermes?
# ═════════════════════════════════════════════════════════════════════

async def test_hermes():
    c = Conductor(verify_signatures=True)
    h = HermesNodeHandler(model=MODEL, node_capabilities=["answer"],
        system_prompt="Reply ONLY with the word HERMES.",
        max_turns=1, toolsets=[])
    c.add_node("h", h)
    t = c.create_task(payload="Identify yourself.", path=["h"])
    r = await c.execute(t)
    ok("hermes completes", r.is_complete)
    ok("hermes hop signed", r.hops[0].signature != "" if r.hops else False)
    ok("hermes hop succeeded",
       r.hops[0].status == HopStatus.COMPLETE if r.hops else False)


# ═════════════════════════════════════════════════════════════════════
# 20. Full 4-node pipeline with peer hints + signature chain
# ═════════════════════════════════════════════════════════════════════

async def test_full_pipeline():
    c = Conductor(verify_signatures=True)
    c.add_node("triager", node(["triage"],
        '{"priority":"high","type":"design"}', max_tokens=32))
    c.add_node("planner", node(["plan"],
        '{"subtasks":["design","implement"]}', max_tokens=64))
    c.add_node("executor", node(["execute"],
        "Execute based on plan. 2 sentences.", max_tokens=128))
    c.add_node("reviewer", node(["review"],
        '{"approved":true,"quality":"good"}', max_tokens=32))
    c.add_peer_link("planner", "executor")
    await c.send_peer_message("planner", "executor",
                              payload="Focus on API design.")
    t = c.create_task(payload="Design a caching layer",
                      path=["triager", "planner", "executor", "reviewer"])
    r = await c.execute(t)
    ok("4-hop done", r.is_complete and len(r.hops) == 4,
       f"complete={r.is_complete} hops={len(r.hops)}")
    ok("all signed", all(h.signature != "" for h in r.hops))
    ok("chain verifies",
       all(h.verify(c._nodes[h.node_id].secret) for h in r.hops))
    ok("peer in traceroute", "peer" in r.traceroute().lower())
    ok("all nodes have output",
       all(r.context.get(n) for n in ["triager", "planner", "executor", "reviewer"]))


# ═════════════════════════════════════════════════════════════════════
# 21. Does pip install work? (no sys.path hacks)
# ═════════════════════════════════════════════════════════════════════

async def test_packaging():
    import subprocess
    r = subprocess.run(
        ["python3", "-c", "import scionic; print(scionic.__version__)"],
        capture_output=True, text=True, timeout=10)
    ok("import works", r.returncode == 0, r.stderr.strip()[:80])
    ok("version 0.2.0", "0.2.0" in r.stdout)


# ═════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 70)
    print("  scionic eval — fresh, no mocks, real API calls")
    print(f"  Model: {MODEL}")
    print("=" * 70)

    tests = [
        ("1.  Single hop + signing", test_single_hop),
        ("2.  Context flows between hops", test_context_flows),
        ("3.  Path selection (cheapest)", test_path_selection),
        ("4.  Multi-path parallel", test_multipath),
        ("5.  IRQ propagation", test_irq),
        ("6.  IRQ masking", test_irq_mask),
        ("7.  Peer context injection", test_peer_inject),
        ("8.  Trust domain enforcement", test_trust_domains),
        ("9.  Domain capability restrictions", test_domain_capabilities),
        ("10. Signature tamper detection", test_sig_tamper),
        ("11. Reroute on failure", test_reroute),
        ("12. Review retry loop", test_review_retry),
        ("13. Batch execution (concurrent)", test_batch),
        ("14. Pipeline timeout", test_timeout),
        ("15. FlowController + circuit breaker", test_flow),
        ("16. TriageRouter deterministic routing", test_triage_router),
        ("17. Routing rules introspection", test_rules_summary),
        ("18. Transport serialization", test_transport),
        ("19. Hermes adapter", test_hermes),
        ("20. Full 4-node pipeline", test_full_pipeline),
        ("21. Packaging (pip install)", test_packaging),
    ]

    for name, fn in tests:
        print(f"\n{'─' * 70}\n  {name}\n{'─' * 70}")
        t0 = time.time()
        try:
            await fn()
        except Exception as e:
            ok(name, False, f"EXCEPTION: {e}")
            import traceback; traceback.print_exc()
        print(f"  [{time.time()-t0:.1f}s]")

    print(f"\n{'═' * 70}")
    print(f"  {PASS} passed, {FAIL} failed ({time.time()-T0:.1f}s)")
    print(f"{'═' * 70}")
    sys.exit(1 if FAIL else 0)

if __name__ == "__main__":
    asyncio.run(main())
