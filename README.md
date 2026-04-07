# scionic

SCION-inspired path-aware graph execution middleware for agent orchestration.

Takes the core ideas from [SCION](https://scion-architecture.net/) — the next-gen internet architecture — and applies them as a general-purpose agent routing protocol. Zero dependencies (stdlib only; `httpx` optional for LLM adapter).

## Features

| Feature | What It Does |
|---------|--------------|
| **Packet-carried state** | Tasks carry their own route + accumulated context through the graph |
| **Hop signing** | Each node cryptographically signs its contribution — verifiable execution proof |
| **Traceroute** | Live execution trace with timing, tokens, and cost per hop |
| **Path selection** | Caller chooses the route by capability, cost, latency, or trust domain |
| **Multi-path** | Same task, multiple paths, parallel execution — compare results |
| **IRQ interrupts** | Any node can signal others async; conductor masks by priority |
| **IRQ retry** | Conductor re-executes failed hops with feedback injected |
| **Peer messaging** | Direct node-to-node context sharing that bypasses the conductor |
| **Trust domains** | Isolation boundaries — cross-domain hops blocked without peering |
| **Reroute on failure** | Conductor auto-finds alternate node with same capability |
| **SmartConductor** | LLM-driven routing — model reads beacons and assembles paths |
| **Transport layer** | JSON serialization + async queues for distributed nodes |

## Quick Start

```bash
pip install -e .
python tests/eval.py    # 62+ assertions, real OpenRouter API calls, no mocks
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  SmartConductor (LLM-driven) or Conductor (deterministic)│
│  BeaconRegistry · PathSelector · IRQBus · PeerNetwork    │
├──────────────────────────────────────────────────────────┤
│  TaskForwarder (hop verification + trust domain checks)  │
├──────────────────────────────────────────────────────────┤
│  Nodes                                                   │
│  ┌────────┐  ┌──────────┐  ┌────────┐  ┌────────────┐  │
│  │ linter │←→│ reviewer │  │ writer │  │ approver   │  │
│  │(Hermes)│  │  (LLM)   │  │ (LLM)  │  │   (LLM)    │  │
│  └────────┘  └──────────┘  └────────┘  └────────────┘  │
│     peer link ───┘                                       │
├──────────────────────────────────────────────────────────┤
│  Transport: LocalTransport (queues) | Redis | WebSocket  │
├──────────────────────────────────────────────────────────┤
│  Adapters: Hermes Agent | LLM (OpenRouter) | Custom      │
└──────────────────────────────────────────────────────────┘
```

## Usage

### Basic pipeline

```python
from scionic import Conductor
from scionic.adapters.llm import LLMNodeHandler

conductor = Conductor()
conductor.add_node("researcher", LLMNodeHandler(
    model="anthropic/claude-haiku-4.5", api_key="sk-or-...",
    node_capabilities=["search"], system_prompt="You research topics."))
conductor.add_node("writer", LLMNodeHandler(
    model="anthropic/claude-haiku-4.5", api_key="sk-or-...",
    node_capabilities=["draft"], system_prompt="You write reports."))

task = conductor.create_task(
    payload="Write about SCION networking",
    path=["researcher", "writer"],
)
result = await conductor.execute(task)
print(result.traceroute())
print(result.context["writer"])
```

### SmartConductor (LLM picks the path)

```python
from scionic import SmartConductor

conductor = SmartConductor(
    model="anthropic/claude-haiku-4.5",
    api_key="sk-or-...",
)
# Register nodes...
task = conductor.create_task(payload="Review this code for security issues")
# LLM reads beacons and decides: security_scanner → reviewer → approver
result = await conductor.execute(task)
```

### Hermes nodes (agents with tools)

```python
from scionic.adapters.hermes import HermesAdapter

adapter = HermesAdapter(conductor)
adapter.add_agent("coder",
    capabilities=["code", "terminal"],
    system_prompt="You write and test code.",
    toolsets=["terminal", "file"],
    max_turns=5)
```

### Trust domains

```python
conductor.add_trust_domain("internal", "Internal", allowed_peers=["dmz"])
conductor.add_trust_domain("dmz", "DMZ", allowed_peers=["internal"])
conductor.add_trust_domain("external", "External")  # isolated

conductor.add_node("secure", handler, trust_domain="internal")
conductor.add_node("gateway", handler, trust_domain="dmz")
conductor.add_node("untrusted", handler, trust_domain="external")

# internal → dmz: OK (peered)
# internal → external: BLOCKED (no peering)
```

### IRQ retry with feedback

```python
from scionic import IRQType, IRQPriority

# After execution, request a retry with feedback
await conductor.request_retry(
    task=result,
    source="reviewer",
    target="writer",
    feedback="Mention golden retrievers specifically.",
)
result = await conductor.execute_with_retry(result)
```

### Transport (distributed nodes)

```python
from scionic.transport import LocalTransport, task_to_json, task_from_json

transport = LocalTransport(serialize=True)  # JSON round-trip
await transport.send("node_b", task)
received = await transport.receive("node_b")  # Deserialized from JSON
```

## Code Review Pipeline (Example App)

```bash
python examples/code_review.py
```

Routes code through: linter (Hermes) → security scanner → code reviewer → approval gate. With trust domains, peer messaging, and full traceroute.

## Eval

```bash
python tests/eval.py
```

13 scenarios, 62+ assertions, all real OpenRouter API calls:

1. Basic forwarding + hop signing
2. Context accumulation across hops
3. Auto path selection by capability + cost
4. Multi-path parallel execution
5. IRQ interrupt propagation
6. IRQ masking
7. Peer context injection
8. Trust domain enforcement
9. Hop signature verification (tampered sig detection)
10. Reroute on failure
11. IRQ retry with feedback injection
12. Full 4-node pipeline with peer hints + signature chain
13. Traceroute readability
14. Hermes agent node
15. SmartConductor LLM routing
16. Transport serialization round-trip
17. Code review pipeline end-to-end

## Package Structure

```
scionic/
├── __init__.py           # Public API
├── types.py              # Task, Hop, Beacon, IRQ, PeerMessage, TrustDomain
├── registry.py           # BeaconRegistry — capability discovery
├── path_selector.py      # PathSelector — route assembly
├── node.py               # Node — hop signing, peer injection
├── forwarder.py          # TaskForwarder — verification, trust domains
├── irq_bus.py            # IRQBus — async interrupts with masking
├── peer.py               # PeerNetwork — lateral messaging
├── conductor.py          # Conductor — deterministic orchestration
├── smart_conductor.py    # SmartConductor — LLM-driven routing
├── transport.py          # Serialization + async queue transport
└── adapters/
    ├── hermes.py         # Hermes Agent (tools, terminal, memory)
    └── llm.py            # Any OpenAI-compatible API
```

## License

MIT
