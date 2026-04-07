# scionic

Path-aware graph execution middleware. SCION-inspired routing primitives for agent orchestration.

## What

Takes the core ideas from [SCION](https://scion-architecture.net/) (the next-gen internet architecture) and applies them as a **general-purpose graph execution protocol**:

| SCION Concept | scionic | What It Does |
|---------------|-------------|--------------|
| Beaconing (PCBs) | `BeaconRegistry` | Nodes advertise capabilities; conductor discovers what's available |
| Path selection | `PathSelector` | Caller chooses the route based on cost, latency, trust, capability |
| Packet-carried state | `Task` | The task carries its own route + accumulated context from each hop |
| Hop fields | `Hop` (signed) | Each node signs its contribution — cryptographic execution proof |
| Traceroute | `task.traceroute()` | Live execution trace emerges from hop log |
| ISDs (trust domains) | `TrustDomain` | Isolation boundaries — failures/access don't cascade |
| Peering links | `PeerNetwork` | Direct node-to-node messaging that bypasses the conductor |
| **IRQ interrupts** | `IRQBus` | Any node can signal others async — "the premise changed, stop" |

## Why Not LangGraph / CrewAI / AutoGen

Those are all **tree-shaped orchestrators**: parent spawns children, children report back. No lateral communication, no path awareness, no graph.

scionic gives you:
- **Caller-chosen paths** — the sender decides the route, not the framework
- **Verifiable execution** — each hop is signed, producing an auditable trace
- **Lateral messaging** — agents peer directly without routing through the orchestrator
- **Async interrupts** — any node can fire an IRQ that propagates across the graph
- **Multi-path execution** — same task, multiple paths, compare results
- **No central orchestrator bottleneck** — tasks carry their own route

## Quick Start

```bash
pip install -e .
python examples/research_pipeline.py
```

Output:
```
DEMO 1: Basic Pipeline with Traceroute
Task 0528ed3e traceroute:
  [+] 1. researcher (101ms)
  [+] 2. analyst (151ms)
  [+] 3. writer (101ms)
```

## Usage

```python
import asyncio
from scionic import Conductor, PathPolicy

conductor = Conductor()

# Register nodes with capabilities
conductor.add_node("researcher", ResearcherHandler(),
                   cost_per_call=0.01, avg_latency_ms=100)
conductor.add_node("analyst", AnalystHandler(),
                   cost_per_call=0.02, avg_latency_ms=150)
conductor.add_node("writer", WriterHandler(),
                   cost_per_call=0.03, avg_latency_ms=100)

# Peer links for lateral messaging
conductor.add_peer_link("researcher", "analyst")

# Auto-select path by capabilities
task = conductor.create_task(
    payload="Research topic X",
    required_capabilities=["search", "analyze", "draft"],
    policy=PathPolicy(prefer_low_cost=True),
)

# Or specify path explicitly
task = conductor.create_task(
    payload="Research topic X",
    path=["researcher", "analyst", "writer"],
)

# Execute and inspect
result = asyncio.run(conductor.execute(task))
print(result.traceroute())
print(result.context["writer"])  # Final output
```

## Adapters

### Hermes Agent

```python
from scionic import Conductor
from scionic.adapters import HermesAdapter

conductor = Conductor()
adapter = HermesAdapter(conductor)

adapter.add_agent("researcher",
    capabilities=["search", "rag"],
    model="anthropic/claude-sonnet-4-6",
    system_prompt="You are a research specialist.")

adapter.add_agent("writer",
    capabilities=["draft", "edit"],
    model="anthropic/claude-opus-4-6",
    system_prompt="You are a technical writer.")
```

### Generic LLM (OpenRouter / Ollama / any OpenAI-compatible)

```python
from scionic.adapters import LLMNodeHandler

handler = LLMNodeHandler(
    model="anthropic/claude-sonnet-4-6",
    api_key="sk-or-...",
    base_url="https://openrouter.ai/api/v1",
    node_capabilities=["analyze"],
    system_prompt="You are an analyst.",
)
conductor.add_node("analyst", handler)
```

## IRQ Interrupts

```python
from scionic import IRQType, IRQPriority

# Any node can fire an interrupt
await conductor.fire_irq(
    source="analyst",
    irq_type=IRQType.PREMISE_INVALID,
    reason="Competitor was acquired — reframe analysis",
    task=task,
    priority=IRQPriority.HIGH,
)

# Conductor can mask low-priority IRQs per task
conductor.irq_bus.mask(task.id, IRQPriority.LOW)
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Conductor                           │
│  BeaconRegistry · PathSelector · IRQBus          │
├─────────────────────────────────────────────────┤
│              TaskForwarder                       │
│  Routes tasks hop-by-hop along their path        │
├─────────────────────────────────────────────────┤
│              Nodes                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │researcher│←→│ analyst  │  │  writer  │      │
│  │  (search)│  │(analyze) │  │  (draft) │      │
│  └──────────┘  └──────────┘  └──────────┘      │
│       peer link ───┘                             │
├─────────────────────────────────────────────────┤
│              Adapters                            │
│  Hermes · LLM (OpenRouter) · Custom              │
└─────────────────────────────────────────────────┘
```

## Beyond Agents

The protocol is transport- and domain-agnostic. The same primitives work for:

- **Data pipelines** — ETL stages as nodes, with path selection by cost/latency
- **Payment routing** — choose routes by fee, jurisdiction, counterparty trust
- **Supply chain** — verified intermediaries with cryptographic hop signing
- **Mesh/IoT** — packet-carried state means no routing tables on constrained devices

## License

MIT
