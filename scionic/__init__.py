"""
scionic: Path-aware graph execution middleware.

SCION-inspired routing primitives for agent orchestration, data pipelines,
and any domain where messages traverse a graph with sender-chosen,
cryptographically verified, multi-path routing.
"""

__version__ = "0.1.0"

from .types import (
    Task,
    Hop,
    Beacon,
    IRQ,
    PeerMessage,
    TrustDomain,
    PathPolicy,
    NodeID,
    Path,
    IRQPriority,
    IRQType,
    HopStatus,
)
from .registry import BeaconRegistry
from .path_selector import PathSelector
from .forwarder import TaskForwarder
from .irq_bus import IRQBus
from .peer import PeerLink, PeerNetwork
from .conductor import Conductor
from .node import Node, NodeHandler

__all__ = [
    "Task", "Hop", "Beacon", "IRQ", "PeerMessage", "TrustDomain",
    "PathPolicy", "NodeID", "Path", "IRQPriority", "IRQType", "HopStatus",
    "BeaconRegistry", "PathSelector", "TaskForwarder",
    "IRQBus", "PeerLink", "PeerNetwork",
    "Conductor", "Node", "NodeHandler",
]
