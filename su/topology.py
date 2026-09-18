"""Fault network topology — Su et al. (2026), Section III-A (Def. 4, 5) and III-C.

A fault network is a rooted tree.  Nodes are faults (root = main fault), and
each edge carries an ordered attribute pair (e_strike, e_dip) drawn from
{"X", "Y", "P"} describing how the child fault relates to its parent in the
strike and in the dip direction respectively.

  "P"  non-intersecting in the voxel grid (not strict parallelism)
  "X"  crossing
  "Y"  branching / truncating junction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


EDGE_ATTRS = ("X", "Y", "P")

#: Definition 4 — the 7 admissible topologies; (X,Y) and (Y,X) are excluded.
VALID_EDGES = (
    ("X", "X"), ("X", "P"),
    ("Y", "Y"), ("Y", "P"),
    ("P", "X"), ("P", "Y"), ("P", "P"),
)
assert len(VALID_EDGES) == 7


@dataclass
class TreeNode:
    """T(i) of Eq. (30)."""
    id: int
    children: list[int] = field(default_factory=list)
    in_edge: Optional[tuple[str, str]] = None
    parent_id: Optional[int] = None
    node_attr: dict = field(default_factory=dict)      # G(i), D(i) — Eq. (31)-(33)


class FaultTree:
    def __init__(self):
        self.nodes: dict[int, TreeNode] = {0: TreeNode(id=0)}

    def __len__(self):
        return len(self.nodes)

    def add_child(self, child_id: int, parent_id: int, edge: tuple[str, str]):
        """AddChild of Algorithm 3, lines 1-4."""
        self.nodes[child_id] = TreeNode(id=child_id, in_edge=edge, parent_id=parent_id)
        self.nodes[parent_id].children.append(child_id)
        self.__dict__.pop('_forests', None)

    def direction_forest(self, k: int):
        """Fig. 10(a): remove P edges in direction k, retaining every node."""
        if '_forests' not in self.__dict__:
            self._forests = {}
        if k not in self._forests:
            forest = FaultTree()
            forest.nodes = {i: TreeNode(id=i) for i in self.nodes}
            for i, node in self.nodes.items():
                if node.parent_id is not None and node.in_edge[k-1] != 'P':
                    forest.add_child(i, node.parent_id, node.in_edge)
            forest.components = {}
            for i in forest.nodes:
                root = i
                while forest.nodes[root].parent_id is not None:
                    root = forest.nodes[root].parent_id
                forest.components[i] = root
            self._forests[k] = forest
        return self._forests[k]

    def depth(self) -> int:
        """Tree depth in the paper's convention: root alone counts as 1."""
        def d(i):
            ch = self.nodes[i].children
            return 1 if not ch else 1 + max(d(c) for c in ch)
        return d(0)

    def edges(self) -> list[tuple[int, int, tuple[str, str]]]:
        return [(n.parent_id, n.id, n.in_edge)
                for n in self.nodes.values() if n.parent_id is not None]

    def describe(self) -> str:
        rows = [f"root 0 (Nf={len(self)}, depth={self.depth()})"]
        for p, c, e in sorted(self.edges()):
            rows.append(f"  {p} -> {c}  edge={e[0]},{e[1]}")
        return "\n".join(rows)


# --------------------------------------------------------------------------
# Algorithm 3 — ValidateEdge
# --------------------------------------------------------------------------

#: Constraint 1 (Algorithm 3, line 8): mutually exclusive sibling edges.
C_SIBLING = {("X", "X"), ("X", "P"), ("P", "X")}

# Constraint 2 (Algorithm 3, lines 10-19), transcribed from the published
# algorithm.  Note C_p^2 and C_p^3 each absorb C_p^1 and C_c^1, so a ("Y","Y")
# parent is caught by all three rules and may only take ("Y","Y") or ("P","P")
# children — consistent with Section III-C placing Y-type edges lowest in the
# hierarchy.
C_PARENT_1 = {("Y", "Y")}
C_CHILD_1 = {("Y", "P"), ("P", "Y")}

C_PARENT_2 = {("X", "P")} | C_PARENT_1 | C_CHILD_1
C_CHILD_2 = {("X", "X"), ("P", "X")}

C_PARENT_3 = {("P", "X")} | C_PARENT_1 | C_CHILD_1
C_CHILD_3 = {("X", "X"), ("X", "P")}


def validate_edge(tree: FaultTree, edge: tuple[str, str], parent_id: int) -> bool:
    """Algorithm 3, ValidateEdge(e, i) — transcribed from the published text."""
    node = tree.nodes[parent_id]

    # Constraint 1 — sibling edges
    c_prep = {edge} | {tree.nodes[c].in_edge for c in node.children}
    if C_SIBLING <= c_prep:
        return False

    # Constraint 2 — parent/child hierarchy
    e_p = node.in_edge
    if e_p is not None:
        if e_p in C_PARENT_1 and edge in C_CHILD_1:
            return False
        if e_p in C_PARENT_2 and edge in C_CHILD_2:
            return False
        if e_p in C_PARENT_3 and edge in C_CHILD_3:
            return False
    return True


# --------------------------------------------------------------------------
# Algorithm 2 — Tree construction
# --------------------------------------------------------------------------

def build_tree(edge_pool: list[tuple[str, str]], n_out: int,
               rng: np.random.Generator) -> FaultTree:
    """Algorithm 2.  |edge_pool| = Nf - 1, so the tree ends with Nf nodes."""
    tree = FaultTree()
    pool = list(edge_pool)
    queue = [0]
    k = 1

    # Step 1 — tree growth (lines 3-17)
    while queue and pool:
        i = int(rng.choice(len(queue)))
        node_id = queue.pop(i)
        r = n_out - len(tree.nodes[node_id].children)
        if r <= 0:
            continue
        # Algorithm 2 line 8 shuffles the pool itself, so iterate over edge
        # VALUES; indices into the original list go stale as edges are consumed.
        shuffled = [pool[i] for i in rng.permutation(len(pool))]
        for e in shuffled:
            if r <= 0 or not pool:
                break
            if e not in pool:                 # already consumed this round
                continue
            if validate_edge(tree, e, node_id):
                tree.add_child(k, node_id, e)
                queue.append(k)
                pool.remove(e)
                k += 1
                r -= 1

    # Step 2 — fallback assignment (lines 18-24)
    fallback = ("Y", "Y")
    while pool:
        candidates = [n for n in tree.nodes if len(tree.nodes[n].children) < n_out]
        if not candidates:
            break
        i = int(rng.choice(candidates))
        tree.add_child(k, i, fallback)
        pool.pop(int(rng.integers(len(pool))))
        k += 1

    return tree


def tree_from_spec(spec: list[tuple[int, tuple[str, str]]]) -> FaultTree:
    """Build a deterministic tree from an explicit [(parent, edge), ...] list.

    Used for the five named patterns of Fig. 15, where the paper prescribes the
    topology rather than sampling it.  Child ids are assigned 1..len(spec).
    """
    tree = FaultTree()
    for k, (parent, edge) in enumerate(spec, start=1):
        if edge not in VALID_EDGES:
            raise ValueError(f"{edge} is not one of the 7 admissible topologies")
        tree.add_child(k, parent, edge)
    return tree
