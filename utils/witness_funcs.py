import numpy as np

import numpy as np


import numpy as np


def build_in_out(edge_index, num_nodes):
    """
    Build both list-based and set-based in/out adjacencies.

    - lists preserve multiplicity  -> use for degree-based tasks
    - sets remove duplicates       -> use for fan/cycle/SG2/BP2 tasks
    - self-loops are ignored
    """
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()

    out_list = [[] for _ in range(num_nodes)]
    in_list = [[] for _ in range(num_nodes)]

    for u, v in zip(src, dst):
        u = int(u)
        v = int(v)
        if u == v:
            continue
        out_list[u].append(v)
        in_list[v].append(u)

    out_set = [set(neighs) for neighs in out_list]
    in_set = [set(neighs) for neighs in in_list]

    return out_list, in_list, out_set, in_set


def build_unique_in_out(edge_index, num_nodes):
    """
    Backward-compatible wrapper if other code still imports this.
    """
    _, _, out_set, in_set = build_in_out(edge_index, num_nodes)
    return out_set, in_set


def _singleton_witnesses_from_mask(mask):
    return [(int(node),) for node, ok in enumerate(mask) if ok]


def fan_in_three(out_set, in_set):
    """
    At least 3 unique incoming neighbors.
    """
    return _singleton_witnesses_from_mask(
        [len(in_set[node]) > 3 for node in range(len(in_set))]
    )


def fan_out_three(out_set, in_set):
    """
    At least 3 unique outgoing neighbors.
    """
    return _singleton_witnesses_from_mask(
        [len(out_set[node]) > 3 for node in range(len(out_set))]
    )


def deg_in_three(out_list, in_list):
    """
    At least 3 incoming edges, counting multiplicity.
    """
    return _singleton_witnesses_from_mask(
        [len(in_list[node]) > 3 for node in range(len(in_list))]
    )


def deg_out_three(out_list, in_list):
    """
    At least 3 outgoing edges, counting multiplicity.
    """
    return _singleton_witnesses_from_mask(
        [len(out_list[node]) > 3 for node in range(len(out_list))]
    )


"""
witness_funcs.py

Motif / witness detection utilities used by:
- multigraph_generation.py to compute per-graph motifs and build node multi-label targets y
- downstream partitioning/splitting code to filter motif edges or build motif-only subgraphs

Conventions:
- Graphs are treated as directed.
- Self-loops are ignored.
- out_set[u] = set of unique out-neighbors of u
- in_set[v]  = set of unique in-neighbors of v

All motif functions return a list of tuples (the "witnesses") whose node IDs define the motif instance.
"""


def canon_cycle_rotation(cyc):
    """
    Canonicalize a directed cycle up to rotation (NOT reversal).

    Example:
        (2, 3, 1) -> (1, 2, 3)
    Direction is preserved; we only rotate so that the smallest node ID comes first.

    Args:
        cyc: Iterable of node IDs describing the directed cycle in order.

    Returns:
        A tuple representing the same directed cycle but rotated into canonical form.
    """
    cyc = list(cyc)
    m = min(cyc)
    k = cyc.index(m)
    return tuple(cyc[k:] + cyc[:k])


def build_unique_in_out(edge_index, num_nodes):
    """
    Build unique in/out adjacency sets from a PyG-style edge_index.

    Notes:
        - Ignores self-loops.
        - Deduplicates edges by using Python sets (multi-edges collapse to one neighbor entry).

    Args:
        edge_index: Tensor/ndarray-like of shape [2, E] with src in row 0 and dst in row 1.
        num_nodes: Number of nodes in the graph.

    Returns:
        out_set: List[set[int]] where out_set[u] are unique out-neighbors of u.
        in_set:  List[set[int]] where in_set[v] are unique in-neighbors of v.
    """
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    out_set = [set() for _ in range(num_nodes)]
    in_set = [set() for _ in range(num_nodes)]

    for u, v in zip(src, dst):
        u = int(u)
        v = int(v)
        if u == v:
            continue
        out_set[u].add(v)
        in_set[v].add(u)

    return out_set, in_set


def cycles_C2(out_set, in_set):
    """
    Detect 2-cycles (mutual edges): a <-> b.

    Representation:
        Returns unordered pairs (a, b) with a < b to avoid duplicates.

    Args:
        out_set: Unique out-neighbor sets (see build_unique_in_out).
        in_set:  Unique in-neighbor sets (unused here but kept for API symmetry).

    Returns:
        List of tuples (a, b) with a < b.
    """
    cycles = set()
    n = len(out_set)
    for a in range(n):
        for b in out_set[a]:
            # 2-cycle exists if a->b and b->a
            if a != b and a in out_set[b]:
                cycles.add(tuple(sorted((a, b))))
    return list(cycles)


def cycles_C3(out_set, in_set):
    """
    Detect directed 3-cycles: a -> b -> c -> a.

    Canonicalization:
        Cycle tuples are normalized by rotation so that the smallest node ID comes first.

    Args:
        out_set: Unique out-neighbor sets.
        in_set:  Unique in-neighbor sets (unused here).

    Returns:
        List of canonical tuples (a, b, c) describing the cycle in directed order.
    """
    cycles = set()
    n = len(out_set)

    for a in range(n):
        for b in out_set[a]:
            if b == a:
                continue
            for c in out_set[b]:
                if c == a or c == b:
                    continue
                if a in out_set[c]:
                    cycles.add(canon_cycle_rotation((a, b, c)))

    return list(cycles)


def cycles_C4(out_set, in_set):
    """
    Detect directed 4-cycles: a -> b -> c -> d -> a.

    See cycles_C3 for tuple canonicalization rules.

    Args:
        out_set: Unique out-neighbor sets.
        in_set:  Unique in-neighbor sets (unused here).

    Returns:
        List of canonical tuples (a, b, c, d).
    """
    cycles = set()
    n = len(out_set)

    for a in range(n):
        for b in out_set[a]:
            if b == a:
                continue
            for c in out_set[b]:
                if c in (a, b):
                    continue
                for d in out_set[c]:
                    if d in (a, b, c):
                        continue
                    if a in out_set[d]:
                        cycles.add(canon_cycle_rotation((a, b, c, d)))

    return list(cycles)


def cycles_C5(out_set, in_set):
    """
    Detect directed 5-cycles: a -> b -> c -> d -> e -> a.

    Args:
        out_set: Unique out-neighbor sets.
        in_set:  Unique in-neighbor sets (unused here).

    Returns:
        List of canonical tuples (a, b, c, d, e).
    """
    cycles = set()
    n = len(out_set)

    for a in range(n):
        for b in out_set[a]:
            if b == a:
                continue
            for c in out_set[b]:
                if c in (a, b):
                    continue
                for d in out_set[c]:
                    if d in (a, b, c):
                        continue
                    for e in out_set[d]:
                        if e in (a, b, c, d):
                            continue
                        if a in out_set[e]:
                            cycles.add(canon_cycle_rotation((a, b, c, d, e)))

    return list(cycles)


def cycles_C6(out_set, in_set):
    """
    Detect directed 6-cycles: a -> b -> c -> d -> e -> f -> a.

    Args:
        out_set: Unique out-neighbor sets.
        in_set:  Unique in-neighbor sets (unused here).

    Returns:
        List of canonical tuples (a, b, c, d, e, f).
    """
    cycles = set()
    n = len(out_set)

    for a in range(n):
        for b in out_set[a]:
            if b == a:
                continue
            for c in out_set[b]:
                if c in (a, b):
                    continue
                for d in out_set[c]:
                    if d in (a, b, c):
                        continue
                    for e in out_set[d]:
                        if e in (a, b, c, d):
                            continue
                        for f in out_set[e]:
                            if f in (a, b, c, d, e):
                                continue
                            if a in out_set[f]:
                                cycles.add(canon_cycle_rotation((a, b, c, d, e, f)))

    return list(cycles)


def SG2(out_set, in_set):
    """
    Detect 'scatter_gather' witnesses of the form (source, j1, j2, sink):

        source -> j1 -> sink
        source -> j2 -> sink

    Where j1 and j2 are distinct predecessors of sink, and source is a common predecessor of both.

    Canonicalization:
        For a fixed sink, predecessors are iterated in sorted order, which yields stable tuple output.

    Args:
        out_set: Unique out-neighbor sets (unused here; kept for API symmetry).
        in_set:  Unique in-neighbor sets.

    Returns:
        List of tuples (source, j1, j2, sink).
    """
    n = len(out_set)
    W = set()

    for sink in range(n):
        sink_preds = in_set[sink]
        if len(sink_preds) < 2:
            continue

        # convert once; sorting gives stable canonical order
        sink_preds_sorted = sorted(sink_preds)

        L = len(sink_preds_sorted)
        for left in range(L - 1):
            j1 = sink_preds_sorted[left]
            if j1 == sink:
                continue
            left_preds = in_set[j1]
            if not left_preds:
                continue

            for right in range(left + 1, L):
                j2 = sink_preds_sorted[right]
                if j2 == sink:
                    continue
                right_preds = in_set[j2]
                if not right_preds:
                    continue

                # iterate smaller in-set for speed
                if len(left_preds) <= len(right_preds):
                    small, big = left_preds, right_preds
                else:
                    small, big = right_preds, left_preds

                for source in small:
                    if source == sink or source == j1 or source == j2:
                        continue
                    if source in big:
                        W.add((int(source), int(j1), int(j2), int(sink)))
    return list(W)


def sort_bp2(L1, L2, R1, R2):
    """
    Helper for biclique witnesses: sort left nodes and right nodes internally.

    Args:
        L1, L2: Left-part node IDs.
        R1, R2: Right-part node IDs.

    Returns:
        (l1, l2, r1, r2) with l1 < l2 and r1 < r2.
    """
    l1, l2 = (L1, L2) if L1 < L2 else (L2, L1)
    r1, r2 = (R1, R2) if R1 < R2 else (R2, R1)
    return l1, l2, r1, r2


def BP2(out_set, in_set):
    """
    Detect 2x2 directed biclique witnesses of the form (l1, l2, r1, r2):

        l1 -> r1, l1 -> r2
        l2 -> r1, l2 -> r2

    Implementation:
        - Choose two left nodes (l1, l2)
        - Find their common out-neighbors (candidate rights)
        - Choose two right nodes (r1, r2) from that intersection

    Notes:
        - Self-membership is filtered (rights cannot be equal to l1/l2).
        - Witness tuples are canonicalized by sorting lefts and rights.

    Args:
        out_set: Unique out-neighbor sets.
        in_set:  Unique in-neighbor sets (unused here).

    Returns:
        List of tuples (l1, l2, r1, r2) with l1 < l2 and r1 < r2.
    """
    n = len(in_set)
    W = set()

    for l1 in range(n):
        out1 = out_set[l1]
        if len(out1) < 2:
            continue
        for l2 in range(l1 + 1, n):
            out2 = out_set[l2]
            if len(out2) < 2:
                continue

            common = list(out1.intersection(out2))  # candidates for right nodes
            if len(common) < 2:
                continue

            common.sort()
            # choose unordered pairs (r1, r2) from common
            for i in range(len(common) - 1):
                r1 = common[i]
                if r1 in (l1, l2):
                    continue
                for j in range(i + 1, len(common)):
                    r2 = common[j]
                    if r2 in (l1, l2) or r2 == r1:
                        continue
                    l1, l2, r1, r2 = sort_bp2(l1, l2, r1, r2)
                    W.add((int(l1), int(l2), int(r1), int(r2)))
    return list(W)
