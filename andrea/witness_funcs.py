import numpy as np

def canon_cycle_rotation(cyc):
    """
    Canonicalize a directed cycle up to rotation only.
    Example: (2,3,1) -> (1,2,3). Direction is preserved (no reversal).
    """
    cyc = list(cyc)
    m = min(cyc)
    k = cyc.index(m)
    return tuple(cyc[k:] + cyc[:k])

def build_unique_in_out(edge_index, num_nodes):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    out_set = [set() for _ in range(num_nodes)]
    in_set  = [set() for _ in range(num_nodes)]

    for u, v in zip(src, dst):
        u = int(u); v = int(v)
        if u == v:
            continue
        out_set[u].add(v)
        in_set[v].add(u)

    return out_set, in_set

def cycles_C2(out_set, in_set):
    cycles = set()
    n = len(out_set)
    for a in range(n):
        for b in out_set[a]:
            if a != b and a in out_set[b]:
                cycles.add(tuple(sorted((a, b))))
    return list(cycles)

def cycles_C3(out_set, in_set):
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

def sort_bp2(L1,L2,R1,R2):
    l1, l2 = (L1, L2) if L1 < L2 else (L2, L1)
    r1, r2 = (R1, R2) if R1 < R2 else (R2, R1)
    return l1, l2, r1, r2

def BP2(out_set, in_set):
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