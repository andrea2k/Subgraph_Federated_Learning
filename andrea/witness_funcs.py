import torch
import numpy as np

def build_unique_in_out(edge_index, num_nodes):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    out_set = [set() for _ in range(num_nodes)]
    in_set  = [set() for _ in range(num_nodes)]

    for u, v in zip(src, dst):
        u = int(u); v = int(v)
        out_set[u].add(v)
        in_set[v].add(u)

    out = [list(s) for s in out_set]
    inn = [list(s) for s in in_set]
    return out, out_set, inn, in_set

def cycles_C2(out, out_set):
    cycles = []
    n = len(out)
    for a in range(n):
        for b in out[a]:
            if b > a and a in out_set[b]:
                cycles.append((a, b))
    return cycles

def cycles_C3(out, out_set):
    cycles = []
    n = len(out)
    for a in range(n):
        for b in out[a]:
            if b <= a:
                continue
            for c in out[b]:
                if c > a and a in out_set[c]:
                    cycles.append((a, b, c))
    return cycles

def cycles_C4(out, out_set):
    cycles = []
    n = len(out)
    for a in range(n):
        for b in out[a]:
            if b <= a:
                continue
            for c in out[b]:
                if c <= a or c == b:
                    continue
                for d in out[c]:
                    if d == b:
                        continue
                    if d > a and a in out_set[d]:
                        cycles.append((a, b, c, d))
    return cycles

def cycles_C5(out, out_set):
    cycles = []
    n = len(out)
    for a in range(n):
        for b in out[a]:
            if b <= a:
                continue
            for c in out[b]:
                if c <= a or c == b:
                    continue
                for d in out[c]:
                    if d <= a or d == b or d == c:
                        continue
                    for e in out[d]:
                        if e == b or e == c or e == d:
                            continue
                        if e > a and a in out_set[e]:
                            cycles.append((a, b, c, d, e))
    return cycles

def cycles_C6(out, out_set):
    cycles = []
    n = len(out)
    for a in range(n):
        for b in out[a]:
            if b <= a:
                continue
            for c in out[b]:
                if c <= a or c == b:
                    continue
                for d in out[c]:
                    if d <= a or d == b or d == c:
                        continue
                    for e in out[d]:
                        if e <= a or e == b or e == c or e == d:
                            continue
                        for f in out[e]:
                            if f == b or f == c or f == d or f == e:
                                continue
                            if f > a and a in out_set[f]:
                                cycles.append((a, b, c, d, e, f))
    return cycles


def SG2(out_set, in_set):
    """
    Return a list of unique SG2 witnesses (s, j1, j2, i).

    An SG2 witness consists of four nodes where:
    - j1 and j2 are two distinct predecessors of i,
    - s is a common predecessor of both j1 and j2.

    Uniqueness is defined up to permutation of equivalent roles
    (e.g., swapping j1 and j2 or swaapping i and s does not create a new witness).
    """
    n = len(out_set)
    W = set()

    for i in range(n):
        preds = in_set[i]
        if len(preds) < 2:
            continue

        # convert once; sorting gives stable canonical order
        preds_list = sorted(preds)

        L = len(preds_list)
        for a in range(L - 1):
            j1 = preds_list[a]
            if j1 == i:
                continue
            in1 = in_set[j1]
            if not in1:
                continue

            for b in range(a + 1, L):
                j2 = preds_list[b]
                if j2 == i:
                    continue
                in2 = in_set[j2]
                if not in2:
                    continue
                

                # iterate smaller in-set for speed
                if len(in1) <= len(in2):
                    small, big = in1, in2
                else:
                    small, big = in2, in1

                m1, m2 = (j1, j2) if j1 < j2 else (j2, j1)

                for s in small:
                    if s == i or s == j1 or s == j2:
                        continue
                    if s in big:
                        end1, end2 = (s, i) if s < i else (i, s)
                        W.add((int(end1), int(m1), int(m2), int(end2)))
    return list(W)

def BP2(out_set, in_set):
    """
    Return a list of unique BP2 witnesses (l1, l2, r, i).

    A BP2 witness consists of four nodes where:
    - l1 and l2 are two distinct predecessors of i,
    - both l1 and l2 have an outgoing edge to the same node r (r != i).

    Uniqueness ignores the ordering of l1 and l2 + r and i
    (swapping them does not create a new witness).
    """
    n = len(in_set)
    W = set()

    for i in range(n):
        lefts = in_set[i]
        if len(lefts) < 2:
            continue

        lefts_list = sorted(lefts)
        L = len(lefts_list)

        for a in range(L - 1):
            l1 = lefts_list[a]
            if l1 == i:
                continue
            out1 = out_set[l1]
            if not out1:
                continue

            for b in range(a + 1, L):
                l2 = lefts_list[b]
                if l2 == i:
                    continue
                out2 = out_set[l2]
                if not out2:
                    continue
                if len(out1) <= len(out2):
                    small, big = out1, out2
                else:
                    small, big = out2, out1

                x, y = (l1, l2) if l1 < l2 else (l2, l1)

                for r in small:
                    if r == i or r == l1 or r == l2:
                        continue
                    if r in big:
                        r1, r2 = (r, i) if r < i else (i, r)
                        W.add((int(x),int(y),int(r1),int(r2)))
    return list(W)
