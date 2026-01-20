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

def build_unique_adj(edge_index: torch.Tensor, num_nodes: int):
    """
    - adjacency lists + sets for O(1) lookup
    - ignores parallel edges
    """
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    out_set = [set() for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        out_set[int(u)].add(int(v))

    out = [list(s) for s in out_set]
    return out, out_set

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

def SG2_witnesses(out_set, in_set, max_witnesses=None, exclude_s_eq_i=True):    
    n = len(out_set)
    W = set()

    for i in range(n):
        preds = in_set[i]
        if len(preds) < 2:
            continue

        preds_list = sorted(preds)

        L = len(preds_list)
        print(L)
        for a in range(L - 1):
            print(a)
        break

def main():
    graph_data = torch.load("./data/train.pt", weights_only=False)
    edge_index, num_nodes = graph_data.edge_index, graph_data.num_nodes

    out, out_set, inn, in_set = build_unique_in_out(edge_index, num_nodes)
    SG2_witnesses(out_set, in_set)

if __name__ == "__main__":
    main()
