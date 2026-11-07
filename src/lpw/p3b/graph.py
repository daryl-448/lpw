"""Cross-agent sharing structure on a focal view (reports/p3b_prereg.md §P7, §P8 core family).

A view is N focal agents and their eligible incidences (agent, node): non-venue, non-agent counterparties touched by
the agent, with node a global identity (real: raw counterparty id; synthetic: attached node). Degrees count focal
agents of this view only; nothing outside the view is read.
"""
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import wasserstein_distance

LOUVAIN_SEED = 20261104
STRUCT = ["shared_any", "shared_frac", "mean_cp_other_frac", "max_cp_other_frac", "other_agents_frac", "sharing_component_frac"]


def incidence(agent: np.ndarray, node: np.ndarray, N: int) -> tuple[sp.csr_matrix, np.ndarray]:
    """(B (N, n_nodes) 0/1 csr over unique pairs, node ids per column)."""
    agent, node = np.asarray(agent, np.int64), np.asarray(node)
    if len(agent) == 0:
        return sp.csr_matrix((N, 0), dtype=np.int8), np.zeros(0, node.dtype)
    nodes, col = np.unique(node, return_inverse=True)
    B = sp.csr_matrix((np.ones(len(agent), np.int8), (agent, col)), shape=(N, len(nodes)))
    B.data[:] = 1
    B.sum_duplicates()
    B.data[:] = 1
    return B, nodes


def components(B: sp.csr_matrix) -> np.ndarray:
    """Number of focal agents in each agent's connected component of the agent-node bipartite graph."""
    N, C = B.shape
    A = sp.bmat([[None, B], [B.T, None]], format="csr") if C else sp.csr_matrix((N, N))
    _, lab = connected_components(A, directed=False)
    lab = lab[:N]
    return np.bincount(lab)[lab]


def struct_features(B: sp.csr_matrix) -> dict[str, np.ndarray]:
    """The six frozen STRUCT features for every focal agent (zero where the relevant set is empty)."""
    N = B.shape[0]
    den = max(N - 1, 1)
    d = np.asarray(B.sum(0)).ravel().astype(np.float64)
    nC = np.asarray(B.sum(1)).ravel().astype(np.float64)
    Bf = B.astype(np.float64)
    shared = Bf @ (d >= 2).astype(np.float64)
    other = Bf @ (d - 1)
    mx = np.zeros(N)
    if B.nnz:
        rows = np.repeat(np.arange(N), np.diff(B.indptr))
        np.maximum.at(mx, rows, d[B.indices] - 1)
    B2 = Bf[:, d >= 2]
    P = (B2 @ B2.T).tocsr()
    P.setdiag(0)
    P.eliminate_zeros()
    U = np.diff(P.indptr).astype(np.float64)
    K = components(B)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = {"shared_any": (shared > 0).astype(np.float64),
             "shared_frac": np.where(nC > 0, shared / np.maximum(nC, 1), 0.0),
             "mean_cp_other_frac": np.where(nC > 0, other / np.maximum(nC, 1) / den, 0.0),
             "max_cp_other_frac": mx / den,
             "other_agents_frac": U / den,
             "sharing_component_frac": (K - 1) / den}
    for v in f.values():
        assert np.isfinite(v).all()
    return f


def louvain_sizes(B: sp.csr_matrix, nodes: np.ndarray) -> np.ndarray:
    """Agent count of each agent's recovered community: Louvain (seed LOUVAIN_SEED) on the eligible bipartite
    incidence graph, sorted insertion; isolated agents are size-one communities."""
    N = B.shape[0]
    G = nx.Graph()
    G.add_nodes_from(("a", i) for i in range(N))
    coo = B.tocoo()
    order = np.lexsort((nodes[coo.col], coo.row))
    G.add_edges_from((("a", int(coo.row[j])), ("c", int(nodes[coo.col[j]]))) for j in order)
    comms = sorted(nx.community.louvain_communities(G, seed=LOUVAIN_SEED), key=lambda cs: min(cs))
    size = np.zeros(N, np.int64)
    for cs in comms:
        ag = [n[1] for n in cs if n[0] == "a"]
        size[ag] = len(ag)
    assert (size >= 1).all()
    return size


CORE = ["shared_frac/normal", "shared_frac/scam", "other_agents_frac/normal", "other_agents_frac/scam",
        "sharing_component_frac/normal", "sharing_component_frac/scam", "log1p_node_degree", "log1p_community_agents"]


def core_measurements(B: sp.csr_matrix, nodes: np.ndarray) -> dict[str, np.ndarray]:
    """Per-agent and per-node quantities behind the eight core items (roles are applied by the caller)."""
    f = struct_features(B)
    d = np.asarray(B.sum(0)).ravel()
    return {"shared_frac": f["shared_frac"], "other_agents_frac": f["other_agents_frac"],
            "sharing_component_frac": f["sharing_component_frac"], "log1p_node_degree": np.log1p(d.astype(np.float64)),
            "log1p_community_agents": np.log1p(louvain_sizes(B, nodes).astype(np.float64))}


def w1(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))
