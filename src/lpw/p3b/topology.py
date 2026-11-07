"""SPEC §7.3 descriptive topology (reports/p3b_prereg.md §P8): daily undirected transaction graphs and Clauset fits.

A transaction table has one row per unique transaction: (u, v, day, eth). u / v are typed node keys (node_key):
real nodes are raw addresses, synthetic nodes are (type, local index) with the type in the top bits, so agents,
venues and periphery nodes never collide. Self-loops are dropped from simple-graph topology.
"""
import warnings

import networkx as nx
import numpy as np

REAL, S_AGENT, S_VENUE, S_PERIPHERY = 0, 1, 2, 3
_TYPE_SHIFT = 60


def node_key(kind: int, local) -> np.ndarray:
    """Typed synthetic node key: kind in {S_AGENT, S_VENUE, S_PERIPHERY}, local < 2^60. Keys are view-local: an
    independent world or part is a different view and its keys are never compared with this one's."""
    local = np.asarray(local, np.int64)
    assert kind in (S_AGENT, S_VENUE, S_PERIPHERY) and (local >= 0).all() and (local < (1 << _TYPE_SHIFT)).all()
    return (np.int64(kind) << _TYPE_SHIFT) | local


def key_kind(key) -> np.ndarray:
    return np.asarray(key, np.int64) >> _TYPE_SHIFT


def tx_table(tx, u, v, ts, eth, venue) -> dict:
    """One row per transaction identity `tx` from incident recordings. Every recording of one transaction must agree
    exactly in directed endpoints, timestamp, value and venue flag (else ValueError: a schema failure). Distinct
    transaction identities stay distinct even when everything else coincides."""
    tx = np.asarray(tx)
    cols = {"u": np.asarray(u), "v": np.asarray(v), "ts": np.asarray(ts), "eth": np.asarray(eth), "venue": np.asarray(venue)}
    _, first, inv, cnt = np.unique(tx, return_index=True, return_inverse=True, return_counts=True)
    for k, c in cols.items():
        if not (c == c[first][inv]).all():
            raise ValueError(f"duplicate recordings of one transaction disagree on {k}")
    return {**{k: c[first] for k, c in cols.items()}, "n_rows": int(len(tx)), "n_tx": int(len(first)), "n_dup_rows": int(len(tx) - len(first))}


DAILY = ["gcc_ratio", "n_components", "max_kcore", "max_core_frac", "assortativity", "transitivity"]


def daily_metrics(u: np.ndarray, v: np.ndarray) -> dict[str, float]:
    """Metrics of one simple undirected graph (NaN where undefined)."""
    m = u != v
    G = nx.Graph()
    G.add_edges_from(zip(u[m].tolist(), v[m].tolist()))
    n = G.number_of_nodes()
    nan = float("nan")
    if n == 0:
        return {k: nan for k in DAILY}
    comps = list(nx.connected_components(G))
    core = nx.core_number(G)
    kmax = max(core.values())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = nx.degree_assortativity_coefficient(G)
        except (ZeroDivisionError, ValueError):
            r = nan
    r = float(r) if r is not None and np.isfinite(r) else nan
    return {"gcc_ratio": max(len(c) for c in comps) / n, "n_components": float(len(comps)), "max_kcore": float(kmax),
            "max_core_frac": sum(1 for x in core.values() if x == kmax) / n, "assortativity": r,
            "transitivity": float(nx.transitivity(G))}


GRID = np.linspace(0, 0.99, 100)  # continuous xmin candidates: empirical quantiles of the sample being fitted
MIN_VALID_REFITS = 90                # of 100 attempted bootstrap refits; fewer -> gof_p undefined
ZETA_TABLE = 100_000                 # exact discrete sampler: survival table length before integer bisection


def ks_continuous(tail: np.ndarray, xmin: float, alpha: float) -> float:
    """Two-sided KS supremum between the empirical CDF of `tail` (sorted, ties allowed) and the continuous power law,
    using both the right limit and the left limit of every empirical jump."""
    vals, cnt = np.unique(tail, return_counts=True)
    n = len(tail)
    right = np.cumsum(cnt) / n
    left = right - cnt / n
    F = 1 - (vals / xmin) ** (1 - alpha)
    return float(max(np.abs(right - F).max(), np.abs(F - left).max()))


def _cont_fit(x: np.ndarray, min_tail: int):
    """Continuous power law: xmin = the KS-minimal candidate among the GRID quantiles with >= min_tail tail points,
    alpha = closed-form MLE. Returns (D, xmin, alpha, n_tail) or None."""
    xs = np.sort(x)
    best = None
    for xm in np.unique(np.quantile(xs, GRID)):
        tail = xs[np.searchsorted(xs, xm, side="left"):]
        if len(tail) < min_tail or xm <= 0:
            continue
        s = np.log(tail / xm).sum()
        if s <= 0:
            continue
        a = 1 + len(tail) / s
        D = ks_continuous(tail, xm, a)
        if best is None or D < best[0]:
            best = (D, float(xm), float(a), int(len(tail)))
    return best


def zeta_sample(alpha: float, xmin: int, u: np.ndarray) -> np.ndarray:
    """Exact inverse-survival draws from the discrete power law P(X = x) = x^-alpha / zeta(alpha, xmin), x >= xmin
    (the law powerlaw 2.0 fits for discrete data with xmin < 10). u in (0, 1]: X = max{x : P(X >= x) >= u}. A survival
    table covers ZETA_TABLE values; rarer draws are resolved by integer bisection on the Hurwitz zeta."""
    from scipy.special import zeta
    xmin = int(xmin)
    Z = zeta(alpha, xmin)
    xs = np.arange(xmin, xmin + ZETA_TABLE, dtype=np.float64)
    S = zeta(alpha, xs) / Z
    u = np.asarray(u, np.float64)
    out = xs[np.clip(np.searchsorted(-S, -u, side="right") - 1, 0, None)]
    for i in np.flatnonzero(u < S[-1]):
        lo, hi = xmin + ZETA_TABLE - 1, xmin + ZETA_TABLE
        while zeta(alpha, hi) / Z >= u[i]:
            lo, hi = hi, hi * 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if zeta(alpha, mid) / Z >= u[i]:
                lo = mid
            else:
                hi = mid
        out[i] = lo
    return out


def clauset(x: np.ndarray, discrete: bool, n_boot: int = 100, seed: int = 0, min_tail: int = 50) -> dict:
    """Clauset-Shalizi-Newman power-law fit with lognormal likelihood ratio (R, p) and a semi-parametric bootstrap
    goodness-of-fit p: n_boot attempted refits, each with its own xmin search; p = share of valid refits with
    D >= observed D, undefined if fewer than MIN_VALID_REFITS are valid.
    - discrete (degrees): powerlaw's full KS xmin search; tail draws from the exact fitted discrete law (zeta_sample).
    - continuous (strength, amount): KS search over the GRID quantile candidates (pre-freeze change, prereg §P8),
      closed-form alpha, two-sided KS; tail draws are Pareto. The LR uses powerlaw at the chosen xmin.
    The body below xmin is resampled from the data."""
    import powerlaw
    x = np.asarray(x, np.float64)
    x = x[x > 0]
    out = {"n": int(len(x))}
    if len(x) < min_tail:
        return {**out, "defined": False}

    def fit(y):
        if discrete:
            f = powerlaw.Fit(y, discrete=True, verbose=False)
            nt = int((y >= f.xmin).sum())
            return (float(f.power_law.D), float(f.xmin), float(f.power_law.alpha), nt) if nt >= min_tail else None
        return _cont_fit(y, min_tail)

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        b = fit(x)
        if b is None:
            return {**out, "defined": False}
        D, xmin, alpha, ntail = b
        f = powerlaw.Fit(x, discrete=discrete, xmin=xmin, verbose=False)
        R, p = f.distribution_compare("power_law", "lognormal", normalized_ratio=True)
        rng = np.random.default_rng(seed)
        below = x[x < xmin]
        hits = valid = 0
        for _ in range(n_boot):
            k = rng.binomial(len(x), ntail / len(x))
            u = 1.0 - rng.random(k)
            tail = zeta_sample(alpha, int(xmin), u) if discrete else xmin * u ** (-1 / (alpha - 1))
            body = rng.choice(below, len(x) - k) if len(below) and len(x) > k else np.zeros(0)
            y = np.concatenate([tail, body])
            g = fit(y[y > 0])
            if g is None:
                continue
            valid += 1
            hits += g[0] >= D
    gof = hits / valid if valid >= MIN_VALID_REFITS else float("nan")
    return {**out, "defined": True, "xmin": xmin, "alpha": alpha, "n_tail": ntail, "D": D, "lr_R": float(R), "lr_p": float(p),
            "gof_p": gof, "gof_valid": valid}
