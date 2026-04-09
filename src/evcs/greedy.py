import numpy as np

from evcs.utils import _apply_u_matrix
from evcs.methods import sync_solution_state, reassign_y_greedy_multi


def reconstruct_multi_u_greedy(
    m_template,
    Udict_partial,
    distIJ,
    demand_IT,
    P_T,
    policy: str,
    cumulative_install: bool = True,
    top_k_choice: int = 3,      # ✅ NEW: pick among top-k
    p_elite: float = 0.85,      # ✅ NEW: choose best with prob p_elite
    rng=None,                   # ✅ NEW: pass rng from DR for reproducibility
):
    """
    Strong reconstruction (marginal uncovered demand), with optional elite-biased top-k randomness.

    Steps:
      1) clone template, apply partial u
      2) for each period t: add missing chargers using marginal uncovered demand scoring
         - capacity-aware uncovered update (serve up to Q per placed charger)
         - selection: pick best site with prob p_elite else random among top-k
      3) sync x/z then assign y via reassign_y_greedy_multi
    """
    import numpy as np

    if rng is None:
        rng = np.random.default_rng(0)

    # Clone model so we don't mutate template
    m = m_template.clone()

    # 1) apply partial u
    _apply_u_matrix(m, Udict_partial)

    # Cache Q once
    Q = float(m.Q.value)

    # periods
    T = len(P_T)

    # Build Ij_int from arcs for reach scoring
    Ij_int = {}
    for (i, j) in m.Arcs:
        Ij_int.setdefault(int(j), []).append(int(i))

    # Per-site cap
    U_cap = int(m.U.value) if hasattr(m, "U") else int(max(P_T))

    # Make sure demand_IT is indexed by [t][i]
    # (Assuming you already canonicalized to (T, M))
    def demand_at(t, i):
        return float(demand_IT[t, i])

    def x_now(j, t):
        """Chargers at site j in period t (cumulative or per-period)."""
        j = int(j)
        t = int(t)
        if cumulative_install:
            s = 0
            for tt in m.T:
                if int(tt) <= t:
                    s += int(m.u[j, tt].value or 0)
            return s
        return int(m.u[j, t].value or 0)

    # Clamp params safely
    top_k_choice = int(top_k_choice) if top_k_choice is not None else 1
    top_k_choice = max(1, top_k_choice)
    p_elite = float(p_elite)
    p_elite = min(1.0, max(0.0, p_elite))

    for t in range(T):
        # how many already installed in period t?
        already = 0
        for j in m.J:
            already += int(m.u[int(j), int(t)].value or 0)

        missing = max(0, int(P_T[t]) - int(already))
        if missing <= 0:
            continue

        # uncovered nodes in this period (use indices of demand vector)
        M = len(demand_IT[t])
        uncovered = set(range(M))

        for _ in range(missing):
            cands = []
            # evaluate each candidate site j
            for j in m.J:
                jj = int(j)
                if x_now(jj, t) >= U_cap:
                    continue

                # marginal gain = sum of uncovered demand reachable from j
                s = 0.0
                for i in Ij_int.get(jj, []):
                    ii = int(i)
                    if ii in uncovered:
                        s += demand_at(t, ii)

                if s > 1e-12:
                    cands.append((float(s), jj))

            if not cands:
                break

            # sort by score desc
            cands.sort(key=lambda x: x[0], reverse=True)

            # elite-biased top-k selection
            k = min(top_k_choice, len(cands))
            topk = cands[:k]

            if rng.random() < p_elite:
                best_score, best_j = topk[0]
            else:
                best_score, best_j = topk[int(rng.integers(0, k))]

            if best_score <= 1e-12:
                break

            # install one charger at (best_j, t)
            m.u[best_j, int(t)].value = int(m.u[best_j, int(t)].value or 0) + 1

            # capacity-aware uncovered update: mark nodes served up to Q
            cap = Q
            reach_nodes = [int(i) for i in Ij_int.get(int(best_j), []) if int(i) in uncovered]
            reach_nodes.sort(key=lambda i: demand_at(t, i), reverse=True)

            used = 0.0
            for i in reach_nodes:
                di = demand_at(t, i)
                if used + di <= cap + 1e-9:
                    uncovered.discard(int(i))
                    used += di

    # 3) sync x,z then assign y
    sync_solution_state(m, cumulative_install=cumulative_install)
    m = reassign_y_greedy_multi(
        m, distIJ, Ji=None, method_name=policy, cumulative_install=cumulative_install
    )

    return m


def reconstruct_u_dict_fast(
    U_partial: dict,           # keys (j,t) -> installs
    demand_IT,                 # list length T of demand arrays (len M)
    P_T,
    Ij_int: dict,              # j -> list of i in range
    U_cap: int,
    Q: float,
    rng=None,
    cumulative_install: bool = True,
    top_k_choice: int = 3,

):

    import numpy as np

    if rng is None:
        rng = np.random.default_rng(0)

    # Copy (do not mutate caller)
    U = dict(U_partial)

    # Robustly infer J set:
    # Prefer Ij_int keys (all candidate sites); fallback to keys in U
    if Ij_int and len(Ij_int) > 0:
        Js = sorted({int(j) for j in Ij_int.keys()})
    else:
        Js = sorted({int(j) for (j, t) in U.keys()})

    T = len(P_T)

    def u_get(j, t):
        return int(U.get((int(j), int(t)), 0))

    def u_set(j, t, v):
        U[(int(j), int(t))] = int(v)

    def x_now(j, t):
        """Chargers at site j in period t (cumulative or not)."""
        j = int(j)
        t = int(t)
        if cumulative_install:
            # sum u[j,0..t]
            s = 0
            for tt in range(t + 1):
                s += u_get(j, tt)
            return s
        return u_get(j, t)

    proxy_total = 0.0
    Q = float(Q)
    U_cap = int(U_cap)

    for t in range(T):
        # Period t: how many already installed?
        already = 0
        for j in Js:
            already += u_get(j, t)

        missing = max(0, int(P_T[t]) - int(already))
        if missing <= 0:
            continue

        # uncovered demand mask (bool list is faster than set in tight loops)
        M = len(demand_IT[t])
        uncovered = [True] * M

        for _ in range(missing):
            topk = []  # list of (score, j)

            # choose site by marginal uncovered demand (random among top-k)
            for j in Js:
                if x_now(j, t) >= U_cap:
                    continue

                s = 0.0
                neigh = Ij_int.get(int(j), [])
                for i in neigh:
                    ii = int(i)
                    if uncovered[ii]:
                        s += float(demand_IT[t, ii])

                if s <= 1e-12:
                    continue

                topk.append((float(s), int(j)))

            if not topk:
                break

            # keep only top-k by score (k small, this is fast enough)
            k = int(top_k_choice) if top_k_choice is not None else 1
            k = max(1, k)
            topk.sort(key=lambda x: x[0], reverse=True)
            topk = topk[:k]

            # random tie-break among top-k (weighted by score optional; here: uniform)
            p_elite = 0.85  # move this OUTSIDE the loop later if you want

            if rng.random() < p_elite:
                best_score, best_j = topk[0]
            else:
                best_score, best_j = topk[int(rng.integers(0, len(topk)))]



            # place one charger at (best_j, t)
            u_set(best_j, t, u_get(best_j, t) + 1)

            # capacity-aware uncovered update: greedily mark demand as served up to Q
            used = 0.0
            neigh = Ij_int.get(best_j, [])

            # collect reachable uncovered nodes then sort by demand descending
            reach = []
            for i in neigh:
                ii = int(i)
                if uncovered[ii]:
                    reach.append(ii)

            reach.sort(key=lambda ii: float(demand_IT[t, ii]), reverse=True)

            for ii in reach:
                di = float(demand_IT[t, ii])
                if used + di <= Q + 1e-9:
                    uncovered[ii] = False
                    used += di

            # NOTE: this is *not* the DR proxy metric; only a local signal.
            proxy_total += used

    return U, float(proxy_total)
