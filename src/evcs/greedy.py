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
    # ── partial-delta cache arguments (optional) ──────────────────────────────
    delta_cache: dict | None = None,   # (j, t) -> float, updated in-place
    refresh_ratio: float = 1.0,        # R: fraction of non-destroyed sites to refresh
    destroyed_js: set | None = None,   # sites whose capacity changed (always refresh)
    Ji: dict | None = None,            # demand node i -> [sites j covering i]
):
    """
    Greedy reconstruction.  When delta_cache is provided the function reuses
    marginal-gain scores from the previous D&R iteration and only recomputes
    scores for (a) the destroyed region and (b) a random R% of other sites.
    After each greedy placement the cache is updated locally via Ji so that
    subsequent picks within the same reconstruction see fresh values.

    delta_cache is modified **in-place**; pass an empty dict {} on the first
    call and the same dict on every subsequent call.
    """
    import numpy as np

    if rng is None:
        rng = np.random.default_rng(0)

    # Copy (do not mutate caller)
    U = dict(U_partial)

    if Ij_int and len(Ij_int) > 0:
        Js = sorted({int(j) for j in Ij_int.keys()})
    else:
        Js = sorted({int(j) for (j, t) in U.keys()})

    T   = len(P_T)
    Q   = float(Q)
    U_cap = int(U_cap)
    proxy_total = 0.0
    use_cache   = delta_cache is not None

    def u_get(j, t): return int(U.get((int(j), int(t)), 0))
    def u_set(j, t, v): U[(int(j), int(t))] = int(v)

    def x_now(j, t):
        j, t = int(j), int(t)
        if cumulative_install:
            return sum(int(U.get((j, tt), 0)) for tt in range(t + 1))
        return int(U.get((j, t), 0))

    # Build Ji (inverse index) once if caching is on and it wasn't supplied
    if use_cache and Ji is None:
        Ji = {}
        for j_key, nodes in Ij_int.items():
            for i in nodes:
                Ji.setdefault(int(i), []).append(int(j_key))

    # ── Build refresh set for this reconstruction call ────────────────────────
    # Sites in refresh_set get their delta recomputed from the uncovered mask.
    # All other sites use the cached value and are updated locally via Ji.
    if use_cache:
        refresh_set = set(int(j) for j in (destroyed_js or []))
        n_rand = max(0, int(refresh_ratio * len(Js)))
        if n_rand >= len(Js):
            refresh_set = set(Js)           # full refresh = baseline behaviour
        elif n_rand > 0:
            rand_js = rng.choice(Js, size=n_rand, replace=False)
            refresh_set.update(int(j) for j in rand_js)
    else:
        refresh_set = None                  # None means "refresh everything"

    for t in range(T):
        # Precompute x_now for all sites (avoids O(T) inner calls in hot loop)
        xcap = {j: x_now(j, t) for j in Js}

        already = sum(u_get(j, t) for j in Js)
        missing = max(0, int(P_T[t]) - int(already))
        if missing <= 0:
            continue

        M = len(demand_IT[t])
        uncovered = [True] * M

        # After step 0 we switch to Ji-local updates; track with a mutable set
        needs_fresh = set(refresh_set) if use_cache else None

        for step in range(missing):
            topk = []

            for j in Js:
                jj = int(j)
                if xcap[jj] >= U_cap:
                    if use_cache:
                        delta_cache[(jj, t)] = 0.0
                    continue

                if (not use_cache) or jj in needs_fresh or (jj, t) not in delta_cache:
                    # ── fresh computation ─────────────────────────────────────
                    s = 0.0
                    for i in Ij_int.get(jj, []):
                        if uncovered[int(i)]:
                            s += float(demand_IT[t, int(i)])
                    if use_cache:
                        delta_cache[(jj, t)] = s
                else:
                    # ── use cached (locally-updated) score ────────────────────
                    s = max(0.0, delta_cache[(jj, t)])

                if s > 1e-12:
                    topk.append((float(s), jj))

            # After the first greedy step, rely on Ji local updates only
            if step == 0 and needs_fresh is not None:
                needs_fresh = set()

            if not topk:
                break

            k = max(1, int(top_k_choice) if top_k_choice is not None else 1)
            topk.sort(key=lambda x: x[0], reverse=True)
            topk = topk[:k]

            if rng.random() < 0.85:
                best_score, best_j = topk[0]
            else:
                best_score, best_j = topk[int(rng.integers(0, len(topk)))]

            if best_score <= 1e-12:
                break

            # Place one charger
            u_set(best_j, t, u_get(best_j, t) + 1)
            xcap[best_j] = x_now(best_j, t)

            # Capacity-aware uncovered update + local cache update via Ji
            used  = 0.0
            neigh = Ij_int.get(best_j, [])
            reach = [int(i) for i in neigh if uncovered[int(i)]]
            reach.sort(key=lambda ii: float(demand_IT[t, ii]), reverse=True)

            for ii in reach:
                di = float(demand_IT[t, ii])
                if used + di <= Q + 1e-9:
                    uncovered[ii] = False
                    used += di
                    # Subtract this demand from every site that could cover ii
                    if use_cache and Ji is not None:
                        for j_aff in Ji.get(ii, []):
                            key = (int(j_aff), t)
                            if key in delta_cache:
                                delta_cache[key] = max(0.0, delta_cache[key] - di)

            proxy_total += used

    return U, float(proxy_total)
