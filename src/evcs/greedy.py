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
    demand_IT,                 # array (T, M) of demand values
    P_T,
    Ij_int: dict,              # j -> list of i in range
    U_cap: int,
    Q: float,
    rng=None,
    cumulative_install: bool = True,
    top_k_choice: int = 3,    # kept for API compat; heap always picks top-1
    # ── partial-delta cache (between-iteration warm-start) ────────────────────
    delta_cache: dict | None = None,   # (j, t) -> float, updated in-place
    refresh_ratio: float = 1.0,        # R: fraction of non-destroyed sites to refresh
    destroyed_js: set | None = None,   # sites whose capacity changed (always refresh)
    Ji: dict | None = None,            # kept for API compat (no longer used)
    site_coords: np.ndarray | None = None,  # (N, 2) km coords for dist-weighted R%
):
    """
    Greedy reconstruction with lazy greedy heap and partial-delta cache.

    Within each period the heap persists across all greedy steps — instead of
    rescanning all N sites at every step, we pop-recompute-accept/reinsert only
    what is needed.  By submodularity (marginal gains are non-increasing), a
    site is accepted as soon as its fresh score >= the next heap top.

    Between D&R iterations, delta_cache carries the scores from the previous
    reconstruction as warm-start heap values.  destroyed_js + R% of remaining
    sites are force-refreshed; others start from cache (which serves as a
    plausible ordering hint, recomputed lazily when popped).

    dist-weighted R%: when site_coords is provided, the random R% fraction is
    drawn with weight 1/(dist+eps) to the centroid of the destroyed region —
    sites near the destruction are prioritised for refresh.

    delta_cache is updated **in-place**; pass {} on the first call and reuse.
    """
    import heapq

    if rng is None:
        rng = np.random.default_rng(0)

    U = dict(U_partial)

    if Ij_int and len(Ij_int) > 0:
        Js = sorted({int(j) for j in Ij_int.keys()})
    else:
        Js = sorted({int(j) for (j, t) in U.keys()})

    T         = len(P_T)
    Q         = float(Q)
    U_cap     = int(U_cap)
    proxy_total = 0.0
    use_cache   = delta_cache is not None

    def u_get(j, t): return int(U.get((int(j), int(t)), 0))
    def u_set(j, t, v): U[(int(j), int(t))] = int(v)

    def x_now(j, t):
        j, t = int(j), int(t)
        if cumulative_install:
            return sum(int(U.get((j, tt), 0)) for tt in range(t + 1))
        return int(U.get((j, t), 0))

    # ── Build refresh set ─────────────────────────────────────────────────────
    # refresh_set: sites whose delta is recomputed fresh at heap initialisation.
    # All others start from the cache (stale upper-bound ordering hint).
    if use_cache:
        refresh_set = set(int(j) for j in (destroyed_js or []))
        n_rand = max(0, int(refresh_ratio * len(Js)))
        if n_rand >= len(Js):
            refresh_set = set(Js)
        elif n_rand > 0:
            candidates = [j for j in Js if j not in refresh_set]
            if (site_coords is not None
                    and destroyed_js
                    and len(candidates) > n_rand):
                # Distance-weighted: prefer sites near the destroyed centroid
                c_idx    = np.array(list(destroyed_js), dtype=int)
                centroid = site_coords[c_idx].mean(axis=0)
                cand_arr = np.array(candidates)
                dists    = np.linalg.norm(site_coords[cand_arr] - centroid, axis=1)
                w = 1.0 / (dists + 0.1)
                w /= w.sum()
                chosen = rng.choice(cand_arr, size=n_rand, replace=False, p=w)
                refresh_set.update(int(j) for j in chosen)
            else:
                rand_js = rng.choice(Js, size=n_rand, replace=False)
                refresh_set.update(int(j) for j in rand_js)
    else:
        refresh_set = None

    for t in range(T):
        xcap  = {j: x_now(j, t) for j in Js}
        already = sum(u_get(j, t) for j in Js)
        missing = max(0, int(P_T[t]) - int(already))
        if missing <= 0:
            continue

        M = len(demand_IT[t])
        uncovered = [True] * M

        # ── Build initial heap ────────────────────────────────────────────────
        # The heap must be initialised with UPPER BOUNDS on each site's true
        # current delta so that the lazy-greedy accept condition is correct.
        # At step 0, uncovered = all True, so the exact delta = raw_delta[j]
        # = sum of all demand reachable from j (independent of prior iters).
        #
        # * Refresh-set sites: compute fresh → exact raw_delta → written to cache.
        # * Other sites: delta_cache already holds raw_delta from a previous
        #   step-0 computation (see invariant below) → valid upper bound.
        # * Capacity-saturated sites: skipped without touching the cache so that
        #   raw_delta is preserved for when they become eligible again.
        #
        # Cache invariant: delta_cache[(j,t)] is only ever written during step-0
        # initialisation (where uncovered=all-True), so it always equals raw_delta.
        heap = []
        for j in Js:
            jj = int(j)
            if xcap[jj] >= U_cap:
                continue          # skip; preserve cache entry for future iterations

            if (not use_cache) or (refresh_set is None) or jj in refresh_set \
                    or (jj, t) not in delta_cache:
                s = 0.0
                for i in Ij_int.get(jj, []):
                    s += float(demand_IT[t, int(i)])  # all uncovered at step 0
                if use_cache:
                    delta_cache[(jj, t)] = s  # store raw_delta; never overwritten below
            else:
                s = max(0.0, delta_cache[(jj, t)])  # = raw_delta from previous iter

            if s > 1e-12:
                heapq.heappush(heap, (-s, jj))

        # ── Lazy greedy: heap persists across all steps in this period ────────
        k = max(1, int(top_k_choice) if top_k_choice else 1)

        for _step in range(missing):
            confirmed = []   # (fresh_s, j) verified as top-k candidates
            rejects   = []   # to push back after selection

            while heap and len(confirmed) < k:
                neg_stale, j = heapq.heappop(heap)
                stale_s = -neg_stale

                if stale_s <= 1e-12:
                    break
                if xcap[j] >= U_cap:
                    continue  # preserve cache entry (raw_delta) for future iters

                # Recompute fresh delta against current uncovered mask.
                # Do NOT write back to delta_cache — that would overwrite the
                # stored raw_delta with a post-placement underestimate, breaking
                # the upper-bound invariant for future iterations.
                fresh_s = 0.0
                for i in Ij_int.get(j, []):
                    if uncovered[int(i)]:
                        fresh_s += float(demand_IT[t, int(i)])

                next_top = (-heap[0][0]) if heap else 0.0

                if fresh_s >= next_top - 1e-9:
                    # Confirmed as a top-k candidate by submodularity
                    confirmed.append((fresh_s, j))
                elif fresh_s > 1e-12:
                    rejects.append((-fresh_s, j))
                # else: fully covered by prior placements; discard

            for item in rejects:
                heapq.heappush(heap, item)

            if not confirmed:
                break

            # Elite-biased top-k selection (preserves diversification)
            if rng.random() < 0.85 or len(confirmed) == 1:
                best_score, best_j = confirmed[0]
            else:
                best_score, best_j = confirmed[int(rng.integers(0, len(confirmed)))]

            if best_score <= 1e-12:
                break

            # Reinsert non-selected confirmed candidates (with their fresh scores)
            for sc, jj_c in confirmed:
                if jj_c != best_j and sc > 1e-12:
                    heapq.heappush(heap, (-sc, jj_c))

            # Place one charger at best_j, period t
            u_set(best_j, t, u_get(best_j, t) + 1)
            xcap[best_j] = x_now(best_j, t)

            # Capacity-aware uncovered update
            used  = 0.0
            reach = [int(i) for i in Ij_int.get(best_j, []) if uncovered[int(i)]]
            reach.sort(key=lambda ii: float(demand_IT[t, ii]), reverse=True)
            for ii in reach:
                di = float(demand_IT[t, ii])
                if used + di <= Q + 1e-9:
                    uncovered[ii] = False
                    used += di

            proxy_total += used

    return U, float(proxy_total)
