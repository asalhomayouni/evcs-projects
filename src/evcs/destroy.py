import numpy as np


def destroy_multi_u(
    Udict,
    inst,
    rng,
    P_T,
    frac_remove: float = 0.20,
    mode: str = "site_swap",
    seed: int | None = None,
    site_cap: int | None = None,
    cumulative_install: bool = True,
    area_radius: float | None = None,
    area_quantile: float = 0.25,
    local_mix=(0.50, 0.25, 0.25),  # k_units, site_all, site_future
):

    # Local RNG if seed provided (do NOT touch global np.random)
    if seed is not None:
        rng = np.random.default_rng(int(seed))

    mode = (mode or "site_swap").lower().strip()

    # Work on a COPY (do NOT mutate input)
    U_new = {k: int(v) for k, v in Udict.items()}

    keys = list(U_new.keys())
    if not keys:
        return U_new, 0

    total = sum(int(v) for v in U_new.values())
    if total <= 0:
        return U_new, 0

    # infer sets
    Js = sorted({int(j) for (j, t) in U_new.keys()})
    Ts = sorted({int(t) for (j, t) in U_new.keys()})
    T_max = max(Ts) if Ts else 0

    # helpers
    def tot_by_j():
        return {j: sum(int(U_new[(j, t)]) for t in Ts) for j in Js}

    def cum_in_site(j, t):
        """cumulative installs at site j up to period t (inclusive), based on U_new."""
        return sum(int(U_new[(j, tt)]) for tt in Ts if int(tt) <= int(t))

    # ---------------------------------------------------------
    # 1) SITE SWAP  (move schedule from one open site to another)
    # ---------------------------------------------------------
    if mode in ("site_swap"):
        totj = tot_by_j()
        open_sites = [j for j, v in totj.items() if v > 0]
        if not open_sites:
            return U_new, 0

        closed_sites = [j for j, v in totj.items() if v == 0]

        j_out = int(rng.choice(open_sites))

        # choose j_in != j_out, prefer closed
        if closed_sites:
            cand = [j for j in closed_sites if j != j_out]
            if not cand:
                cand = [j for j in Js if j != j_out]
        else:
            cand = [j for j in Js if j != j_out]

        if not cand:
            return U_new, 0

        j_in = int(rng.choice(cand))

        # move schedule period-by-period, enforcing cap if requested
        for t in Ts:
            v = int(U_new[(j_out, t)])
            if v <= 0:
                continue

            # compute how much we can add to j_in at this period
            add = v
            if site_cap is not None:
                if cumulative_install:
                    cur = cum_in_site(j_in, t)
                    remaining = max(0, int(site_cap) - cur)
                    add = min(add, remaining)
                else:
                    remaining = max(0, int(site_cap) - int(U_new[(j_in, t)]))
                    add = min(add, remaining)

            if add > 0:
                U_new[(j_in, t)] = int(U_new[(j_in, t)]) + add

            # remove from j_out regardless (destroy)
            U_new[(j_out, t)] = 0

        # site_swap is a move, not a removal
        return U_new, 0

    # ---------------------------------------------------------
    # 2) LOCAL REMOVE (merged: k_units + site_all + site_future)
    # ---------------------------------------------------------
    if mode == "local_remove":
        totj = tot_by_j()
        open_sites = [j for j, v in totj.items() if v > 0]
        if not open_sites:
            return U_new, 0

        j0 = int(rng.choice(open_sites))

        # choose which subtype to apply
        sub = rng.choice(["k_units", "site_all", "site_future"], p=list(local_mix))

        if sub == "site_all":
            removed = 0
            for t in Ts:
                v = int(U_new[(j0, t)])
                if v > 0:
                    removed += v
                    U_new[(j0, t)] = 0
            return U_new, removed

        if sub == "site_future":
            t_start = int(rng.choice(Ts))
            removed = 0
            for t in Ts:
                if int(t) >= t_start:
                    v = int(U_new[(j0, t)])
                    if v > 0:
                        removed += v
                        U_new[(j0, t)] = 0
            return U_new, removed

        # sub == "k_units"
        periods_with = [t for t in Ts if int(U_new[(j0, t)]) > 0]
        if not periods_with:
            return U_new, 0

        t0 = int(rng.choice(periods_with))
        v = int(U_new[(j0, t0)])
        k = max(1, int(round(float(frac_remove) * v)))
        k = min(k, v)

        U_new[(j0, t0)] = v - k
        return U_new, k

    # ---------------------------------------------------------
    # 3) AREA DESTROY (remove installs within radius across all periods)
    # ---------------------------------------------------------
    if mode == "area_destroy":
        coords_J = np.asarray(inst["coords_J"], dtype=float)

        totj = tot_by_j()
        active = [j for j, v in totj.items() if v > 0]
        if not active:
            return U_new, 0

        j_center = int(rng.choice(active))
        center_xy = coords_J[j_center]

        dists = np.sqrt(np.sum((coords_J - center_xy) ** 2, axis=1))

        # pick radius
        if area_radius is None:
            radius = float(np.quantile(dists, float(area_quantile)))
            if radius <= 1e-12:
                radius = float(np.max(dists)) * 0.10
        else:
            radius = float(area_radius)

        J_neigh = [int(j) for j in range(len(coords_J)) if float(dists[j]) <= radius]

        removed = 0
        for j in J_neigh:
            for t in Ts:
                v = int(U_new[(j, t)])
                if v > 0:
                    removed += v
                    U_new[(j, t)] = 0

        return U_new, removed

    # ---------------------------------------------------------
    raise ValueError(f"Unknown destroy mode: {mode}")
