from evcs.proxy import evaluate_u_numpy_greedy_binary
from evcs.destroy import destroy_multi_u
from evcs.greedy import reconstruct_u_dict_fast


def local_search_u_proxy(
    U_start,
    inst,
    rng,
    P_T,
    demand_TM,
    J_i_list,
    distIJ,
    Q_cap,
    T,
    N,
    cumulative_install,
    max_chargers_per_site,
    ls_moves,
    ls_frac_remove,
    ls_modes,
    Ij_int=None,
    top_k_choice=3,
    collect_visited: bool = False,
):
    """Proxy-guided local search on U-dict.

    Parameters
    ----------
    collect_visited : bool
        When True return ``(visited, proxy_best)`` where *visited* is the list
        of every candidate U generated during the search (used for batch full
        evaluation in the ALNS loop).  When False (default) return
        ``(U_best, proxy_best)`` — unchanged from the original behaviour.
    """
    U_best = dict(U_start)

    proxy_best = float(evaluate_u_numpy_greedy_binary(
        U_best,
        demand_TM=demand_TM,
        J_i_list=J_i_list,
        distIJ=distIJ,
        Q_cap=float(Q_cap),
        T=int(T),
        N=int(N),
        cumulative_install=cumulative_install,
    ))

    visited = [dict(U_start)] if collect_visited else None
    # proxy_scores[0] = proxy of U_start; proxy_scores[k] = proxy after kth LS move
    proxy_scores = [proxy_best] if collect_visited else None

    for _ in range(int(ls_moves)):
        mode = str(rng.choice(list(ls_modes)))

        U_try, _ = destroy_multi_u(
            U_best,
            inst=inst,
            rng=rng,
            P_T=P_T,
            frac_remove=float(ls_frac_remove),
            mode=mode,
            seed=None,
            site_cap=max_chargers_per_site,
            cumulative_install=cumulative_install,
        )

        U_fill, _ = reconstruct_u_dict_fast(
            U_partial=dict(U_try),
            demand_IT=demand_TM,
            P_T=P_T,
            Ij_int=Ij_int,
            U_cap=int(max_chargers_per_site) if max_chargers_per_site is not None else int(max(P_T)),
            Q=float(Q_cap),
            rng=rng,
            cumulative_install=cumulative_install,
            top_k_choice=int(top_k_choice),
        )

        proxy_try = float(evaluate_u_numpy_greedy_binary(
            U_fill,
            demand_TM=demand_TM,
            J_i_list=J_i_list,
            distIJ=distIJ,
            Q_cap=float(Q_cap),
            T=int(T),
            N=int(N),
            cumulative_install=cumulative_install,
        ))

        if collect_visited:
            visited.append(dict(U_fill))
            proxy_scores.append(proxy_try)

        if proxy_try > proxy_best + 1e-9:
            U_best = dict(U_fill)
            proxy_best = float(proxy_try)

    if collect_visited:
        return visited, proxy_scores, float(proxy_best)
    return U_best, float(proxy_best)
