"""
Plotting utilities for DR benchmark results.
Handles both trace formats:
  - OLD (t3-t14): per-iteration aggregated  -> columns: iteration, proxy_max, proxy_min, proxy_mean, best_full
  - NEW (t15+):   per-eval with phase col   -> columns: eval_id, iteration, phase, proxy, best_full
"""
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker


def plot_dr_curve(ax, trace, row, sheet_name):
    # ── detect format ────────────────────────────────────────────────────────
    new_format = "phase" in trace.columns

    if new_format:
        ckpt = trace[trace["phase"] == "checkpoint"].copy()
        dr   = trace[trace["phase"] == "dr"].copy()
        x_iter   = ckpt["iteration"].to_numpy()
        best_arr = ckpt["best_full"].to_numpy()

        if len(dr) > 0:
            # raw scatter — every individual proxy evaluation (faint background)
            ax.scatter(dr["iteration"], dr["proxy"], s=4, alpha=0.15,
                       color="tab:gray", zorder=1, label="_nolegend_")

            # best proxy per iteration = real proxy score of top candidate
            proxy_line = (
                dr.groupby("iteration")["proxy"].max()
                  .reindex(x_iter).to_numpy()
            )
            ax.plot(x_iter, proxy_line, linewidth=1.0, alpha=0.65,
                    color="tab:blue", label="DR proxy (best per batch)")

            valid = dr["proxy"].dropna()
            y_lo  = min(valid.quantile(0.02), best_arr.min())
        else:
            y_lo = best_arr.min()

        y_hi  = best_arr.max()
        n_iters = len(x_iter)
        n_evals = len(dr)
        iters_label = f"iters={n_iters} ({n_evals} evals)"

    else:
        # OLD format: already one row per iteration
        x_iter   = trace["iteration"].to_numpy()
        best_arr = trace["best_full"].ffill().to_numpy()

        if "proxy_max" in trace.columns:
            proxy_col = trace["proxy_max"].to_numpy()
        elif "proxy_mean" in trace.columns:
            proxy_col = trace["proxy_mean"].to_numpy()
        else:
            proxy_col = None

        if proxy_col is not None:
            ax.plot(x_iter, proxy_col, linewidth=1.0, alpha=0.65,
                    color="tab:blue", label="DR fluctuating (current)")
            y_lo = min(np.nanmin(proxy_col), best_arr.min())
        else:
            y_lo = best_arr.min()

        y_hi = best_arr.max()
        n_iters = len(x_iter)
        iters_label = f"iters={n_iters}"

    # ── DR best-so-far (orange, on top) ──────────────────────────────────────
    ax.plot(x_iter, best_arr, linewidth=2.5, color="tab:orange",
            label="DR best-so-far (best_full)")

    # ── y-axis limits ────────────────────────────────────────────────────────
    pad = (y_hi - y_lo) * 0.05 if y_hi > y_lo else 1.0
    ax.set_ylim(y_lo - pad, y_hi + pad)

    # ── metadata & title ─────────────────────────────────────────────────────
    if row is not None:
        instance = row["Instance"]
        N        = int(row["N"])
        T        = int(row["T"])
        seed     = int(row["seed"])
        policy   = row.get("Policy", "")
        exact    = row["Exact_incumbent_raw"]
        gap      = row["Gap_%"]

        if pd.notna(exact):
            ax.axhline(y=float(exact), linestyle="--", linewidth=2.0,
                       color="tab:red", label=f"Exact ({float(exact):.3f})")

        gap_str = f"{gap:.4f}%" if pd.notna(gap) else "N/A"
        title = (
            f"DR vs Exact  |  N={N}, T={T}, seed={seed}  |  policy={policy}\n"
            f"{instance}  -  Gap={gap_str},  DR={best_arr[-1]:.3f},  {iters_label}"
        )
    else:
        title = f"Sheet {sheet_name}  (no benchmark metadata)\n{iters_label}"

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("iteration", fontsize=9)
    ax.set_ylabel("Score", fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.5)
