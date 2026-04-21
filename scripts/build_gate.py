"""
build_gate.py — Train a regression gate on proxy_0 -> full_eval_delta.

Steps
-----
1. Load all trajectories.csv files found under results/trajectories/
2. Train LinearRegression and RandomForestRegressor on proxy_0 -> full_eval_delta
3. Evaluate gate precision/recall at a sweep of prediction thresholds
4. Plot results and save to results/gate/

Usage
-----
    python scripts/build_gate.py
    python scripts/build_gate.py --no-plot
    python scripts/build_gate.py --quality-cutoff -30
"""

from pathlib import Path
import argparse
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAJ_ROOT    = PROJECT_ROOT / "results" / "trajectories"
GATE_DIR     = PROJECT_ROOT / "results" / "gate"
GATE_DIR.mkdir(parents=True, exist_ok=True)

# ── args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--no-plot",        action="store_true")
parser.add_argument(
    "--quality-cutoff", type=float, default=0.0,
    help="full_eval_delta threshold that defines a 'good' candidate (default 0 = improves incumbent)"
)
args = parser.parse_args()

if args.no_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Step 1 — Loading trajectory data")
print("=" * 60)

csv_files = sorted(TRAJ_ROOT.rglob("trajectories.csv"))
if not csv_files:
    sys.exit(f"No trajectories.csv found under {TRAJ_ROOT}")

frames = []
for p in csv_files:
    df_tmp = pd.read_csv(p)
    df_tmp["source"] = p.parent.name
    frames.append(df_tmp)
    print(f"  {p.parent.name}: {len(df_tmp)} rows")

df = pd.concat(frames, ignore_index=True)
print(f"\nTotal trajectories: {len(df)}")

# Keep only rows where proxy_0 and full_eval_delta are valid
df = df.dropna(subset=["proxy_0", "full_eval_delta"])
print(f"Rows after dropna:  {len(df)}")

X_raw = df["proxy_0"].to_numpy(dtype=float).reshape(-1, 1)
y     = df["full_eval_delta"].to_numpy(dtype=float)

print(f"\nfull_eval_delta — mean={y.mean():.2f}  std={y.std():.2f}  "
      f"min={y.min():.2f}  max={y.max():.2f}")
print(f"Fraction with delta >= {args.quality_cutoff}: "
      f"{(y >= args.quality_cutoff).mean():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TRAIN AND EVALUATE MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 2 — Train & evaluate regression models")
print("=" * 60)

scaler  = StandardScaler()
X_s     = scaler.fit_transform(X_raw)
cv      = KFold(n_splits=5, shuffle=True, random_state=0)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge(alpha=1)":   Ridge(alpha=1.0),
    "RandomForest":     RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1),
}

results = {}
for name, model in models.items():
    r2_scores  = cross_val_score(model, X_s, y, cv=cv, scoring="r2")
    mse_scores = -cross_val_score(model, X_s, y, cv=cv, scoring="neg_mean_squared_error")
    results[name] = dict(r2_mean=r2_scores.mean(), r2_std=r2_scores.std(),
                         rmse_mean=np.sqrt(mse_scores.mean()))
    print(f"  {name:30s}  CV R²={r2_scores.mean():+.4f} ± {r2_scores.std():.4f}"
          f"  RMSE={np.sqrt(mse_scores.mean()):.3f}")

# Fit the two primary models on full data for gate deployment
lr = LinearRegression().fit(X_s, y)
rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1).fit(X_s, y)

y_pred_lr = lr.predict(X_s)
y_pred_rf = rf.predict(X_s)

# Pearson r (proxy_0, full_eval_delta) as reference
r_raw = float(np.corrcoef(X_raw.ravel(), y)[0, 1])
print(f"\n  Pearson r(proxy_0, full_eval_delta) = {r_raw:+.4f}")

# Save models
joblib.dump({"scaler": scaler, "model": rf, "model_name": "RandomForest"},
            GATE_DIR / "gate_model.pkl")
print(f"\n  Gate model saved -> {GATE_DIR / 'gate_model.pkl'}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  GATE PRECISION / RECALL AT THRESHOLD SWEEP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 3 — Gate evaluation at prediction thresholds")
print("=" * 60)

QUALITY_CUTOFF = args.quality_cutoff          # actual delta: "good" if >= this
good_actual    = (y >= QUALITY_CUTOFF)        # ground truth good/bad
n_total        = len(y)
n_good         = good_actual.sum()

print(f"\n  'Good' defined as full_eval_delta >= {QUALITY_CUTOFF}")
print(f"  Good candidates: {n_good}/{n_total}  ({100*n_good/n_total:.1f}%)")

# Sweep predicted-delta thresholds (gate passes candidate if pred >= threshold)
thresholds = np.linspace(y.min() * 0.5, y.max() * 0.5, 300)

gate_rows = []
for pred_name, y_pred in [("Linear", y_pred_lr), ("RandomForest", y_pred_rf)]:
    for tau in thresholds:
        passed      = y_pred >= tau            # gate lets these through
        skipped     = ~passed

        tp = (passed  &  good_actual).sum()    # good  AND passed  (want high)
        fp = (passed  & ~good_actual).sum()    # bad   AND passed
        fn = (skipped &  good_actual).sum()    # good  AND skipped (want low)
        tn = (skipped & ~good_actual).sum()    # bad   AND skipped

        skip_rate   = skipped.mean()
        recall      = tp / max(n_good, 1)
        precision   = tp / max(tp + fp, 1)
        f1          = 2*precision*recall / max(precision + recall, 1e-9)

        gate_rows.append(dict(
            model=pred_name, threshold=tau,
            skip_rate=skip_rate, recall=recall, precision=precision, f1=f1,
            tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        ))

gate_df = pd.DataFrame(gate_rows)

# Print representative operating points for RandomForest
print(f"\n  {'threshold':>12} {'skip_rate':>10} {'recall':>8} {'precision':>10} {'f1':>6}")
print("  " + "-" * 52)
rf_gate = gate_df[gate_df["model"] == "RandomForest"].copy()
for tau in np.percentile(thresholds, [10, 25, 40, 50, 60, 75, 90]):
    row = rf_gate.iloc[(rf_gate["threshold"] - tau).abs().argmin()]
    print(f"  {row['threshold']:>12.1f} {row['skip_rate']:>10.3f} {row['recall']:>8.3f}"
          f" {row['precision']:>10.3f} {row['f1']:>6.3f}")

# Save gate evaluation table
gate_csv = GATE_DIR / "gate_evaluation.csv"
gate_df.to_csv(gate_csv, index=False)
print(f"\n  Gate evaluation saved -> {gate_csv}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PLOT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 4 — Plotting")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    f"Gate analysis — proxy_0 -> full_eval_delta  "
    f"(quality cutoff={QUALITY_CUTOFF},  n={n_total})",
    fontsize=12
)

# ── Panel [0,0]: scatter proxy_0 vs full_eval_delta ──────────────────────────
ax = axes[0, 0]
ax.scatter(X_raw.ravel(), y, s=6, alpha=0.25, color="steelblue", label="data")
sort_idx = np.argsort(X_raw.ravel())
ax.plot(X_raw.ravel()[sort_idx], y_pred_lr[sort_idx],
        color="firebrick", linewidth=1.5, label=f"LR (r={r_raw:+.3f})")
ax.plot(X_raw.ravel()[sort_idx], y_pred_rf[sort_idx],
        color="darkorange", linewidth=1.5, label="RF")
ax.axhline(QUALITY_CUTOFF, color="black", linewidth=0.8, linestyle="--",
           label=f"quality cutoff={QUALITY_CUTOFF}")
ax.set_xlabel("proxy_0  (after D&R, before LS)")
ax.set_ylabel("full_eval_delta")
ax.set_title("Regression fit")
ax.legend(fontsize=8)

# ── Panel [0,1]: residuals (LR) ───────────────────────────────────────────────
ax = axes[0, 1]
resid = y - y_pred_lr
ax.scatter(y_pred_lr, resid, s=6, alpha=0.25, color="steelblue")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("LR predicted delta")
ax.set_ylabel("residual (actual − predicted)")
r2_lr_full = r2_score(y, y_pred_lr)
ax.set_title(f"LR residuals  (R²={r2_lr_full:.4f})")

# ── Panel [0,2]: residuals (RF) ───────────────────────────────────────────────
ax = axes[0, 2]
resid_rf = y - y_pred_rf
ax.scatter(y_pred_rf, resid_rf, s=6, alpha=0.25, color="darkorange")
ax.axhline(0, color="black", linewidth=0.8)
r2_rf_full = r2_score(y, y_pred_rf)
ax.set_xlabel("RF predicted delta")
ax.set_ylabel("residual (actual − predicted)")
ax.set_title(f"RF residuals  (R²={r2_rf_full:.4f})")

# ── Panel [1,0]: skip-rate vs recall trade-off ───────────────────────────────
ax = axes[1, 0]
for model_name, color in [("Linear", "steelblue"), ("RandomForest", "darkorange")]:
    sub = gate_df[gate_df["model"] == model_name]
    ax.plot(sub["skip_rate"], sub["recall"], color=color, linewidth=1.5, label=model_name)
ax.set_xlabel("skip rate  (fraction of candidates filtered out)")
ax.set_ylabel("recall  (fraction of good candidates that pass)")
ax.set_title("Skip rate vs Recall trade-off")
ax.legend()
ax.invert_xaxis()
ax.set_xlim(1.0, 0.0)

# ── Panel [1,1]: threshold vs skip-rate and recall (RF) ──────────────────────
ax = axes[1, 1]
sub = gate_df[gate_df["model"] == "RandomForest"]
ax.plot(sub["threshold"], sub["skip_rate"], color="firebrick",  label="skip rate")
ax.plot(sub["threshold"], sub["recall"],    color="steelblue",  label="recall")
ax.plot(sub["threshold"], sub["precision"], color="darkorange", label="precision")
ax.plot(sub["threshold"], sub["f1"],        color="purple",     label="F1", linestyle="--")
ax.axvline(QUALITY_CUTOFF, color="black", linewidth=0.8, linestyle=":",
           label=f"cutoff={QUALITY_CUTOFF}")
ax.set_xlabel("prediction threshold τ  (pass if pred_delta ≥ τ)")
ax.set_ylabel("rate")
ax.set_title("RF gate metrics vs threshold")
ax.legend(fontsize=8)

# ── Panel [1,2]: precision vs recall (RF) ────────────────────────────────────
ax = axes[1, 2]
for model_name, color in [("Linear", "steelblue"), ("RandomForest", "darkorange")]:
    sub = gate_df[gate_df["model"] == model_name]
    ax.plot(sub["recall"], sub["precision"], color=color, linewidth=1.5, label=model_name)
baseline_precision = n_good / n_total
ax.axhline(baseline_precision, color="gray", linewidth=0.8, linestyle="--",
           label=f"no-gate baseline={baseline_precision:.3f}")
ax.set_xlabel("recall")
ax.set_ylabel("precision")
ax.set_title("Precision–Recall curve")
ax.legend(fontsize=8)

fig.tight_layout()
fig_path = GATE_DIR / "gate_analysis.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"  Figure saved -> {fig_path}")

if not args.no_plot:
    plt.show()

print("\n===== DONE =====")
