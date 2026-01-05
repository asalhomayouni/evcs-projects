import time
import csv
import pathlib
from collections import defaultdict

class RunLogger:
    """
    Simple logger for local search and simulated annealing runs.

    Each run creates a small in-memory log of (iteration, move, score, time, etc.)
    that you can later save to CSV under results/runs/.
    """

    def __init__(self, N, seed, policy, method, restart_id=0, out_dir="results/runs"):
        self.meta = dict(
            N=N,
            seed=seed,
            policy=policy,
            method=method,
            restart_id=restart_id,
            start_time=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.records = []             # list of dicts
        self.counters = defaultdict(int)  # move-type improvement counters
        self.start_clock = time.perf_counter()

        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.out_dir = pathlib.Path(out_dir)
        self.out_path = (
            self.out_dir / f"run_{policy}_N{N}_seed{seed}_{method}_r{restart_id}.csv"
        )

    # -----------------------------------------------------------
    # during-run logging
    # -----------------------------------------------------------
    def log_iter(self, iteration, move_type, obj_value, detail="", temp=None, accepted=True):
        """Record one iteration of LS or SA."""
        elapsed = time.perf_counter() - self.start_clock
        rec = dict(
            iter=iteration,
            move=move_type,
            obj=obj_value,
            temp=temp if temp is not None else "",
            accepted=int(accepted),
            detail=detail,
            elapsed=elapsed,
        )
        self.records.append(rec)
        if accepted and move_type:
            self.counters[move_type] += 1

    # -----------------------------------------------------------
    # finalize run
    # -----------------------------------------------------------
    def finish_run(self, final_obj):
        self.meta["final_obj"] = final_obj
        self.meta["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.meta["runtime_sec"] = time.perf_counter() - self.start_clock

    # -----------------------------------------------------------
    # save to disk
    # -----------------------------------------------------------
    def save_csv(self):
        """Dump all iteration records to CSV."""
        if not self.records:
            print(f"⚠️ No records to save for {self.meta}")
            return

        # Write per-iteration log
        with open(self.out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)
        print(f"✅ Saved iteration log → {self.out_path}")

    def summary_dict(self):
        """Return one row summary for batch table (e.g., to add to DataFrame)."""
        row = {
            **self.meta,
            "runtime_sec": round(self.meta.get("runtime_sec", 0), 3),
            "final_obj": self.meta.get("final_obj", None),
        }
        # Add counters like moves_openclose, moves_merge, ...
        for move, count in self.counters.items():
            row[f"moves_{move}"] = count
        return row
