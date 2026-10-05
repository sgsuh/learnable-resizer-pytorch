"""Summarize finished runs into a Markdown table.

Example:
    python scripts/summarize.py --runs-dir outputs --pattern "s2_*"
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def load_run(run_dir: Path, last_k: int) -> dict | None:
    metrics_path, summary_path = run_dir / "metrics.jsonl", run_dir / "summary.json"
    if not (metrics_path.exists() and summary_path.exists()):
        return None
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    tail = records[-last_k:]
    return {
        "run": run_dir.name,
        "final": records[-1]["val_top1"],
        "last_k": statistics.mean(r["val_top1"] for r in tail),
        "best": max(r["val_top1"] for r in records),
        "gflops": summary["gflops"],
        "params_resizer": summary["params_resizer"],
    }


def fmt(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default="outputs")
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--last-k", type=int, default=3, help="Average val top-1 over the last k epochs.")
    args = parser.parse_args()

    runs = [r for d in sorted(Path(args.runs_dir).glob(args.pattern)) if d.is_dir() and (r := load_run(d, args.last_k))]

    print("| run | final top-1 | last-3 mean top-1 | best top-1 | GFLOPs | resizer params |")
    print("|---|---|---|---|---|---|")
    for r in runs:
        print(
            f"| {r['run']} | {r['final']:.2f} | {r['last_k']:.2f} | {r['best']:.2f} "
            f"| {r['gflops']:.2f} | {r['params_resizer']:,} |"
        )

    groups = defaultdict(list)
    for r in runs:
        groups[re.sub(r"_seed\d+$", "", r["run"])].append(r)
    if any(len(g) > 1 for g in groups.values()):
        print(f"\nAggregated over seeds (mean ± std):\n")
        print("| config | seeds | final top-1 | last-3 mean top-1 | best top-1 |")
        print("|---|---|---|---|---|")
        for name, group in groups.items():
            print(
                f"| {name} | {len(group)} | {fmt([r['final'] for r in group])} "
                f"| {fmt([r['last_k'] for r in group])} | {fmt([r['best'] for r in group])} |"
            )


if __name__ == "__main__":
    main()
