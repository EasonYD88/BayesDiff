"""
scripts/scaling/s02_merge_shards.py
──────────────────────────────────
Merge sharded sampling outputs from multi-GPU Slurm array runs.

Input layout:
  <run_dir>/shards/
    shard_0of4/
      all_embeddings.npz
      sampling_summary.json
    shard_1of4/
      all_embeddings.npz
      sampling_summary.json
    ...

Output:
  <run_dir>/all_embeddings.npz
  <run_dir>/sampling_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded sampling outputs")
    parser.add_argument(
        "--shards_dir",
        type=str,
        required=True,
        help="Directory that contains shard_<i>of<n>/ subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for merged all_embeddings.npz and sampling_summary.json "
        "(default: parent of shards_dir)",
    )
    parser.add_argument(
        "--expected_shards",
        type=int,
        default=None,
        help="Expected number of shard directories with all_embeddings.npz.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    shards_dir = Path(args.shards_dir)
    if not shards_dir.exists():
        raise FileNotFoundError(f"shards_dir not found: {shards_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else shards_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = sorted([p for p in shards_dir.iterdir() if p.is_dir()])
    valid_dirs = [d for d in shard_dirs if (d / "all_embeddings.npz").exists()]

    if not valid_dirs:
        raise RuntimeError(f"No shard directories with all_embeddings.npz under: {shards_dir}")

    if args.expected_shards is not None and len(valid_dirs) != args.expected_shards:
        raise RuntimeError(
            f"Expected {args.expected_shards} shard dirs, found {len(valid_dirs)}"
        )

    merged_embeddings = {}
    shard_reports = []
    timing_all = []
    all_errors = []

    for shard_dir in valid_dirs:
        npz_file = shard_dir / "all_embeddings.npz"
        summary_file = shard_dir / "sampling_summary.json"

        data = np.load(npz_file, allow_pickle=True)
        shard_keys = list(data.files)
        for key in shard_keys:
            if key in merged_embeddings:
                raise RuntimeError(
                    f"Duplicate pocket key '{key}' found in {npz_file}; "
                    "check shard assignment."
                )
            merged_embeddings[key] = data[key]

        report = {
            "shard_dir": str(shard_dir),
            "npz": str(npz_file),
            "n_keys": len(shard_keys),
            "keys_preview": shard_keys[:3],
        }

        if summary_file.exists():
            with open(summary_file) as f:
                s = json.load(f)
            timing = s.get("timing", [])
            timing_all.extend(timing)
            report["summary"] = str(summary_file)
            report["n_timing_entries"] = len(timing)

            for item in timing:
                if "error" in item:
                    all_errors.append(
                        {
                            "pdb_code": item.get("pdb_code"),
                            "error": item.get("error"),
                            "summary_file": str(summary_file),
                        }
                    )

        shard_reports.append(report)

    out_npz = output_dir / "all_embeddings.npz"
    np.savez(out_npz, **merged_embeddings)

    out_summary = output_dir / "sampling_summary.json"
    merged_summary = {
        "source_shards_dir": str(shards_dir),
        "n_shard_dirs_total": len(shard_dirs),
        "n_shard_dirs_merged": len(valid_dirs),
        "n_pockets_merged": len(merged_embeddings),
        "n_errors": len(all_errors),
        "timing": timing_all,
        "shards": shard_reports,
        "errors": all_errors,
    }
    with open(out_summary, "w") as f:
        json.dump(merged_summary, f, indent=2)

    print("=== Merge complete ===")
    print(f"Shards dir:         {shards_dir}")
    print(f"Shard dirs merged:  {len(valid_dirs)}")
    print(f"Merged pocket keys: {len(merged_embeddings)}")
    print(f"Merged npz:         {out_npz}")
    print(f"Merged summary:     {out_summary}")


if __name__ == "__main__":
    main()
