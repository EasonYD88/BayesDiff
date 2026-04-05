"""
scripts/scaling/s01_sample_shard.py
──────────────────────────────────
Shard wrapper for scripts/pipeline/s02_sample_molecules.py.

This script does NOT modify the original sampling code. It only:
  1) reads the full pocket list,
  2) selects a deterministic shard by index,
  3) writes a temporary shard pocket list,
  4) invokes scripts/pipeline/s02_sample_molecules.py on that shard.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one shard of sampling")
    parser.add_argument("--pocket_list", type=str, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--shard_index", type=int, required=True)

    parser.add_argument("--pdbbind_dir", type=str, required=True)
    parser.add_argument("--targetdiff_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--num_steps", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(
            f"--shard_index must be in [0, {args.num_shards - 1}], got {args.shard_index}"
        )

    pocket_list_path = Path(args.pocket_list)
    if not pocket_list_path.exists():
        raise FileNotFoundError(f"Pocket list not found: {pocket_list_path}")

    all_codes = [
        line.strip()
        for line in pocket_list_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    shard_codes = [
        code for idx, code in enumerate(all_codes) if idx % args.num_shards == args.shard_index
    ]

    print("=== Shard selection ===")
    print(f"Total pockets:  {len(all_codes)}")
    print(f"Num shards:     {args.num_shards}")
    print(f"Shard index:    {args.shard_index}")
    print(f"Shard pockets:  {len(shard_codes)}")

    if not shard_codes:
        print("No pockets assigned to this shard. Exiting cleanly.")
        return

    with tempfile.TemporaryDirectory(prefix="bayesdiff_shard_") as tmpdir:
        shard_list = Path(tmpdir) / "pockets_shard.txt"
        shard_list.write_text("\n".join(shard_codes) + "\n")

        cmd = [
            sys.executable,
            "scripts/pipeline/s02_sample_molecules.py",
            "--pocket_list",
            str(shard_list),
            "--pdbbind_dir",
            args.pdbbind_dir,
            "--targetdiff_dir",
            args.targetdiff_dir,
            "--num_samples",
            str(args.num_samples),
            "--num_steps",
            str(args.num_steps),
            "--device",
            args.device,
            "--output_dir",
            args.output_dir,
        ]

        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
