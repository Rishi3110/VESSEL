"""Command line: ``python -m vessel run -c configs/default.yaml``."""
from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vessel", description="Dry-bulk freight rate forecasting")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="full pipeline: tune, evaluate, explain, backtest, live signal, report")
    r.add_argument("-c", "--config", default="configs/default.yaml")
    r.add_argument("--trials", type=int, help="override model.n_trials")
    r.add_argument("--horizon", type=int, help="override horizon (periods)")
    r.add_argument("--out", help="override output_dir")

    g = sub.add_parser("generate-data", help="write the synthetic market dataset")
    g.add_argument("--out", default="data/synthetic")
    g.add_argument("--seed", type=int, default=7)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "generate-data":
        from vessel.synthetic import generate

        print(f"wrote {generate(args.out, seed=args.seed)}")
        return 0

    from vessel.config import load_config
    from vessel.pipeline import run

    overrides: dict = {}
    if args.trials is not None:
        overrides.setdefault("model", {})["n_trials"] = args.trials
    if args.horizon is not None:
        overrides["horizon"] = args.horizon
    if args.out:
        overrides["output_dir"] = args.out
    cfg = load_config(args.config, overrides)
    summary = run(cfg)
    print("\n".join(["", "== Verdict ==", *summary["conclusions"], "", "== Live signal ==",
                     json.dumps({k: v for k, v in summary["live"].items() if k != "top_drivers"}, indent=2),
                     f"\nReport: {cfg.output_dir}/report.html"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
