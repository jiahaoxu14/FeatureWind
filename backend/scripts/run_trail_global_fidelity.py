#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from featurewind.eval.trail_global_fidelity import GlobalTrailFidelityConfig, MAIN_EXPERIMENT_NAME, run_trail_global_fidelity


def _parse_datasets(raw: str) -> tuple[str, ...]:
    values = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one dataset is required.")
    return tuple(values)


def _parse_feature_indices(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    values = [int(token.strip()) for token in str(raw).split(",") if token.strip()]
    return tuple(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=MAIN_EXPERIMENT_NAME)
    parser.add_argument(
        "--output-dir",
        default="var/output/trail_global_fidelity/latest",
        help="Directory where the compact paper-facing artifacts will be written.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of fixed feature-space steps per trail.",
    )
    parser.add_argument(
        "--delta-frac",
        type=float,
        default=0.1,
        help="Per-step feature increment as a fraction of std(X[:, k], ddof=0).",
    )
    parser.add_argument(
        "--fd-epsilon-frac",
        type=float,
        default=0.02,
        help="Centered finite-difference epsilon as a fraction of std(X[:, k], ddof=0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed for synthetic generation and nonlinear DR fitting.",
    )
    parser.add_argument(
        "--datasets",
        default="curved_global_umap,near_linear_global_umap",
        help="Comma-separated synthetic datasets to run.",
    )
    parser.add_argument(
        "--dr-backend",
        default="umap",
        choices=["umap", "parametric_umap"],
        help="Nonlinear DR backend. Standard UMAP is the default and required path.",
    )
    parser.add_argument(
        "--grid-res",
        type=int,
        default=48,
        help="Grid resolution used to build the interpolated vector field.",
    )
    parser.add_argument(
        "--rollout-substeps",
        type=int,
        default=4,
        help="Internal RK2 substeps per reported feature step for the static trail rollout.",
    )
    parser.add_argument(
        "--feature-indices",
        default=None,
        help="Optional comma-separated feature indices to evaluate instead of each dataset's defaults.",
    )
    args = parser.parse_args()

    result = run_trail_global_fidelity(
        GlobalTrailFidelityConfig(
            output_dir=Path(args.output_dir),
            steps=int(args.steps),
            delta_frac=float(args.delta_frac),
            fd_epsilon_frac=float(args.fd_epsilon_frac),
            seed=int(args.seed),
            datasets=_parse_datasets(args.datasets),
            feature_indices=_parse_feature_indices(args.feature_indices),
            dr_backend=str(args.dr_backend),
            grid_res=int(args.grid_res),
            rollout_substeps=int(args.rollout_substeps),
        )
    )

    print(f"{MAIN_EXPERIMENT_NAME} completed.")
    print(
        f"Attempted / valid / invalid: {result['attempted_case_count']} / "
        f"{result['valid_case_count']} / {result['invalid_case_count']}"
    )
    print(f"Summary CSV: {result['summary_metrics_csv']}")
    print(f"Per-case JSON: {result['per_case_json']}")
    print(f"Endpoint plot: {result['step_endpoint_error_png']}")
    print(f"Path plot: {result['step_path_deviation_png']}")
    print(f"Win-rate plot: {result['step_winrate_png']}")
    print(f"Representative paths: {result['representative_paths_png']}")
    print(f"Paper summary: {result['paper_summary_md']}")


if __name__ == "__main__":
    main()
