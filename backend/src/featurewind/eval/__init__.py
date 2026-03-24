"""Evaluation utilities for FeatureWind."""

__all__ = ["run_trail_global_fidelity"]


def run_trail_global_fidelity(*args, **kwargs):
    from .trail_global_fidelity import run_trail_global_fidelity as _run_trail_global_fidelity

    return _run_trail_global_fidelity(*args, **kwargs)
