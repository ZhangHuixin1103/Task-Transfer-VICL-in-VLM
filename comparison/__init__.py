"""Reproducible comparison benchmarks for T2T-VICL and VICL baselines."""

__all__ = ["benchmark_callable", "count_parameters", "profile_flops"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from . import metrics

    return getattr(metrics, name)
