"""
Grid search over GLP / ensemble hyperparameters with train/test validation.

Sweeps a Cartesian product of ``n_epochs``, ``alpha``, and ``test_size``
values, runs train/test split validation for each cell, and collects
``macro_f1``, ``accuracy``, and the noise-aware error breakdown
(``noise_error_share``, ``noise_errors``, ``hard_errors``, ``total_errors``)
into a polars DataFrame for side-by-side comparison.
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Callable, Dict, List, Optional, Union

import networkit as nk
import polars as pl

from guidedLP.common.exceptions import ConfigurationError, ValidationError
from guidedLP.common.id_mapper import IDMapper
from guidedLP.common.logging_config import LoggingTimer, get_logger
from guidedLP.common.seed_input import SeedInput

from .propagation import ensemble_label_propagation, guided_label_propagation
from .validation import train_test_split_validation

logger = get_logger(__name__)


def _bind_random_seed(propagator: "Callable", seed: int) -> "Callable":
    """Return a propagator wrapper that injects ``random_seed=seed``.

    Used so the GLP single-run path can be seeded per-cell without colliding
    with :func:`train_test_split_validation`'s own ``random_seed`` parameter
    (which controls the train/test split). The wrapper preserves
    ``__name__`` so error messages in the validation layer stay informative.
    """

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("random_seed", seed)
        return propagator(*args, **kwargs)

    wrapped.__name__ = getattr(propagator, "__name__", "propagator")
    return wrapped


class GLPGridSearch:
    """Grid search over GLP hyperparameters with train/test validation.

    Sweeps ``n_epochs × alpha × test_size`` (optionally repeated with
    different random seeds), runs train/test split validation on each cell,
    and stores per-cell ``macro_f1``, ``accuracy``, and noise-aware error
    metrics in a polars DataFrame.

    The default propagator is :func:`ensemble_label_propagation` with
    ``enable_noise_category=True`` so the ``"noise"`` category is available;
    set ``enable_noise_category=False`` to fall back to a single
    :func:`guided_label_propagation` run per cell (in which case
    ``noise_error_share`` will be 0.0 everywhere, since nothing can predict
    ``"noise"``).

    Parameters
    ----------
    graph : nk.Graph
        NetworkIt graph for propagation.
    id_mapper : IDMapper
        Original-to-internal node ID mapping.
    seed_labels : SeedInput
        Full labelled seed set. Split into train/test inside each cell. Accepts
        dict, polars.DataFrame, or pandas.DataFrame.
    labels : List[str]
        Real-class labels. Do NOT include ``"noise"`` — it is auto-added when
        ``enable_noise_category=True``.
    n_epochs_grid : Optional[List[int]], default [10]
        Ensemble epochs to try. ``[1]`` runs a single propagation per cell.
    alpha_grid : Optional[List[float]], default [0.85]
        Propagation alpha values to try; each must be in ``[0, 1]``.
    test_size_grid : Optional[List[float]], default [0.2]
        Test-fraction values to try; each must be in ``(0, 1)``.
    n_repeats : int, default 1
        Repeat every (n_epochs, alpha, test_size) combination with different
        random seeds. Repeat ``r`` uses seed ``random_seed + r``. The results
        DataFrame has one row per (combination, repeat); :meth:`summary`
        averages across repeats.
    random_seed : Optional[int], default 42
        Base seed for the train/test split and the ensemble's noise sampling.
        Pass ``None`` for non-reproducible runs.
    stratify : bool, default True
        Whether to stratify the train/test split by label.
    directional : bool, default False
        Forwarded to the propagator. When ``True`` the propagator returns
        an ``(out_df, in_df)`` tuple and ``directional_pass`` must be set.
    directional_pass : Optional[str], default None
        Which directional pass to evaluate (``"out"`` or ``"in"``). Required
        when ``directional=True``.
    enable_noise_category : bool, default True
        Auto-add a ``"noise"`` category. Required for ``noise_error_share`` to
        be a meaningful metric.
    noise_ratio : float, default 0.1
        Fraction of non-seed nodes assigned as noise seeds.
    n_workers : Optional[int], default None
        Workers for ``ensemble_label_propagation`` within a cell. ``None``
        uses the ensemble's default (cap at ``min(n_epochs, cpu_count)``).
    seed_node_col, seed_label_col : str
        Column names when ``seed_labels`` is a DataFrame.
    **glp_kwargs
        Forwarded to the propagator on every cell (e.g. ``max_iterations``,
        ``convergence_threshold``, ``weight_transform``,
        ``confidence_threshold``).

    Attributes
    ----------
    results : Optional[pl.DataFrame]
        Populated by :meth:`run`. Columns:
        ``n_epochs, alpha, test_size, repeat, random_seed,
        accuracy, macro_f1, macro_precision, macro_recall,
        total_errors, noise_errors, hard_errors, noise_error_share,
        train_count, test_count, runtime_seconds, cell_index``.

    Examples
    --------
    >>> grid = GLPGridSearch(
    ...     graph, mapper, seeds, ["left", "right"],
    ...     n_epochs_grid=[1, 10, 20],
    ...     alpha_grid=[0.5, 0.85, 0.95],
    ...     test_size_grid=[0.2, 0.3],
    ...     n_repeats=3,
    ...     random_seed=42,
    ... )
    >>> df = grid.run()
    >>> print(grid.summary())
    >>> best = grid.best(metric="macro_f1")
    """

    DEFAULT_N_EPOCHS: List[int] = [10]
    DEFAULT_ALPHA: List[float] = [0.85]
    DEFAULT_TEST_SIZE: List[float] = [0.2]

    def __init__(
        self,
        graph: nk.Graph,
        id_mapper: IDMapper,
        seed_labels: SeedInput,
        labels: List[str],
        *,
        n_epochs_grid: Optional[List[int]] = None,
        alpha_grid: Optional[List[float]] = None,
        test_size_grid: Optional[List[float]] = None,
        n_repeats: int = 1,
        random_seed: Optional[int] = 42,
        stratify: bool = True,
        directional: bool = False,
        directional_pass: Optional[str] = None,
        enable_noise_category: bool = True,
        noise_ratio: float = 0.1,
        n_workers: Optional[int] = None,
        seed_node_col: str = "node_id",
        seed_label_col: str = "label",
        **glp_kwargs: Any,
    ) -> None:
        self.graph = graph
        self.id_mapper = id_mapper
        self.seed_labels = seed_labels
        self.labels = list(labels)

        self.n_epochs_grid = (
            list(n_epochs_grid) if n_epochs_grid is not None else list(self.DEFAULT_N_EPOCHS)
        )
        self.alpha_grid = (
            list(alpha_grid) if alpha_grid is not None else list(self.DEFAULT_ALPHA)
        )
        self.test_size_grid = (
            list(test_size_grid) if test_size_grid is not None else list(self.DEFAULT_TEST_SIZE)
        )

        if n_repeats < 1:
            raise ConfigurationError(
                f"n_repeats must be >= 1, got {n_repeats}",
                parameter="n_repeats",
                value=n_repeats,
            )

        if directional and directional_pass is None:
            raise ConfigurationError(
                "directional=True requires directional_pass='out' or 'in'",
                parameter="directional_pass",
                value=directional_pass,
            )

        self._validate_grids()

        self.n_repeats = n_repeats
        self.random_seed = random_seed
        self.stratify = stratify
        self.directional = directional
        self.directional_pass = directional_pass
        self.enable_noise_category = enable_noise_category
        self.noise_ratio = noise_ratio
        self.n_workers = n_workers
        self.seed_node_col = seed_node_col
        self.seed_label_col = seed_label_col
        self.glp_kwargs = glp_kwargs

        self.results: Optional[pl.DataFrame] = None

    def _validate_grids(self) -> None:
        if not self.n_epochs_grid:
            raise ConfigurationError(
                "n_epochs_grid cannot be empty", parameter="n_epochs_grid"
            )
        for e in self.n_epochs_grid:
            if not isinstance(e, int) or e < 1:
                raise ConfigurationError(
                    f"n_epochs values must be integers >= 1, got {e!r}",
                    parameter="n_epochs_grid",
                    value=e,
                )
        if not self.alpha_grid:
            raise ConfigurationError(
                "alpha_grid cannot be empty", parameter="alpha_grid"
            )
        for a in self.alpha_grid:
            if not 0.0 <= float(a) <= 1.0:
                raise ConfigurationError(
                    f"alpha values must be in [0, 1], got {a}",
                    parameter="alpha_grid",
                    value=a,
                )
        if not self.test_size_grid:
            raise ConfigurationError(
                "test_size_grid cannot be empty", parameter="test_size_grid"
            )
        for t in self.test_size_grid:
            if not 0.0 < float(t) < 1.0:
                raise ConfigurationError(
                    f"test_size values must be in (0, 1), got {t}",
                    parameter="test_size_grid",
                    value=t,
                )

    def run(self) -> pl.DataFrame:
        """Execute the grid search.

        Returns
        -------
        pl.DataFrame
            One row per (n_epochs, alpha, test_size, repeat). Stored on
            ``self.results``.
        """
        combos = list(
            itertools.product(
                self.test_size_grid,
                self.alpha_grid,
                self.n_epochs_grid,
                range(self.n_repeats),
            )
        )
        n_cells = len(combos)
        logger.info(
            f"GLPGridSearch: running {n_cells} cells "
            f"({len(self.test_size_grid)} test_size × "
            f"{len(self.alpha_grid)} alpha × "
            f"{len(self.n_epochs_grid)} n_epochs × "
            f"{self.n_repeats} repeat{'s' if self.n_repeats != 1 else ''})"
        )

        rows: List[Dict[str, Any]] = []
        with LoggingTimer(f"GLPGridSearch over {n_cells} cells"):
            for cell_idx, (test_size, alpha, n_epochs, repeat) in enumerate(combos):
                row = self._run_cell(test_size, alpha, n_epochs, repeat)
                row["cell_index"] = cell_idx
                rows.append(row)
                logger.info(
                    f"Cell {cell_idx + 1}/{n_cells} "
                    f"[n_epochs={n_epochs}, alpha={alpha}, test_size={test_size}, "
                    f"repeat={repeat}] → "
                    f"macro_f1={row['macro_f1']:.3f}, "
                    f"noise_error_share={row['noise_error_share']:.3f}"
                )

        self.results = pl.DataFrame(rows)
        return self.results

    def _run_cell(
        self, test_size: float, alpha: float, n_epochs: int, repeat: int
    ) -> Dict[str, Any]:
        cell_seed: Optional[int] = (
            self.random_seed + repeat if self.random_seed is not None else None
        )

        # ensemble_label_propagation requires n_epochs >= 2, so n_epochs == 1
        # always uses guided_label_propagation directly (with noise category
        # toggled by self.enable_noise_category — GLP supports it too).
        use_ensemble = n_epochs > 1
        base_propagator: Callable = (
            ensemble_label_propagation if use_ensemble else guided_label_propagation
        )

        propagator_kwargs: Dict[str, Any] = dict(self.glp_kwargs)
        propagator_kwargs["alpha"] = alpha
        propagator_kwargs["directional"] = self.directional
        propagator_kwargs["enable_noise_category"] = self.enable_noise_category
        propagator_kwargs["noise_ratio"] = self.noise_ratio

        propagator: Callable = base_propagator
        if use_ensemble:
            propagator_kwargs["n_epochs"] = n_epochs
            if self.n_workers is not None:
                propagator_kwargs["n_workers"] = self.n_workers
            if cell_seed is not None:
                propagator_kwargs["base_seed"] = cell_seed
        else:
            # GLP's `random_seed` collides with train_test_split_validation's
            # own `random_seed` parameter (which controls the split), so we
            # can't forward it via **glp_kwargs. Bind it to the propagator
            # callable instead.
            if cell_seed is not None:
                propagator = _bind_random_seed(base_propagator, cell_seed)

        t0 = time.perf_counter()
        results = train_test_split_validation(
            graph=self.graph,
            id_mapper=self.id_mapper,
            seed_labels=self.seed_labels,
            labels=self.labels,
            test_size=test_size,
            stratify=self.stratify,
            random_seed=cell_seed,
            seed_node_col=self.seed_node_col,
            seed_label_col=self.seed_label_col,
            propagator=propagator,
            directional_pass=self.directional_pass,
            **propagator_kwargs,
        )
        runtime = time.perf_counter() - t0

        return {
            "n_epochs": n_epochs,
            "alpha": float(alpha),
            "test_size": float(test_size),
            "repeat": repeat,
            "random_seed": cell_seed if cell_seed is not None else -1,
            "accuracy": float(results["accuracy"]),
            "macro_f1": float(results["macro_f1"]),
            "macro_precision": float(results["macro_precision"]),
            "macro_recall": float(results["macro_recall"]),
            "total_errors": int(results["total_errors"]),
            "noise_errors": int(results["noise_errors"]),
            "hard_errors": int(results["hard_errors"]),
            "noise_error_share": float(results["noise_error_share"]),
            "train_count": int(results["train_size"]),
            "test_count": int(results["test_size"]),
            "runtime_seconds": runtime,
        }

    def summary(self, sort_by: str = "macro_f1", descending: bool = True) -> pl.DataFrame:
        """Compact view of the grid, averaged across repeats.

        With ``n_repeats == 1`` this returns a per-cell view with just the
        key columns. With ``n_repeats > 1`` it groups by
        ``(n_epochs, alpha, test_size)`` and reports mean and (sample) std
        across repeats for the headline metrics.

        Parameters
        ----------
        sort_by : str, default "macro_f1"
            Column to sort by. Must exist in the returned DataFrame.
        descending : bool, default True
            Sort order. ``True`` puts the best macro_f1 / highest
            noise_error_share at the top.

        Returns
        -------
        pl.DataFrame
        """
        if self.results is None:
            raise ValidationError("Call .run() before .summary().")

        if self.n_repeats == 1:
            view = self.results.select(
                [
                    "n_epochs",
                    "alpha",
                    "test_size",
                    "macro_f1",
                    "noise_error_share",
                    "accuracy",
                    "noise_errors",
                    "hard_errors",
                    "total_errors",
                    "runtime_seconds",
                ]
            )
            if sort_by not in view.columns:
                raise ValidationError(
                    f"sort_by={sort_by!r} not in summary columns: {view.columns}"
                )
            return view.sort(sort_by, descending=descending)

        agg = self.results.group_by(["n_epochs", "alpha", "test_size"]).agg(
            [
                pl.col("macro_f1").mean().alias("macro_f1_mean"),
                pl.col("macro_f1").std(ddof=1).alias("macro_f1_std"),
                pl.col("noise_error_share").mean().alias("noise_error_share_mean"),
                pl.col("noise_error_share").std(ddof=1).alias("noise_error_share_std"),
                pl.col("accuracy").mean().alias("accuracy_mean"),
                pl.col("accuracy").std(ddof=1).alias("accuracy_std"),
                pl.col("noise_errors").mean().alias("noise_errors_mean"),
                pl.col("hard_errors").mean().alias("hard_errors_mean"),
                pl.col("total_errors").mean().alias("total_errors_mean"),
                pl.col("runtime_seconds").mean().alias("runtime_seconds_mean"),
                pl.len().alias("n_repeats"),
            ]
        )

        # Allow sorting by the user-facing metric name; remap to its mean column.
        sort_col = sort_by
        if sort_col in {"macro_f1", "noise_error_share", "accuracy"}:
            sort_col = f"{sort_col}_mean"
        if sort_col not in agg.columns:
            raise ValidationError(
                f"sort_by={sort_by!r} not in summary columns: {agg.columns}"
            )
        return agg.sort(sort_col, descending=descending)

    def best(
        self, metric: str = "macro_f1", maximize: bool = True
    ) -> Dict[str, Any]:
        """Return the single best row (as a dict) by ``metric``.

        Parameters
        ----------
        metric : str, default "macro_f1"
            Column in ``self.results`` to optimise over.
        maximize : bool, default True
            ``True`` picks the row with the largest value; ``False`` the
            smallest (e.g. for ``total_errors``).
        """
        if self.results is None:
            raise ValidationError("Call .run() before .best().")
        if metric not in self.results.columns:
            raise ValidationError(
                f"metric={metric!r} not in results columns: {self.results.columns}"
            )
        idx = (
            int(self.results[metric].arg_max())
            if maximize
            else int(self.results[metric].arg_min())
        )
        return self.results.row(idx, named=True)

    def pivot(
        self,
        metric: str = "macro_f1",
        index: str = "alpha",
        columns: str = "n_epochs",
        test_size: Optional[float] = None,
        agg: str = "mean",
    ) -> pl.DataFrame:
        """Pivot a chosen metric for quick visual comparison.

        Reshapes the long results into an ``index × columns`` table for the
        chosen metric. Useful for spotting trends along one axis at a time.

        Parameters
        ----------
        metric : str, default "macro_f1"
            Metric column to pivot.
        index : str, default "alpha"
            Column to use as rows (one of ``n_epochs``, ``alpha``, ``test_size``).
        columns : str, default "n_epochs"
            Column to use as the pivoted columns.
        test_size : Optional[float], default None
            If both ``index`` and ``columns`` differ from ``test_size``, the
            grid has multiple test_size slices. Pass one to filter; defaults
            to the first ``test_size`` in the grid.
        agg : str, default "mean"
            Aggregator across repeats and any uncollapsed axes
            (``"mean"`` or ``"first"``).
        """
        if self.results is None:
            raise ValidationError("Call .run() before .pivot().")
        if metric not in self.results.columns:
            raise ValidationError(
                f"metric={metric!r} not in results columns: {self.results.columns}"
            )
        axis_cols = {"n_epochs", "alpha", "test_size"}
        if index not in axis_cols or columns not in axis_cols or index == columns:
            raise ValidationError(
                f"index and columns must be distinct values from {axis_cols}, "
                f"got index={index!r}, columns={columns!r}"
            )

        df = self.results
        third_axis = (axis_cols - {index, columns}).pop()
        if df[third_axis].n_unique() > 1:
            chosen = test_size if third_axis == "test_size" else df[third_axis].min()
            if chosen is None:
                chosen = df[third_axis].min()
            df = df.filter(pl.col(third_axis) == chosen)
            logger.info(
                f"pivot: collapsing {third_axis}={chosen} (the grid has "
                f"multiple {third_axis} values)"
            )

        agg_expr = (
            pl.col(metric).mean() if agg == "mean" else pl.col(metric).first()
        )
        return (
            df.group_by([index, columns])
            .agg(agg_expr.alias(metric))
            .pivot(values=metric, index=index, on=columns)
            .sort(index)
        )


def get_grid_search_summary(grid: GLPGridSearch) -> str:
    """Format a GLPGridSearch's results as a human-readable string.

    Quick alternative to printing the DataFrame for logs / notebooks.
    """
    if grid.results is None:
        return "GLPGridSearch has not been run yet — call .run() first."

    lines = [
        "=== GLPGridSearch Summary ===",
        f"Cells run:          {len(grid.results)}",
        f"n_epochs grid:      {grid.n_epochs_grid}",
        f"alpha grid:         {grid.alpha_grid}",
        f"test_size grid:     {grid.test_size_grid}",
        f"Repeats per cell:   {grid.n_repeats}",
        "",
        "Per-cell results (sorted by macro_f1, best first):",
    ]
    summary_df = grid.summary(sort_by="macro_f1", descending=True)
    lines.append(str(summary_df))

    try:
        best_f1 = grid.best(metric="macro_f1", maximize=True)
        best_noise = grid.best(metric="noise_error_share", maximize=True)
        lines.extend(
            [
                "",
                "Best macro_f1: "
                f"n_epochs={best_f1['n_epochs']}, alpha={best_f1['alpha']}, "
                f"test_size={best_f1['test_size']} → "
                f"macro_f1={best_f1['macro_f1']:.3f}, "
                f"noise_error_share={best_f1['noise_error_share']:.3f}",
                "Best noise_error_share: "
                f"n_epochs={best_noise['n_epochs']}, alpha={best_noise['alpha']}, "
                f"test_size={best_noise['test_size']} → "
                f"noise_error_share={best_noise['noise_error_share']:.3f}, "
                f"macro_f1={best_noise['macro_f1']:.3f}",
            ]
        )
    except ValidationError:
        pass

    return "\n".join(lines)
