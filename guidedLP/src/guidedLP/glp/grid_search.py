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

from guidedLP.common.exceptions import (
    ComputationError,
    ConfigurationError,
    ConvergenceError,
    ValidationError,
)
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

    Pass ``test_seeds`` to swap the random split for a fixed test set —
    the ``test_size`` axis then collapses to a single cell, and the
    result frame records ``test_size`` as NaN. See the ``test_seeds``
    parameter below.

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
        Full labelled seed set. Split into train/test inside each cell
        (or used as the training pool when ``test_seeds`` is set).
        Accepts dict, polars.DataFrame, or pandas.DataFrame.
    labels : List[str]
        Real-class labels. Do NOT include ``"noise"`` — it is auto-added when
        ``enable_noise_category=True``.
    n_epochs_grid : Optional[List[int]], default [10]
        Ensemble epochs to try. ``[1]`` runs a single propagation per cell.
    alpha_grid : Optional[List[float]], default [0.85]
        Propagation alpha values to try; each must be in ``[0, 1]``.
    test_size_grid : Optional[List[float]], default [0.2]
        Test-fraction values to try; each must be in ``(0, 1)``. Mutually
        exclusive with ``test_seeds`` — passing both raises
        ``ConfigurationError``.
    test_seeds : Optional[SeedInput], default None
        Fixed test set forwarded to :func:`train_test_split_validation`.
        When provided, the train/test split is no longer random — the
        ``test_size`` axis collapses to a single cell and ``stratify``
        is ignored. ``random_seed`` / ``n_repeats`` still affect noise
        sampling (when ``enable_noise_category=True``) and the ensemble's
        per-epoch resampling (when ``n_epochs > 1``), but produce
        identical rows otherwise. The result frame records ``test_size``
        as NaN for these cells.
    distance_prior_grid : Optional[List[bool]], default [False]
        Whether to enable the per-class geodesic distance prior (see
        :func:`guided_label_propagation`'s ``distance_prior``). Pass
        ``[False, True]`` to compare with vs. without in the same grid.
        For backwards compatibility, if this is omitted but
        ``distance_prior=True`` appears in ``**glp_kwargs``, the latter
        becomes a single-cell default.
    distance_prior_exponent_grid : Optional[List[float]], default [1.0]
        Decay exponent ``β`` for the distance prior
        ``D[i, c] = 1/(d+1)^β``. Larger ``β`` makes the prior more
        sensitive to distance, confining real classes closer to their
        seeds and pushing far-from-seed nodes toward "noise". Only has
        effect on cells where ``distance_prior=True``; with
        ``distance_prior=False`` the value is recorded but does not
        change the propagation. Typical sweep: ``[1.0, 2.0, 3.0, 5.0]``.
    noise_ratio_grid : Optional[List[float]], default [``noise_ratio``]
        Noise-seed ratios to try (each value passed as
        :func:`ensemble_label_propagation`'s ``noise_ratio``). Higher
        values raise the baseline noise probability everywhere, which
        tends to convert "hard" misclassifications into "safe" noise
        misclassifications. If omitted, falls back to ``[noise_ratio]``
        so existing single-value usage keeps working.
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
        Multiplier on the user-seed count for the noise sample size
        (``n_noise = max(1, int(noise_ratio * n_user_seeds))``, capped
        at the non-seed pool). Values ``> 1`` are allowed.
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
        ``n_epochs, alpha, test_size, distance_prior,
        distance_prior_exponent, noise_ratio, repeat, random_seed,
        accuracy, macro_f1, macro_precision, macro_recall,
        total_errors, noise_errors, hard_errors, noise_error_share,
        train_count, test_count, runtime_seconds, error, cell_index``.

        Cells whose propagation fails (e.g. ``ConvergenceError``) are
        recorded as rows with NaN metric values and a non-null ``error``
        message — the grid keeps running through the remaining cells.

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
    DEFAULT_DISTANCE_PRIOR: List[bool] = [False]

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
        test_seeds: Optional[SeedInput] = None,
        distance_prior_grid: Optional[List[bool]] = None,
        distance_prior_exponent_grid: Optional[List[float]] = None,
        noise_ratio_grid: Optional[List[float]] = None,
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
        if test_seeds is not None:
            if test_size_grid is not None:
                raise ConfigurationError(
                    "test_size_grid and test_seeds are mutually exclusive: "
                    "with test_seeds the test set is fixed, so sweeping "
                    "test_size is meaningless. Omit test_size_grid.",
                    parameter="test_size_grid",
                )
            # Sentinel: collapses the test_size axis to a single cell.
            # The actual value is unused (the validator gets test_seeds
            # instead of test_size); recorded as NaN in the result frame.
            self.test_size_grid = [float("nan")]
        else:
            self.test_size_grid = (
                list(test_size_grid) if test_size_grid is not None else list(self.DEFAULT_TEST_SIZE)
            )
        self.test_seeds = test_seeds

        # `distance_prior` can also arrive via **glp_kwargs from older calls;
        # the explicit grid wins, but if the grid is omitted we honour the
        # kwarg as a single-cell default so existing code keeps working.
        if distance_prior_grid is None:
            fallback = bool(glp_kwargs.pop("distance_prior", False))
            self.distance_prior_grid = [fallback]
        else:
            glp_kwargs.pop("distance_prior", None)
            self.distance_prior_grid = list(distance_prior_grid)

        # Same kwarg-fallback pattern for the decay exponent. Defaults to
        # [1.0] (matches the legacy 1/(d+1) shape).
        if distance_prior_exponent_grid is None:
            fallback_exp = float(glp_kwargs.pop("distance_prior_exponent", 1.0))
            self.distance_prior_exponent_grid = [fallback_exp]
        else:
            glp_kwargs.pop("distance_prior_exponent", None)
            self.distance_prior_exponent_grid = [
                float(x) for x in distance_prior_exponent_grid
            ]

        # `noise_ratio` is an explicit constructor kwarg (not via **glp_kwargs).
        # When the grid is omitted we fall back to that single value; when
        # given the grid wins and the constructor kwarg becomes irrelevant.
        if noise_ratio_grid is None:
            self.noise_ratio_grid = [float(noise_ratio)]
        else:
            self.noise_ratio_grid = [float(x) for x in noise_ratio_grid]

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
        # Skip the range check when test_seeds is set: the sentinel NaN
        # value collapses the axis and is never passed to the validator.
        if self.test_seeds is None:
            for t in self.test_size_grid:
                if not 0.0 < float(t) < 1.0:
                    raise ConfigurationError(
                        f"test_size values must be in (0, 1), got {t}",
                        parameter="test_size_grid",
                        value=t,
                    )
        if not self.distance_prior_grid:
            raise ConfigurationError(
                "distance_prior_grid cannot be empty",
                parameter="distance_prior_grid",
            )
        for d in self.distance_prior_grid:
            if not isinstance(d, bool):
                raise ConfigurationError(
                    f"distance_prior values must be bool, got {d!r}",
                    parameter="distance_prior_grid",
                    value=d,
                )
        if not self.noise_ratio_grid:
            raise ConfigurationError(
                "noise_ratio_grid cannot be empty",
                parameter="noise_ratio_grid",
            )
        for r in self.noise_ratio_grid:
            if not (isinstance(r, (int, float)) and r >= 0):
                raise ConfigurationError(
                    f"noise_ratio values must be non-negative numbers, got {r!r}",
                    parameter="noise_ratio_grid",
                    value=r,
                )
        if not self.distance_prior_exponent_grid:
            raise ConfigurationError(
                "distance_prior_exponent_grid cannot be empty",
                parameter="distance_prior_exponent_grid",
            )
        for e in self.distance_prior_exponent_grid:
            if not (isinstance(e, (int, float)) and e >= 0):
                raise ConfigurationError(
                    f"distance_prior_exponent values must be non-negative "
                    f"numbers, got {e!r}",
                    parameter="distance_prior_exponent_grid",
                    value=e,
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
                self.distance_prior_grid,
                self.distance_prior_exponent_grid,
                self.noise_ratio_grid,
                range(self.n_repeats),
            )
        )
        n_cells = len(combos)
        logger.info(
            f"GLPGridSearch: running {n_cells} cells "
            f"({len(self.test_size_grid)} test_size × "
            f"{len(self.alpha_grid)} alpha × "
            f"{len(self.n_epochs_grid)} n_epochs × "
            f"{len(self.distance_prior_grid)} distance_prior × "
            f"{len(self.distance_prior_exponent_grid)} distance_prior_exponent × "
            f"{len(self.noise_ratio_grid)} noise_ratio × "
            f"{self.n_repeats} repeat{'s' if self.n_repeats != 1 else ''})"
        )

        rows: List[Dict[str, Any]] = []
        with LoggingTimer(f"GLPGridSearch over {n_cells} cells"):
            for cell_idx, (
                test_size, alpha, n_epochs, distance_prior,
                distance_prior_exponent, noise_ratio, repeat,
            ) in enumerate(combos):
                row = self._run_cell(
                    test_size, alpha, n_epochs, distance_prior,
                    distance_prior_exponent, noise_ratio, repeat,
                )
                row["cell_index"] = cell_idx
                rows.append(row)
                if row.get("error") is None:
                    logger.info(
                        f"Cell {cell_idx + 1}/{n_cells} "
                        f"[n_epochs={n_epochs}, alpha={alpha}, "
                        f"test_size={test_size}, "
                        f"distance_prior={distance_prior}, "
                        f"distance_prior_exponent={distance_prior_exponent}, "
                        f"noise_ratio={noise_ratio}, repeat={repeat}] → "
                        f"macro_f1={row['macro_f1']:.3f}, "
                        f"noise_error_share={row['noise_error_share']:.3f}"
                    )
                else:
                    logger.info(
                        f"Cell {cell_idx + 1}/{n_cells} "
                        f"[n_epochs={n_epochs}, alpha={alpha}, "
                        f"test_size={test_size}, "
                        f"distance_prior={distance_prior}, "
                        f"distance_prior_exponent={distance_prior_exponent}, "
                        f"noise_ratio={noise_ratio}, repeat={repeat}] → "
                        f"FAILED ({row['error']})"
                    )

        self.results = pl.DataFrame(rows)
        # Force `error` to a string column so it concatenates cleanly across
        # runs even when all cells succeeded (would otherwise be Null dtype).
        if "error" in self.results.columns:
            self.results = self.results.with_columns(
                pl.col("error").cast(pl.Utf8)
            )
        return self.results

    def _run_cell(
        self,
        test_size: float,
        alpha: float,
        n_epochs: int,
        distance_prior: bool,
        distance_prior_exponent: float,
        noise_ratio: float,
        repeat: int,
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
        propagator_kwargs["noise_ratio"] = noise_ratio
        propagator_kwargs["distance_prior"] = distance_prior
        propagator_kwargs["distance_prior_exponent"] = distance_prior_exponent

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

        validator_kwargs: Dict[str, Any] = dict(
            graph=self.graph,
            id_mapper=self.id_mapper,
            seed_labels=self.seed_labels,
            labels=self.labels,
            stratify=self.stratify,
            random_seed=cell_seed,
            seed_node_col=self.seed_node_col,
            seed_label_col=self.seed_label_col,
            propagator=propagator,
            directional_pass=self.directional_pass,
            **propagator_kwargs,
        )
        if self.test_seeds is not None:
            # Fixed test set: validator ignores test_size / stratify /
            # random_seed for splitting, but random_seed still propagates
            # through propagator_kwargs to drive noise sampling.
            validator_kwargs["test_seeds"] = self.test_seeds
        else:
            validator_kwargs["test_size"] = test_size

        t0 = time.perf_counter()
        try:
            results = train_test_split_validation(**validator_kwargs)
        except (ConvergenceError, ComputationError) as exc:
            # Record the cell as a failed run rather than crashing the whole
            # grid. Metrics become NaN/None; the `error` column carries the
            # one-line reason so the user can spot which cells failed and
            # why. Other exception types (ValidationError, ConfigurationError)
            # still propagate — those indicate misuse, not numerical issues.
            runtime = time.perf_counter() - t0
            error_msg = str(exc)
            logger.warning(
                f"Cell failed [n_epochs={n_epochs}, alpha={alpha}, "
                f"test_size={test_size}, distance_prior={distance_prior}, "
                f"distance_prior_exponent={distance_prior_exponent}, "
                f"noise_ratio={noise_ratio}, repeat={repeat}]: {error_msg}"
            )
            return {
                "n_epochs": n_epochs,
                "alpha": float(alpha),
                "test_size": float(test_size),
                "distance_prior": bool(distance_prior),
                "distance_prior_exponent": float(distance_prior_exponent),
                "noise_ratio": float(noise_ratio),
                "repeat": repeat,
                "random_seed": cell_seed if cell_seed is not None else -1,
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
                "macro_precision": float("nan"),
                "macro_recall": float("nan"),
                "total_errors": None,
                "noise_errors": None,
                "hard_errors": None,
                "noise_error_share": float("nan"),
                "train_count": None,
                "test_count": None,
                "runtime_seconds": runtime,
                "error": error_msg,
            }

        runtime = time.perf_counter() - t0
        return {
            "n_epochs": n_epochs,
            "alpha": float(alpha),
            "test_size": float(test_size),
            "distance_prior": bool(distance_prior),
            "distance_prior_exponent": float(distance_prior_exponent),
            "noise_ratio": float(noise_ratio),
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
            "error": None,
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

        cols_single = [
            "n_epochs",
            "alpha",
            "test_size",
            "distance_prior",
            "distance_prior_exponent",
            "noise_ratio",
            "macro_f1",
            "noise_error_share",
            "accuracy",
            "noise_errors",
            "hard_errors",
            "total_errors",
            "runtime_seconds",
        ]
        if "error" in self.results.columns:
            cols_single.append("error")

        if self.n_repeats == 1:
            view = self.results.select(cols_single)
            if sort_by not in view.columns:
                raise ValidationError(
                    f"sort_by={sort_by!r} not in summary columns: {view.columns}"
                )
            return view.sort(sort_by, descending=descending, nulls_last=True)

        agg_exprs = [
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
        if "error" in self.results.columns:
            agg_exprs.append(
                pl.col("error").is_not_null().sum().alias("n_failed")
            )

        agg = self.results.group_by(
            [
                "n_epochs", "alpha", "test_size",
                "distance_prior", "distance_prior_exponent", "noise_ratio",
            ]
        ).agg(agg_exprs)

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
        # Skip cells whose propagation failed (NaN/null metrics, non-null error).
        valid = (
            self.results.filter(pl.col("error").is_null())
            if "error" in self.results.columns else self.results
        )
        if len(valid) == 0:
            raise ValidationError(
                "All grid cells failed; nothing to pick from. Inspect "
                "`grid.results` to see the per-cell error messages."
            )
        idx = (
            int(valid[metric].arg_max())
            if maximize
            else int(valid[metric].arg_min())
        )
        return valid.row(idx, named=True)

    def failed_cells(self) -> pl.DataFrame:
        """Return the subset of cells whose propagation failed.

        Empty if every cell succeeded. Useful for diagnosing which
        ``(alpha, n_epochs, ...)`` combinations are too aggressive
        for your graph / convergence_threshold.
        """
        if self.results is None:
            raise ValidationError("Call .run() before .failed_cells().")
        if "error" not in self.results.columns:
            return self.results.head(0)
        return self.results.filter(pl.col("error").is_not_null())

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
        axis_cols = {
            "n_epochs", "alpha", "test_size",
            "distance_prior", "distance_prior_exponent", "noise_ratio",
        }
        if index not in axis_cols or columns not in axis_cols or index == columns:
            raise ValidationError(
                f"index and columns must be distinct values from {axis_cols}, "
                f"got index={index!r}, columns={columns!r}"
            )

        df = self.results
        # Collapse the unused axes (every axis except `index` and `columns`).
        # `test_size` honours the explicit `test_size=` arg; everything else
        # filters to its first observed value to keep the pivot 2D.
        for other in axis_cols - {index, columns}:
            if df[other].n_unique() <= 1:
                continue
            if other == "test_size" and test_size is not None:
                chosen: Any = test_size
            else:
                chosen = df[other].min()
            df = df.filter(pl.col(other) == chosen)
            logger.info(
                f"pivot: collapsing {other}={chosen} (the grid has "
                f"multiple {other} values)"
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

    n_failed = (
        int(grid.results.filter(pl.col("error").is_not_null()).height)
        if "error" in grid.results.columns else 0
    )

    lines = [
        "=== GLPGridSearch Summary ===",
        f"Cells run:             {len(grid.results)}",
        f"Cells failed:          {n_failed}",
        f"n_epochs grid:         {grid.n_epochs_grid}",
        f"alpha grid:            {grid.alpha_grid}",
        f"test_size grid:        {grid.test_size_grid}",
        f"distance_prior grid:   {grid.distance_prior_grid}",
        f"distance_prior_exponent grid: {grid.distance_prior_exponent_grid}",
        f"noise_ratio grid:      {grid.noise_ratio_grid}",
        f"Repeats per cell:      {grid.n_repeats}",
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
                f"test_size={best_f1['test_size']}, "
                f"distance_prior={best_f1['distance_prior']}, "
                f"noise_ratio={best_f1['noise_ratio']} → "
                f"macro_f1={best_f1['macro_f1']:.3f}, "
                f"noise_error_share={best_f1['noise_error_share']:.3f}",
                "Best noise_error_share: "
                f"n_epochs={best_noise['n_epochs']}, alpha={best_noise['alpha']}, "
                f"test_size={best_noise['test_size']}, "
                f"distance_prior={best_noise['distance_prior']}, "
                f"noise_ratio={best_noise['noise_ratio']} → "
                f"noise_error_share={best_noise['noise_error_share']:.3f}, "
                f"macro_f1={best_noise['macro_f1']:.3f}",
            ]
        )
    except ValidationError:
        pass

    return "\n".join(lines)
