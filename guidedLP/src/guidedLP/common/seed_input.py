"""
Seed-input normalization for Guided Label Propagation and related methods.

GLP and its affiliated methods (validation, balancing, statistics) all operate
internally on a single canonical seed representation::

    Dict[Any, str]   # original_node_id -> label

But callers find different shapes more natural depending on where the data
comes from. This module accepts any of the four supported shapes and converts
them to the canonical dict:

1. ``Dict[Any, str]``        — canonical (e.g. ``{"u1": "left", "u2": "right"}``)
2. ``Dict[str, List[Any]]``  — label-keyed (e.g. ``{"left": ["u1", "u3"], "right": ["u2"]}``)
3. ``polars.DataFrame``      — two columns (default ``node_id`` and ``label``)
4. ``pandas.DataFrame``      — two columns (default ``node_id`` and ``label``)

Pandas is an optional dependency: pandas inputs only work if pandas is installed.
The library itself only uses polars internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Union

import polars as pl

from guidedLP.common.exceptions import ValidationError

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

try:
    import pandas as _pd  # type: ignore[import-not-found]

    _HAS_PANDAS = True
except ImportError:  # pragma: no cover
    _pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False


SeedInput = Union[
    Dict[Any, str],
    Dict[str, List[Any]],
    pl.DataFrame,
    "pd.DataFrame",
]
"""Type alias for any supported seed-input shape.

Accepted shapes (see module docstring for examples):

- ``Dict[Any, str]``       — node_id → label
- ``Dict[str, List[Any]]`` — label → list of node_ids
- ``polars.DataFrame``
- ``pandas.DataFrame``     (only when pandas is installed)
"""


def normalize_seed_input(
    seeds: SeedInput,
    node_col: str = "node_id",
    label_col: str = "label",
) -> Dict[Any, str]:
    """
    Convert any supported seed shape into the canonical ``Dict[node, label]``.

    Parameters
    ----------
    seeds : SeedInput
        Seed input in any of the four supported shapes (see module docstring).
    node_col : str, default "node_id"
        Column name for node IDs when ``seeds`` is a DataFrame. Ignored for dict inputs.
    label_col : str, default "label"
        Column name for labels when ``seeds`` is a DataFrame. Ignored for dict inputs.

    Returns
    -------
    Dict[Any, str]
        Canonical mapping of original node ID to its label (label cast to ``str``).
        Returned dict is always a fresh copy — callers may mutate it freely.

    Raises
    ------
    ValidationError
        If the input shape is unsupported, columns are missing, the dict mixes
        formats, or a node appears with conflicting labels.

    Examples
    --------
    >>> normalize_seed_input({"u1": "left", "u2": "right"})
    {'u1': 'left', 'u2': 'right'}

    >>> normalize_seed_input({"left": ["u1"], "right": ["u2"]})
    {'u1': 'left', 'u2': 'right'}

    >>> import polars as pl
    >>> df = pl.DataFrame({"node_id": ["u1", "u2"], "label": ["left", "right"]})
    >>> normalize_seed_input(df)
    {'u1': 'left', 'u2': 'right'}
    """
    if isinstance(seeds, dict):
        return _from_dict(seeds)

    if isinstance(seeds, pl.DataFrame):
        return _from_dataframe(seeds, node_col, label_col, kind="polars")

    if _HAS_PANDAS and isinstance(seeds, _pd.DataFrame):
        return _from_dataframe(seeds, node_col, label_col, kind="pandas")

    raise ValidationError(
        f"Unsupported seed input type: {type(seeds).__name__}. "
        "Expected dict, polars.DataFrame, or pandas.DataFrame."
    )


def _from_dict(seeds: dict) -> Dict[Any, str]:
    """Normalize a dict-shaped seed input.

    Auto-detects whether values are labels (``Dict[node, label]``) or lists of
    nodes (``Dict[label, List[node]]``) by inspecting the first value, then
    validates the rest of the dict matches that shape.
    """
    if not seeds:
        return {}

    first_value = next(iter(seeds.values()))
    is_inverse = isinstance(first_value, (list, tuple, set))

    if is_inverse:
        return _invert_label_keyed_dict(seeds)

    result: Dict[Any, str] = {}
    for node, label in seeds.items():
        if isinstance(label, (list, tuple, set)):
            raise ValidationError(
                f"Inconsistent dict format: first entry suggested {{node: label}} "
                f"but node {node!r} maps to a {type(label).__name__}"
            )
        result[node] = str(label)
    return result


def _invert_label_keyed_dict(seeds: dict) -> Dict[Any, str]:
    """Invert ``{label: [node, ...]}`` into ``{node: label}``."""
    result: Dict[Any, str] = {}
    for label, node_iterable in seeds.items():
        if not isinstance(node_iterable, (list, tuple, set)):
            raise ValidationError(
                f"Inconsistent dict format: expected list of nodes for label "
                f"{label!r}, got {type(node_iterable).__name__}"
            )
        for node in node_iterable:
            if node in result and result[node] != label:
                raise ValidationError(
                    f"Node {node!r} appears under multiple labels "
                    f"({result[node]!r} and {label!r})"
                )
            result[node] = str(label)
    return result


def _from_dataframe(
    df: Any, node_col: str, label_col: str, kind: str
) -> Dict[Any, str]:
    """Normalize a polars or pandas DataFrame.

    ``kind`` is "polars" or "pandas" and selects which accessor to use to pull
    the columns as Python lists.
    """
    columns = list(df.columns)
    missing = [c for c in (node_col, label_col) if c not in columns]
    if missing:
        raise ValidationError(
            f"{kind}.DataFrame is missing required column(s) {missing}. "
            f"Available columns: {columns}. "
            f"Pass seed_node_col/seed_label_col to override the defaults."
        )

    if kind == "polars":
        nodes = df[node_col].to_list()
        labels = df[label_col].to_list()
    else:  # pandas
        nodes = df[node_col].tolist()
        labels = df[label_col].tolist()

    result: Dict[Any, str] = {}
    for node, label in zip(nodes, labels):
        if node is None:
            raise ValidationError(
                f"Null value in '{node_col}' column of seed DataFrame"
            )
        if label is None:
            raise ValidationError(
                f"Null label for node {node!r} in seed DataFrame"
            )
        if node in result and result[node] != str(label):
            raise ValidationError(
                f"Node {node!r} appears multiple times with conflicting labels "
                f"({result[node]!r} and {str(label)!r})"
            )
        result[node] = str(label)
    return result
