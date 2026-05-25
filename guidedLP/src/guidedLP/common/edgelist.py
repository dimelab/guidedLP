"""
EdgeList container — peer of NetworkIt Graph for memory-efficient edge storage.

An EdgeList wraps a Polars DataFrame whose ``src`` and ``tgt`` columns are
``UInt32`` (or ``UInt64``) codes — the same integer values NetworkIt uses
internally as node IDs. Like ``nk.Graph``, an EdgeList travels paired with
an :class:`~guidedLP.common.id_mapper.IDMapper` that handles the
original-ID ↔ code translation at the I/O boundary.

Storing edges as fixed-width integer codes (rather than the original Utf8
identifiers) is what makes large bipartite projections tractable in memory:
two ``UInt32`` columns hold ~5–10× less data than two ``Utf8`` columns of
typical node-ID strings.
"""

from dataclasses import dataclass
from typing import Any

import polars as pl

from guidedLP.common.exceptions import ValidationError


_SUPPORTED_CODE_DTYPES = (pl.UInt32, pl.UInt64)


def _dtype_capacity(dtype: Any) -> int:
    """Number of distinct codes representable in ``dtype`` (i.e. dtype_max + 1)."""
    if dtype == pl.UInt32:
        return 1 << 32
    if dtype == pl.UInt64:
        return 1 << 64
    raise ValidationError(
        f"Unsupported code_dtype: {dtype}. Use pl.UInt32 or pl.UInt64."
    )


@dataclass(frozen=True, eq=False)
class EdgeList:
    """
    Memory-efficient edge container paired with an :class:`IDMapper`.

    Attributes
    ----------
    df : pl.DataFrame
        Columns ``src``, ``tgt`` (both ``code_dtype``) and optional
        ``weight`` (``Float64``). ``src``/``tgt`` values are NetworkIt-style
        internal IDs in the range ``0..n_nodes-1``.
    directed : bool
        Whether the edges should be interpreted as directed.
    bipartite : bool
        Whether the graph is bipartite. Partition membership is recorded on
        the paired :class:`IDMapper` (see ``source_partition_originals`` /
        ``target_partition_originals``), not on the EdgeList itself.
    n_nodes : int
        Total number of nodes in the mapper. May exceed the number of
        distinct codes appearing in ``df`` if the mapper records orphan
        nodes.
    code_dtype : pl.DataType, default pl.UInt32
        Integer width used for the code columns. ``pl.UInt32`` supports up
        to ~4.29B unique nodes; pass ``pl.UInt64`` for larger graphs.

    Notes
    -----
    Pairs with an :class:`IDMapper` exactly the way ``nk.Graph`` does: the
    EdgeList stores codes (= internal IDs) and the mapper stores the
    bidirectional translation to/from original IDs. Original IDs only
    appear at construction boundaries (``build_edgelist_from_frame``,
    ``graph_to_edges``, etc.) and on public function returns.

    The dataclass is ``frozen=True`` to discourage in-place mutation — to
    derive a filtered or otherwise transformed EdgeList, construct a new
    one with the desired ``df``. ``eq=False`` is set because Polars
    DataFrames don't implement value-equality the way ``dataclass``'s
    generated ``__eq__`` expects.
    """

    df: pl.DataFrame
    directed: bool
    bipartite: bool
    n_nodes: int
    code_dtype: Any = pl.UInt32

    def __post_init__(self) -> None:
        if self.code_dtype not in _SUPPORTED_CODE_DTYPES:
            raise ValidationError(
                f"code_dtype must be pl.UInt32 or pl.UInt64; got {self.code_dtype}"
            )

        missing = {"src", "tgt"} - set(self.df.columns)
        if missing:
            raise ValidationError(
                f"EdgeList df is missing required columns: {sorted(missing)}. "
                f"Got columns: {self.df.columns}"
            )

        for col in ("src", "tgt"):
            actual = self.df[col].dtype
            if actual != self.code_dtype:
                raise ValidationError(
                    f"EdgeList '{col}' column has dtype {actual}, expected "
                    f"code_dtype={self.code_dtype}. Re-encode the column or "
                    f"pass a matching code_dtype."
                )

        if "weight" in self.df.columns and self.df["weight"].dtype != pl.Float64:
            raise ValidationError(
                f"EdgeList 'weight' column must be Float64, got "
                f"{self.df['weight'].dtype}."
            )

        if self.n_nodes < 0:
            raise ValidationError(f"n_nodes must be non-negative, got {self.n_nodes}")

        capacity = _dtype_capacity(self.code_dtype)
        if self.n_nodes > capacity:
            upgrade = "pl.UInt64" if self.code_dtype == pl.UInt32 else None
            hint = f" Pass code_dtype={upgrade}." if upgrade else ""
            raise ValidationError(
                f"EdgeList has {self.n_nodes:,} nodes but code_dtype="
                f"{self.code_dtype} supports max {capacity:,} unique values.{hint}"
            )

    def number_of_edges(self) -> int:
        """Return the number of edges (rows in ``df``)."""
        return self.df.height

    def number_of_nodes(self) -> int:
        """Return the total number of nodes (= paired mapper size)."""
        return self.n_nodes

    def is_weighted(self) -> bool:
        """Whether a ``weight`` column is present."""
        return "weight" in self.df.columns

    def __repr__(self) -> str:
        kind = "bipartite" if self.bipartite else ("directed" if self.directed else "undirected")
        weighted = ", weighted" if self.is_weighted() else ""
        return (
            f"<EdgeList: {self.number_of_edges():,} edges, "
            f"{self.number_of_nodes():,} nodes, "
            f"{kind}, {self.code_dtype}{weighted}>"
        )
