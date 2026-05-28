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
from typing import TYPE_CHECKING, Any, Tuple

import polars as pl

from guidedLP.common.exceptions import ValidationError

if TYPE_CHECKING:
    from guidedLP.common.id_mapper import IDMapper


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

    **Extra columns are allowed.** Validation enforces ``src``, ``tgt``
    (and ``weight`` if present) but does not forbid additional columns.
    This is what lets temporal pipelines carry a ``timestamp`` column
    alongside the codes — see ``build_edgelist_from_frame``'s
    ``passthrough_cols`` kwarg and :func:`temporal_bipartite_to_unipartite`.

    **Row-order contract.** Standard EdgeList producers/consumers preserve
    input row order so callers can pre-sort once and rely on that order
    surviving across builds/transforms. Specifically:

    - :func:`build_edgelist_from_frame` keeps the input edgelist's row
      order through every group_by / unique / filter step (achieved via
      ``maintain_order=True`` on the aggregating operations).
    - :func:`apply_backbone`'s EdgeList path uses left-joins on the input
      frame and a kept-mask filter, so the surviving rows are a
      subsequence of the input — original order preserved.

    Functions that derive a new EdgeList from these (e.g. projections)
    don't inherit the input's order because the output is a different
    edge set entirely.
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

    def attach(
        self,
        extra_df: pl.DataFrame,
        id_mapper: "IDMapper",
    ) -> Tuple["EdgeList", "IDMapper"]:
        """Append extra edges (in original IDs) to this EdgeList.

        Decodes self's edges back to original IDs, concatenates with
        ``extra_df``, and re-encodes the union via
        :func:`guidedLP.network.construction.build_edgelist_from_frame`
        into a fresh ``(EdgeList, IDMapper)``. Any nodes in ``extra_df``
        that aren't in ``id_mapper`` get codes in the returned mapper;
        existing nodes get re-coded (codes are not guaranteed to be
        stable across the call).

        Caller is responsible for the contents of ``extra_df`` — no
        filtering, deduplication, or directional mirroring is performed.
        If you want bidirectional edges on a directed EdgeList, emit
        both directions in the frame.

        Parameters
        ----------
        extra_df : pl.DataFrame
            Edges in original IDs. Must have columns ``source_id``,
            ``target_id``, and ``weight``. When self is weighted,
            ``weight`` must be a numeric dtype — Float64 is used as-is,
            other numeric dtypes (Int*, UInt*, Float32) are auto-cast
            to Float64.
        id_mapper : IDMapper
            The mapper paired with self. Used only to decode self's
            existing codes back to original IDs; the returned mapper
            is a fresh one covering the union.

        Returns
        -------
        new_edgelist : EdgeList
            Re-encoded EdgeList covering self's edges plus
            ``extra_df``'s edges. ``directed``, ``bipartite``, and
            ``code_dtype`` are inherited from self.
        new_id_mapper : IDMapper
            Fresh mapper for the union. Original-ID coverage equals
            the union of ``id_mapper``'s entries and any new IDs
            referenced by ``extra_df``.

        Raises
        ------
        ValidationError
            If ``extra_df`` is missing required columns; if its
            ``weight`` column has a non-numeric dtype when self is
            weighted (numeric non-Float64 dtypes are auto-cast).

        Notes
        -----
        Time Complexity: O(E_self + E_extra). One decode pass over
        self, one Polars concat, one Utf8 → UInt32 re-encode.

        Memory: peak is the sum of the decoded self frame, the
        ``extra_df``, and the intermediate Utf8 frame inside
        ``build_edgelist_from_frame``. Filter ``extra_df`` before
        the call if you want to skip rows the re-encode would
        otherwise materialize.
        """
        from guidedLP.network.construction import build_edgelist_from_frame

        required = {"source_id", "target_id", "weight"}
        missing = required - set(extra_df.columns)
        if missing:
            raise ValidationError(
                f"extra_df is missing required columns: {sorted(missing)}. "
                f"Got columns: {extra_df.columns}"
            )
        if self.is_weighted():
            wt_dtype = extra_df["weight"].dtype
            if wt_dtype != pl.Float64:
                if wt_dtype.is_numeric():
                    extra_df = extra_df.with_columns(
                        pl.col("weight").cast(pl.Float64)
                    )
                else:
                    raise ValidationError(
                        f"extra_df 'weight' column must be a numeric dtype "
                        f"castable to Float64 to match this weighted "
                        f"EdgeList; got {wt_dtype}."
                    )

        src_codes = self.df["src"].to_list()
        tgt_codes = self.df["tgt"].to_list()
        weights = (
            self.df["weight"].to_list()
            if self.is_weighted()
            else [1.0] * self.number_of_edges()
        )
        self_decoded = pl.DataFrame(
            {
                "source_id": id_mapper.get_original_batch(src_codes),
                "target_id": id_mapper.get_original_batch(tgt_codes),
                "weight": weights,
            }
        )

        # Cast both frames' ID columns to Utf8 so the concat is
        # dtype-safe even when self's originals aren't strings (e.g.
        # int user IDs concatenated with str synthetic anchor IDs).
        # The re-encode below assigns fresh integer codes anyway, so
        # the downstream mapper's originals are consistently Utf8.
        self_decoded = self_decoded.with_columns(
            pl.col("source_id").cast(pl.Utf8),
            pl.col("target_id").cast(pl.Utf8),
        )
        extra_cast = (
            extra_df
            .select(["source_id", "target_id", "weight"])
            .with_columns(
                pl.col("source_id").cast(pl.Utf8),
                pl.col("target_id").cast(pl.Utf8),
            )
        )

        combined = pl.concat([self_decoded, extra_cast])

        return build_edgelist_from_frame(
            combined,
            source_col="source_id",
            target_col="target_id",
            weight_col="weight",
            directed=self.directed,
            bipartite=self.bipartite,
            auto_weight=False,
            remove_duplicates=False,
            code_dtype=self.code_dtype,
            verbose=False,
        )
