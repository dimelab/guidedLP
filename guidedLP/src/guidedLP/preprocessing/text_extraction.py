"""
Text-content extraction from a ``[sender, post, datetime]`` post table.

Each function in this module takes a Polars DataFrame describing senders and
their posts and returns a long-form bipartite edge list mapping each sender to
the content elements they shared. One row is emitted per match — a post that
mentions three URLs produces three output rows. Posts with no matches are
dropped (they would be isolated nodes in the resulting bipartite graph).

The output schema is ``[sender_col, <content>, datetime_col]`` (datetime column
omitted when ``datetime_col=None``). Input row order is preserved; original IDs
in the sender column are passed through untouched so the result drops straight
into :func:`guidedLP.network.construction.build_graph_from_edgelist`.

All extraction is done with Polars' native regex/string kernels (Rust ``regex``
crate, not Python ``re``), so the work is vectorized over the column and runs
in a single pass without materializing intermediate Python lists.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

import polars as pl

from guidedLP.common.exceptions import ValidationError


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# URL pattern: matches http(s):// URLs and bare ``www.`` URLs. The character
# class excludes whitespace and common surrounding punctuation (quotes,
# brackets, angle brackets) so the match doesn't eat the rest of a sentence.
# Trailing sentence-final punctuation is stripped in a post-processing step
# (see ``_TRAILING_PUNCT``) because a sentence like "see https://x.com." should
# match ``https://x.com``, not ``https://x.com.``. The ``(?i)`` flag makes
# the scheme/prefix recognition case-insensitive (RFC 3986 specifies URL
# schemes as case-insensitive, so ``HTTPS://`` and ``WWW.`` should match too).
_URL_PATTERN = r"""(?i)(?:https?://|www\.)[^\s<>"'`\[\]{}()]+"""

# Characters stripped from the right edge of an extracted URL. Covers
# sentence-final punctuation and closing brackets that the URL regex itself
# is permissive enough to swallow when no closing bracket was present in the
# source text.
_TRAILING_PUNCT = ".,;:!?)]}>\"'`"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _validate_input(
    df: pl.DataFrame,
    sender_col: str,
    post_col: str,
    datetime_col: Optional[str],
) -> None:
    """Check that the requested columns exist and ``post_col`` is a string column."""
    if not isinstance(df, pl.DataFrame):
        raise ValidationError(
            f"Expected polars.DataFrame, got {type(df).__name__}",
            expected="polars.DataFrame",
        )

    required = [sender_col, post_col]
    if datetime_col is not None:
        required.append(datetime_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValidationError(
            f"Input DataFrame is missing required column(s) {missing}. "
            f"Available columns: {list(df.columns)}",
            details={"missing": missing, "available": list(df.columns)},
        )

    post_dtype = df.schema[post_col]
    if post_dtype != pl.Utf8:
        raise ValidationError(
            f"Post column '{post_col}' must be a string column, got {post_dtype}",
            field=post_col,
            expected="Utf8/String",
        )


def _carry_cols(sender_col: str, datetime_col: Optional[str]) -> List[str]:
    """Columns to keep alongside the extracted content."""
    cols = [sender_col]
    if datetime_col is not None:
        cols.append(datetime_col)
    return cols


def _extract_long(
    df: pl.DataFrame,
    sender_col: str,
    post_col: str,
    datetime_col: Optional[str],
    pattern: str,
    out_col: str,
) -> pl.DataFrame:
    """
    Apply ``pattern`` to ``post_col`` and explode matches into one row per match.

    Returns a DataFrame with columns ``[sender_col, out_col(, datetime_col)]`` —
    posts with no matches are dropped. This is the common kernel used by all
    three public extractors.
    """
    carry = _carry_cols(sender_col, datetime_col)
    return (
        df.select(
            carry
            + [pl.col(post_col).str.extract_all(pattern).alias(out_col)]
        )
        .explode(out_col)
        .filter(pl.col(out_col).is_not_null())
    )


# ---------------------------------------------------------------------------
# Public extractors
# ---------------------------------------------------------------------------


def extract_urls(
    df: pl.DataFrame,
    sender_col: str = "sender",
    post_col: str = "post",
    datetime_col: Optional[str] = "datetime",
    url_col: str = "url",
) -> pl.DataFrame:
    """
    Extract URLs from posts into a long-form sender→URL edge list.

    Each row in the input may contain zero, one, or many URLs in ``post_col``.
    The function emits one output row per extracted URL, carrying the sender
    (and datetime, when present) alongside it. Posts with no URLs contribute
    no rows. URLs starting with ``http://``, ``https://``, or a bare ``www.``
    are detected. Trailing sentence punctuation (``.,;:!?)]}'"``) is stripped
    from each match so e.g. ``"see https://x.com."`` yields ``https://x.com``.

    Parameters
    ----------
    df : pl.DataFrame
        Input table of posts. Must contain ``sender_col`` and ``post_col``;
        ``datetime_col`` is optional (see below).
    sender_col : str, default "sender"
        Name of the column identifying the post author. Values are passed
        through untouched (original IDs preserved).
    post_col : str, default "post"
        Name of the string column containing the post text. Must be Utf8.
    datetime_col : Optional[str], default "datetime"
        Name of the timestamp column to carry through to the output. Pass
        ``None`` to omit timestamps entirely (output will be a pure 2-column
        sender→URL edge list).
    url_col : str, default "url"
        Name of the URL column in the output.

    Returns
    -------
    pl.DataFrame
        Long-form edge list with columns ``[sender_col, url_col(, datetime_col)]``,
        one row per extracted URL. Input row order is preserved; URLs from the
        same post appear consecutively in the order they occurred in the text.

    Raises
    ------
    ValidationError
        If required columns are missing or ``post_col`` is not a string column.

    Complexity
    ----------
    O(N · L) where N is the number of input rows and L is the average post
    length — a single vectorized pass over ``post_col`` with the Rust regex
    engine, then a flat explode and filter.

    Examples
    --------
    >>> import polars as pl
    >>> from guidedLP.preprocessing import extract_urls
    >>> df = pl.DataFrame({
    ...     "sender": ["alice", "bob", "carol"],
    ...     "post":   ["see https://x.com and http://y.org!",
    ...                "no links here",
    ...                "check www.example.com/path"],
    ...     "datetime": ["2024-01-01", "2024-01-02", "2024-01-03"],
    ... })
    >>> extract_urls(df).shape
    (3, 3)
    """
    _validate_input(df, sender_col, post_col, datetime_col)

    result = _extract_long(
        df,
        sender_col=sender_col,
        post_col=post_col,
        datetime_col=datetime_col,
        pattern=_URL_PATTERN,
        out_col=url_col,
    )

    return result.with_columns(
        pl.col(url_col).str.strip_chars_end(_TRAILING_PUNCT)
    )


def extract_domains(
    df: pl.DataFrame,
    sender_col: str = "sender",
    post_col: str = "post",
    datetime_col: Optional[str] = "datetime",
    domain_col: str = "domain",
    strip_www: bool = True,
) -> pl.DataFrame:
    """
    Extract URLs from posts and reduce each one to its host (domain).

    Runs :func:`extract_urls` and then strips the scheme, path, query, fragment,
    and (by default) the leading ``www.`` from each URL. Domains are lowercased
    for consistency since hostnames are case-insensitive. The output is a long-
    form sender→domain edge list ready for bipartite graph construction.

    This intentionally does not depend on the public suffix list (no
    ``tldextract``), so ``sub.example.co.uk`` becomes ``sub.example.co.uk``,
    not ``example.co.uk``. For effective-TLD-based grouping, post-process the
    output yourself with ``tldextract``.

    Parameters
    ----------
    df : pl.DataFrame
        Input table of posts. Same requirements as :func:`extract_urls`.
    sender_col : str, default "sender"
        Author column. Values passed through untouched.
    post_col : str, default "post"
        Post-text column. Must be Utf8.
    datetime_col : Optional[str], default "datetime"
        Timestamp column to carry through. Pass ``None`` to drop it.
    domain_col : str, default "domain"
        Name of the domain column in the output.
    strip_www : bool, default True
        If True, drop a leading ``www.`` from each host so ``www.bbc.co.uk``
        and ``bbc.co.uk`` collapse to the same domain. Set False to keep
        the ``www.`` subdomain distinct.

    Returns
    -------
    pl.DataFrame
        Long-form edge list with columns
        ``[sender_col, domain_col(, datetime_col)]``, one row per extracted
        domain (one per URL — duplicates within a post are kept, so a post
        linking ``x.com`` twice produces two rows).

    Raises
    ------
    ValidationError
        If required columns are missing or ``post_col`` is not a string column.

    Complexity
    ----------
    Same as :func:`extract_urls`: O(N · L). Domain reduction is a constant-
    overhead pair of regex replacements on the extracted URL column.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "sender": ["alice"],
    ...     "post":   ["see https://www.bbc.co.uk/news and https://nytimes.com/x?y=1"],
    ...     "datetime": ["2024-01-01"],
    ... })
    >>> extract_domains(df)["domain"].to_list()
    ['bbc.co.uk', 'nytimes.com']
    """
    urls = extract_urls(
        df,
        sender_col=sender_col,
        post_col=post_col,
        datetime_col=datetime_col,
        url_col="_glp_url_tmp",
    )

    # Strip scheme, then take everything up to the first path/query/fragment
    # delimiter or port — that's the host. Lowercase since hostnames are
    # case-insensitive. ``(?i)`` makes scheme/www stripping case-insensitive.
    host_expr = (
        pl.col("_glp_url_tmp")
        .str.replace(r"(?i)^https?://", "")
        .str.extract(r"^([^/?#:]+)", 1)
        .str.to_lowercase()
    )
    if strip_www:
        host_expr = host_expr.str.replace(r"^www\.", "")

    return (
        urls.with_columns(host_expr.alias(domain_col))
        .drop("_glp_url_tmp")
        .filter(pl.col(domain_col).is_not_null() & (pl.col(domain_col) != ""))
    )


def extract_keywords(
    df: pl.DataFrame,
    keywords: Sequence[str],
    sender_col: str = "sender",
    post_col: str = "post",
    datetime_col: Optional[str] = "datetime",
    keyword_col: str = "keyword",
    case_sensitive: bool = False,
    whole_word: bool = True,
) -> pl.DataFrame:
    """
    Extract occurrences of a fixed keyword list from posts.

    Builds a single alternation regex from ``keywords`` (with ``re.escape`` so
    keywords containing regex metacharacters are treated as literals) and runs
    it against ``post_col``. Each match emits one output row, so a post that
    mentions the same keyword twice produces two rows — this preserves the
    raw mention count for downstream weighting.

    Parameters
    ----------
    df : pl.DataFrame
        Input table of posts.
    keywords : Sequence[str]
        Literal keywords/phrases to look for. Regex metacharacters are escaped
        — pass plain strings, not patterns. An empty sequence yields an empty
        result with the expected schema.
    sender_col : str, default "sender"
        Author column. Passed through untouched.
    post_col : str, default "post"
        Post-text column. Must be Utf8.
    datetime_col : Optional[str], default "datetime"
        Timestamp column to carry through. Pass ``None`` to drop it.
    keyword_col : str, default "keyword"
        Name of the keyword column in the output.
    case_sensitive : bool, default False
        If False (the default), matching is case-insensitive and the extracted
        keyword column is lowercased so all variants of a keyword (``Climate``,
        ``CLIMATE``, ``climate``) collapse to one node in downstream graphs.
        If True, both matching and the output keyword preserve original case.
    whole_word : bool, default True
        If True (the default), wrap the alternation in ``\\b`` word boundaries
        so ``"climate"`` matches in ``"climate change"`` but not in
        ``"acclimate"``. Set False for substring matching. Useful when one of
        the keywords is itself a multi-word phrase containing spaces (word
        boundaries still work but at the phrase edges, not within).

    Returns
    -------
    pl.DataFrame
        Long-form edge list with columns
        ``[sender_col, keyword_col(, datetime_col)]``, one row per match.

    Raises
    ------
    ValidationError
        If required columns are missing or ``post_col`` is not a string column.

    Complexity
    ----------
    O(N · L · |K|) worst case where |K| is the number of keywords, but in
    practice closer to O(N · L) because the underlying regex engine compiles
    the alternation into a single automaton and scans each post once.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "sender":   ["alice", "bob"],
    ...     "post":     ["climate policy and vaccine debate",
    ...                  "Climate is changing"],
    ...     "datetime": ["2024-01-01", "2024-01-02"],
    ... })
    >>> extract_keywords(df, ["climate", "vaccine"])["keyword"].to_list()
    ['climate', 'vaccine', 'climate']
    """
    _validate_input(df, sender_col, post_col, datetime_col)

    if not keywords:
        # Build an empty result with the right schema by selecting an empty
        # slice of the input and adding a typed empty keyword column.
        carry = _carry_cols(sender_col, datetime_col)
        return df.head(0).select(
            carry + [pl.lit(None, dtype=pl.Utf8).alias(keyword_col)]
        )

    escaped = [re.escape(k) for k in keywords]
    alternation = "|".join(escaped)
    if whole_word:
        pattern = rf"\b(?:{alternation})\b"
    else:
        pattern = rf"(?:{alternation})"
    if not case_sensitive:
        pattern = f"(?i){pattern}"

    result = _extract_long(
        df,
        sender_col=sender_col,
        post_col=post_col,
        datetime_col=datetime_col,
        pattern=pattern,
        out_col=keyword_col,
    )

    if not case_sensitive:
        result = result.with_columns(pl.col(keyword_col).str.to_lowercase())

    return result
