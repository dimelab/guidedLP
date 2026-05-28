"""
Text-content extraction from a ``[sender, post, datetime]`` post table.

Each function in this module takes a Polars DataFrame describing senders and
their posts and returns a bipartite edge list mapping each sender to the
content elements they shared.

- ``extract_urls`` and ``extract_domains`` emit one row per match (long form).
- ``extract_keywords`` aggregates by default to ``[sender, keyword, mentions,
  first_seen]`` to keep memory bounded on large corpora — when ``keywords``
  is left as ``None`` (the default) every word in every post is treated as a
  candidate, so the un-aggregated row count is N_posts × words_per_post which
  can easily reach billions. Aggregation collapses to N_senders × vocab_size,
  which is typically 2–4 orders of magnitude smaller. Pass ``output="long"``
  to keep one row per mention, or ``output="lazy"`` to get a ``LazyFrame`` you
  can ``sink_parquet`` straight to disk without materializing.

Input row order is preserved (in long mode); original IDs in the sender column
are passed through untouched so the result drops straight into
:func:`guidedLP.network.construction.build_graph_from_edgelist`.

All extraction is done with Polars' native regex/string kernels (Rust ``regex``
crate, not Python ``re``), so the work is vectorized over the column and runs
in a single pass without materializing intermediate Python lists.

The NLP-flavored options on ``extract_keywords`` (``stop_words=``, ``stem=``,
``lemmatize=``, auto language detection) require the optional ``[nlp]`` extra
to be installed: ``pip install 'guidedLP[nlp]'``. Without it, the basic
"every word" / "literal keyword match" paths still work.
"""

from __future__ import annotations

from typing import Callable, FrozenSet, Iterable, List, Optional, Sequence, Union

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

# Word pattern used by ``extract_keywords`` when ``keywords=None`` (tokenize
# every word in the post). ``\w`` is Unicode-aware in the Rust regex engine
# polars uses, so this matches non-ASCII letters (å, ø, é, ü, …) correctly
# without needing ``\p{L}``. Apostrophe-containing words like "don't" get
# split into "don" and "t" — a known limitation of simple word tokenization;
# users who need linguistic tokenization should pre-tokenize and pass the
# tokens via ``keywords=``.
_WORD_PATTERN = r"\b\w+\b"

_VALID_OUTPUT_MODES = frozenset({"aggregated", "long", "lazy"})


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


# ---------------------------------------------------------------------------
# Keyword-extraction helpers
# ---------------------------------------------------------------------------


def _wants_auto_language(opt: object) -> bool:
    """True if a stop_words/stem/lemmatize option asks for auto-detected language."""
    return opt is True


def _resolve_op_language(
    opt: Union[bool, str, None], auto_language: Optional[str]
) -> Optional[str]:
    """
    Resolve a stem/lemmatize argument to a concrete language code, or None.

    ``False``/``None`` → operation disabled (returns None).
    ``True``           → use auto_language (must have been pre-resolved).
    ``str``            → use that explicit ISO 639-1 code.
    """
    if opt is False or opt is None:
        return None
    if opt is True:
        if auto_language is None:
            raise ValidationError(
                "Language auto-detection failed to produce a code but was "
                "requested by stem=True / lemmatize=True. Pass an explicit "
                "language= code instead."
            )
        return auto_language
    if isinstance(opt, str):
        return opt
    raise ValidationError(
        f"Expected bool or str for stem/lemmatize option, got {type(opt).__name__}"
    )


def _preprocess_keyword_list(
    keywords: Sequence[str],
    case_sensitive: bool,
    stemmer: Optional[Callable[[str], str]],
    lemmatizer: Optional[Callable[[str], str]],
) -> FrozenSet[str]:
    """
    Apply the same case/stem/lemma transformations to the user's keyword list
    that we apply to extracted words. Otherwise ``keywords=["climate"]`` with
    ``stem=True`` would never match anything because the extracted column
    contains ``"climat"`` after stemming.

    Multi-word keywords (containing whitespace) raise — the all-word extraction
    path can't match phrases.
    """
    out: List[str] = []
    for raw in keywords:
        if " " in raw or "\t" in raw:
            raise ValidationError(
                f"Multi-word keyword '{raw}' is not supported in the all-word "
                "extraction path. Pass single words; for phrase matching, "
                "preprocess your posts (e.g. join 'climate change' into "
                "'climate_change' before calling extract_keywords).",
                field="keywords",
                value=raw,
            )
        w = raw if case_sensitive else raw.lower()
        if stemmer is not None:
            w = stemmer(w)
        if lemmatizer is not None:
            w = lemmatizer(w)
        out.append(w)
    return frozenset(out)


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------


def extract_keywords(
    df: pl.DataFrame,
    keywords: Optional[Sequence[str]] = None,
    *,
    # Column names
    sender_col: str = "sender",
    post_col: str = "post",
    datetime_col: Optional[str] = "datetime",
    keyword_col: str = "keyword",
    # Output shape
    output: str = "aggregated",
    count_col: str = "mentions",
    first_seen_col: str = "first_seen",
    # Matching
    case_sensitive: bool = False,
    # NLP preprocessing (all default to off — opting in requires [nlp] extras)
    stop_words: Union[bool, str, Iterable[str], None] = False,
    stem: Union[bool, str, None] = False,
    lemmatize: Union[bool, str, None] = False,
    min_word_length: int = 1,
    # Language (explicit override of auto-detection)
    language: Optional[str] = None,
    language_sample_size: int = 1000,
    language_sample_seed: int = 42,
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Tokenize posts and emit a sender→keyword edge list, with optional NLP
    preprocessing and aggregation.

    Default behavior (no arguments beyond ``df``) tokenizes every post into
    individual words via ``\\b\\w+\\w`` (Unicode-aware), lowercases, and
    aggregates to ``[sender, keyword, mentions, first_seen]`` — one row per
    distinct (sender, word) pair, with the mention count and earliest
    timestamp. This collapses 10M posts × ~80 unique-words/post worth of
    raw mentions down to ``n_senders × vocab_size``, which typically fits in
    memory comfortably.

    Pass ``keywords=[...]`` to filter the output to a specific vocabulary
    (post-tokenization, post-preprocessing — so a keyword ``"climate"`` with
    ``stem=True`` is also stemmed to ``"climat"`` before comparison).

    Parameters
    ----------
    df : pl.DataFrame
        Input post table.
    keywords : Sequence[str] or None, default None
        Vocabulary to restrict the output to. ``None`` (the default) emits all
        words found in the posts. When provided, must be single-word tokens —
        multi-word phrases raise ValidationError.
    sender_col, post_col, datetime_col : str
        Column names. ``post_col`` must be Utf8. ``datetime_col=None`` drops
        timestamps entirely (and consequently the ``first_seen`` column).
    keyword_col : str, default "keyword"
        Output column holding the extracted word.
    output : {"aggregated", "long", "lazy"}, default "aggregated"
        - ``"aggregated"`` — one row per (sender, keyword) with ``mentions``
          and (if ``datetime_col`` is set) ``first_seen``. Default because it's
          the memory-bounded form most downstream tasks need.
        - ``"long"`` — one row per mention (no count/first_seen columns).
          Useful for temporal analysis but row count = total mentions across
          all posts, so use carefully on large corpora.
        - ``"lazy"`` — return a ``pl.LazyFrame`` of the long form so the caller
          can ``.sink_parquet()`` directly to disk without materializing.
          Incompatible with ``stem=`` / ``lemmatize=`` (they force collection).
    count_col, first_seen_col : str
        Output column names for ``output="aggregated"``.
    case_sensitive : bool, default False
        If False (default), extracted words are lowercased and matching against
        ``keywords`` / ``stop_words`` is case-insensitive. If True, original
        case is preserved end-to-end (and you'll need to supply ``stop_words``
        / ``keywords`` in matching case).
    stop_words : bool | str | Iterable[str] | None, default False
        Stop-word filtering applied after tokenization.

        - ``False`` / ``None`` — no filtering.
        - ``True`` — auto-detect language from a random sample of posts (see
          ``language_sample_size``) and use that language's built-in list.
        - ``str`` (e.g. ``"en"``, ``"da"``) — use that ISO 639-1 language's
          built-in list.
        - ``Iterable[str]`` — use exactly this custom list (lowercased).

        Built-in lists require ``pip install 'guidedLP[nlp]'`` (provides
        ``stopwordsiso``). Custom-list mode has no extra dependency.
    stem : bool | str | None, default False
        Snowball stemming applied to each extracted word.

        - ``False`` / ``None`` — no stemming.
        - ``True`` — auto-detect language, stem in that language.
        - ``str`` — explicit ISO 639-1 code.

        Requires ``pip install 'guidedLP[nlp]'`` (provides ``snowballstemmer``).
    lemmatize : bool | str | None, default False
        Same interface as ``stem``, but uses ``simplemma`` (provided by the
        ``[nlp]`` extra). Applied AFTER stemming if both are requested — an
        unusual combination but the function doesn't forbid it.
    min_word_length : int, default 1
        Drop tokens shorter than this many characters. ``1`` (default) keeps
        everything; ``2`` is a common choice to drop single-letter artifacts.
    language : str, optional
        Explicit ISO 639-1 code that overrides auto-detection. Set this when
        you already know the corpus language to skip the detection sample
        (saves ~1 s on large corpora).
    language_sample_size, language_sample_seed : int
        Sample configuration for language detection, only consulted when auto-
        detection is actually triggered.

    Returns
    -------
    pl.DataFrame or pl.LazyFrame
        Schema by ``output`` mode:

        - ``"aggregated"`` → ``[sender_col, keyword_col, count_col,
          first_seen_col]`` (last column omitted when ``datetime_col=None``).
        - ``"long"`` → ``[sender_col(, datetime_col), keyword_col]``.
        - ``"lazy"`` → ``LazyFrame`` of the long form.

    Raises
    ------
    ValidationError
        Missing columns, wrong post dtype, invalid ``output`` mode, multi-word
        keyword, or invalid stem/lemma argument shape.
    ImportError
        If ``stop_words``/``stem``/``lemmatize``/auto-language is requested
        but the corresponding ``[nlp]`` package isn't installed. The error
        message tells you which install command to run.

    Complexity
    ----------
    Tokenization + filtering is O(N · L) where N is the number of posts and L
    is the average post length. Aggregation adds an O(M log M) ``group_by``
    step where M is the post-filter row count. Stemming/lemmatization adds an
    O(V) Python-level pass over the (post-aggregation) unique vocabulary V —
    so it's cheap relative to tokenization.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "sender":   ["alice", "alice", "bob"],
    ...     "post":     ["climate climate vaccine",
    ...                  "climate policy",
    ...                  "Climate matters"],
    ...     "datetime": ["2024-01-01", "2024-01-03", "2024-01-02"],
    ... })

    Default = all words, aggregated:

    >>> result = extract_keywords(df).sort(["sender", "keyword"])
    >>> result.select(["sender", "keyword", "mentions"]).to_dicts()  # doctest: +SKIP
    [{'sender': 'alice', 'keyword': 'climate', 'mentions': 3},
     {'sender': 'alice', 'keyword': 'policy', 'mentions': 1},
     {'sender': 'alice', 'keyword': 'vaccine', 'mentions': 1},
     {'sender': 'bob', 'keyword': 'climate', 'mentions': 1},
     {'sender': 'bob', 'keyword': 'matters', 'mentions': 1}]

    Filter to a vocabulary:

    >>> extract_keywords(df, keywords=["climate", "vaccine"]).sort(["sender"])  # doctest: +SKIP
    """
    _validate_input(df, sender_col, post_col, datetime_col)

    if output not in _VALID_OUTPUT_MODES:
        raise ValidationError(
            f"output must be one of {sorted(_VALID_OUTPUT_MODES)}, got {output!r}",
            field="output",
            value=output,
        )

    # ---- Resolve language (auto-detect if any preprocessing asks for it) ---
    needs_auto = any(
        _wants_auto_language(opt) for opt in (stop_words, stem, lemmatize)
    )
    auto_lang: Optional[str] = language
    if needs_auto and auto_lang is None:
        from guidedLP.preprocessing._language import detect_majority_language

        auto_lang = detect_majority_language(
            df,
            post_col=post_col,
            sample_size=language_sample_size,
            seed=language_sample_seed,
        )

    # ---- Resolve stop word set --------------------------------------------
    from guidedLP.preprocessing._stop_words import resolve_stop_words

    stop_set = resolve_stop_words(stop_words, auto_lang or "")

    # ---- Resolve stemmer / lemmatizer -------------------------------------
    stem_lang = _resolve_op_language(stem, auto_lang)
    lemma_lang = _resolve_op_language(lemmatize, auto_lang)

    stemmer: Optional[Callable[[str], str]] = None
    if stem_lang is not None:
        from guidedLP.preprocessing._nlp import make_stemmer

        stemmer = make_stemmer(stem_lang)

    lemmatizer: Optional[Callable[[str], str]] = None
    if lemma_lang is not None:
        from guidedLP.preprocessing._nlp import make_lemmatizer

        lemmatizer = make_lemmatizer(lemma_lang)

    if output == "lazy" and (stemmer is not None or lemmatizer is not None):
        raise ValidationError(
            "output='lazy' is incompatible with stem= / lemmatize= because "
            "those steps force materialization. Use output='aggregated' "
            "(default) or output='long'.",
            field="output",
        )

    # ---- Build the lazy tokenization pipeline ------------------------------
    carry = _carry_cols(sender_col, datetime_col)

    # Lowercase BEFORE extraction so str.extract_all sees lowercase text — same
    # cost (one pass over the post column) but avoids materializing the
    # uppercase word column we'd otherwise have to lowercase per row.
    post_expr = pl.col(post_col)
    if not case_sensitive:
        post_expr = post_expr.str.to_lowercase()

    pipeline = (
        df.lazy()
        .select(
            carry
            + [post_expr.str.extract_all(_WORD_PATTERN).alias(keyword_col)]
        )
        .explode(keyword_col)
        .filter(pl.col(keyword_col).is_not_null())
    )

    if min_word_length > 1:
        pipeline = pipeline.filter(
            pl.col(keyword_col).str.len_chars() >= min_word_length
        )

    if stop_set:
        pipeline = pipeline.filter(
            ~pl.col(keyword_col).is_in(list(stop_set))
        )

    # ---- Branch on output mode --------------------------------------------
    if output == "lazy":
        if keywords is not None:
            kw_set = _preprocess_keyword_list(keywords, case_sensitive, None, None)
            pipeline = pipeline.filter(pl.col(keyword_col).is_in(list(kw_set)))
        return pipeline

    if output == "long":
        result = pipeline.collect()
        if stemmer is not None:
            result = result.with_columns(
                pl.col(keyword_col).map_elements(stemmer, return_dtype=pl.Utf8)
            )
        if lemmatizer is not None:
            result = result.with_columns(
                pl.col(keyword_col).map_elements(lemmatizer, return_dtype=pl.Utf8)
            )
        if keywords is not None:
            kw_set = _preprocess_keyword_list(
                keywords, case_sensitive, stemmer, lemmatizer
            )
            result = result.filter(pl.col(keyword_col).is_in(list(kw_set)))
        return result

    # ---- output == "aggregated" -------------------------------------------
    agg_exprs: List[pl.Expr] = [pl.len().alias(count_col)]
    if datetime_col is not None:
        agg_exprs.append(pl.col(datetime_col).min().alias(first_seen_col))

    aggregated = (
        pipeline.group_by([sender_col, keyword_col])
        .agg(agg_exprs)
        .collect()
    )

    # Stem/lemma is applied AFTER the first group_by — at this point we're
    # working on the much smaller (sender, word) deduplicated frame, so the
    # Python-level map_elements is cheap. We then re-group to collapse words
    # that share a stem/lemma (e.g. "climate"/"climates" → "climat").
    if stemmer is not None or lemmatizer is not None:
        if stemmer is not None:
            aggregated = aggregated.with_columns(
                pl.col(keyword_col).map_elements(stemmer, return_dtype=pl.Utf8)
            )
        if lemmatizer is not None:
            aggregated = aggregated.with_columns(
                pl.col(keyword_col).map_elements(lemmatizer, return_dtype=pl.Utf8)
            )
        regroup: List[pl.Expr] = [pl.col(count_col).sum().alias(count_col)]
        if datetime_col is not None:
            regroup.append(pl.col(first_seen_col).min().alias(first_seen_col))
        aggregated = aggregated.group_by([sender_col, keyword_col]).agg(regroup)

    if keywords is not None:
        kw_set = _preprocess_keyword_list(
            keywords, case_sensitive, stemmer, lemmatizer
        )
        aggregated = aggregated.filter(pl.col(keyword_col).is_in(list(kw_set)))

    return aggregated
