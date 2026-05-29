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
  Set ``method="rake"`` to switch from "every word" tokenization to RAKE
  (Rapid Automatic Keyword Extraction) — phrases are scored across the whole
  corpus and each sender keeps their top ``top_n`` highest-scoring phrases
  (unigrams up to ``max_phrase_length``-grams).

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
_VALID_METHODS = frozenset({"all", "rake"})


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------

# Known tracking / analytics query parameters stripped from URLs when
# ``normalize=True``. Names are matched case-insensitively. Curated from
# spreadAnalysis's ``link_utils.py`` plus a handful of widely-deployed click-ID
# parameters; ambiguous single-letter or very generic names (``r``, ``type``,
# ``set``, ``print``, ``do``, ``from``, ``ref``, ``src``, ``feature``, ``list``,
# ``keywords``) are intentionally excluded to avoid stripping path-meaningful
# parameters. Extend by editing this tuple and re-deriving the regexes below.
_TRACKING_PARAMS = (
    # Google Analytics / UTM family
    "utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term",
    "utm_name", "utm_referrer", "utm_id", "utm_creative_format",
    "utm_marketing_tactic",
    # Click IDs (ad networks)
    "fbclid",                                # Facebook
    "gclid", "gclsrc", "dclid",              # Google
    "msclkid",                               # Microsoft / Bing
    "yclid",                                 # Yandex
    "twclid",                                # Twitter
    "li_fat_id",                             # LinkedIn
    # Email-marketing trackers
    "mc_cid", "mc_eid",                      # Mailchimp
    "mkt_tok",                               # Marketo
    "_hsenc", "_hsmi",                       # HubSpot
    "vero_id", "vero_conv",                  # Vero
    # Platform-specific share trackers
    "igshid",                                # Instagram
    "si",                                    # Spotify, YouTube short URLs
    "__twitter_impression",                  # Twitter beacon
    "ref_src",                               # Twitter referral
    "trk", "trkCampaign",                    # LinkedIn
    # News / publisher trackers
    "ncid", "ocid",                          # Yahoo, MSN
    "ito",                                   # Microsoft News
    "smid",                                  # NYT
    "wt_zmc",                                # Die Zeit
    "cmp", "CMP",                            # Various
    "soc_src", "soc_trk",                    # Generic social
    "at_medium", "at_campaign",              # BBC
    "dmcid",
    "mktci", "mktcval", "mktcid",            # Marketo variants
    "ns_mchannel", "ns_campaign",
    "wtrid", "rcode", "smtyp",
    "cid_source", "flattrss_redirect",
    # Yahoo GUCE / consent banners
    "guccounter", "guce_referrer", "guce_referrer_sig",
    # Misc
    "refsrc", "noredirect", "highlightedUpdateUrns",
)

_TRACKING_ALTERNATION = "|".join(_TRACKING_PARAMS)

# Pass 1: strip a contiguous run of tracking params at the *start* of the
# query string, preserving the leading ``?`` so any kept params downstream
# remain parseable. The trailing ``&?`` consumes the separator before the
# next param so we don't leave a stray ``?&`` behind.
_TRACKING_LEADING_REGEX = (
    r"(?i)\?(?:(?:" + _TRACKING_ALTERNATION + r")=[^&#]*&?)+"
)
# Pass 2: strip any tracking param that appears later in the query string
# (preceded by ``&``). Pass 1 already consumed leading runs, so the remaining
# matches here are always non-first.
_TRACKING_TRAILING_REGEX = (
    r"(?i)&(?:" + _TRACKING_ALTERNATION + r")=[^&#]*"
)


def _normalize_url_expr(expr: pl.Expr) -> pl.Expr:
    """
    Build a Polars expression that canonicalizes a column of URLs.

    Steps, in order:

    1. Unescape HTML-encoded ampersands (``amp;`` and ``amp%3B``) that often
       leak in from HTML-pasted sources.
    2. Cut markdown link tails: a stray ``](...`` is treated as a sentinel
       and everything from there to the end of the URL string is dropped.
    3. Lowercase the scheme (``HTTPS://`` → ``https://``).
    4. Lowercase the host portion only — the path/query/fragment keeps its
       original case since those are case-sensitive in HTTP.
    5. Canonicalize YouTube URLs:
       - ``youtu.be/<id>`` → ``https://youtube.com/watch?v=<id>``
       - ``youtube.com/watch?...&v=<id>&...`` → ``https://youtube.com/watch?v=<id>``
    6. Strip well-known tracking parameters (``utm_*``, ``fbclid``, ``gclid``,
       ``mc_cid``, …) — see :data:`_TRACKING_PARAMS`.
    7. Clean up dangling ``?`` or ``&`` left after parameter removal
       (e.g. ``site.com?`` → ``site.com``, ``site.com?#frag`` → ``site.com#frag``).
    8. Strip trailing sentence punctuation.

    All steps are pure Polars/regex expressions — no Python row loops, so the
    whole pipeline stays vectorized over the column.
    """
    # 1) Unescape ampersands from HTML-pasted URLs.
    expr = expr.str.replace_all("amp%3B", "").str.replace_all("amp;", "")

    # 2) Markdown ``](`` tail cut. The default URL regex already excludes
    # ``]`` and ``(`` so this is defensive — it covers callers that pass in
    # pre-extracted URLs from a looser pattern.
    expr = expr.str.replace(r"\]\(.*$", "")

    # 3) Scheme → lowercase. The replacement string is literal lowercase, so
    # any case variant of the case-insensitive match is normalized.
    expr = expr.str.replace(r"(?i)^https://", "https://")
    expr = expr.str.replace(r"(?i)^http://", "http://")

    # 4) Host → lowercase, preserving path case. Split into scheme + host +
    # rest via regex captures, lowercase the host, recombine. ``str.replace``
    # of an unmatched pattern is a no-op, so bare ``www.``-style URLs (no
    # scheme) flow through with an empty scheme part.
    after_scheme = expr.str.replace(r"^https?://", "")
    scheme_part = expr.str.extract(r"^(https?://)", 1).fill_null("")
    host_part = (
        after_scheme.str.extract(r"^([^/?#]+)", 1)
        .fill_null("")
        .str.to_lowercase()
    )
    path_part = after_scheme.str.replace(r"^[^/?#]+", "")
    expr = scheme_part + host_part + path_part

    # 5) YouTube canonicalization. Both rules anchor on ``^...$`` so they
    # only fire on full-URL matches; non-YouTube URLs flow through.
    expr = expr.str.replace(
        r"^https?://(?:www\.)?youtu\.be/([A-Za-z0-9_-]+).*$",
        "https://youtube.com/watch?v=$1",
    )
    expr = expr.str.replace(
        r"^https?://(?:www\.)?youtube\.com/watch\?(?:[^&]*&)*v=([A-Za-z0-9_-]+).*$",
        "https://youtube.com/watch?v=$1",
    )

    # 6) Tracking-param stripping (two passes — see regex definitions above
    # for why this isn't a single pattern).
    expr = expr.str.replace_all(_TRACKING_LEADING_REGEX, "?")
    expr = expr.str.replace_all(_TRACKING_TRAILING_REGEX, "")

    # 7) Tidy dangling query-string delimiters left after param removal.
    expr = expr.str.replace(r"\?#", "#")
    expr = expr.str.replace(r"\?$", "")
    expr = expr.str.replace(r"&$", "")

    # 8) Trailing sentence punctuation.
    expr = expr.str.strip_chars_end(_TRAILING_PUNCT)

    return expr


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
    normalize: bool = True,
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
    normalize : bool, default True
        If True, canonicalize each extracted URL: lowercase scheme + host
        (preserving path case), strip ``utm_*``/``fbclid``/``gclid``/etc.
        tracking parameters, collapse ``youtu.be/<id>`` and
        ``youtube.com/watch?...&v=<id>`` into a canonical form, and clean up
        HTML-escaped ``amp;``. This is the default because the same article
        shared with different tracking codes would otherwise produce distinct
        URL nodes when fed into ``build_graph_from_edgelist``. Set to False to
        keep the raw extracted URLs and apply only the trailing-punctuation
        strip. See :func:`_normalize_url_expr` for the full step list and
        :data:`_TRACKING_PARAMS` for the parameter list.

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
    engine, then a flat explode and filter. Normalization adds a constant
    number of additional regex passes over the (much shorter) URL column.

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

    if normalize:
        url_expr = _normalize_url_expr(pl.col(url_col))
    else:
        url_expr = pl.col(url_col).str.strip_chars_end(_TRAILING_PUNCT)

    return result.with_columns(url_expr.alias(url_col))


def extract_domains(
    df: pl.DataFrame,
    sender_col: str = "sender",
    post_col: str = "post",
    datetime_col: Optional[str] = "datetime",
    domain_col: str = "domain",
    strip_www: bool = True,
    normalize: bool = True,
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
    normalize : bool, default True
        Forwarded to :func:`extract_urls`. The most domain-visible effect is
        that ``youtu.be/<id>`` URLs collapse to the ``youtube.com`` domain
        rather than being treated as a separate host. Set to False to keep
        platforms' short-URL hosts distinct.

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
        normalize=normalize,
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
    # Extraction method
    method: str = "all",
    top_n: int = 50,
    max_phrase_length: int = 3,
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
    method : {"all", "rake"}, default "all"
        Extraction strategy.

        - ``"all"`` (default) — tokenize every word in every post; the result
          is the full vocabulary (optionally filtered by ``keywords=``).
        - ``"rake"`` — run RAKE (Rapid Automatic Keyword Extraction) over the
          whole corpus to score multi-word keyphrases, then keep each sender's
          top ``top_n`` highest-scoring phrases. Output is always aggregated;
          ``output="long"``/``"lazy"`` and explicit ``keywords=`` are not
          supported in this mode. Requires a non-empty ``stop_words`` (RAKE
          splits phrases at stop words, so passing ``stop_words=False`` would
          make every post a single phrase). Pure-Python implementation — no
          extra dependency beyond what ``stop_words=`` already needs.
    top_n : int, default 50
        ``method="rake"`` only: number of top-scoring phrases to keep per
        sender. Senders with fewer distinct candidate phrases keep all they
        have (no padding).
    max_phrase_length : int, default 3
        ``method="rake"`` only: preferred maximum number of words per output
        phrase. Defaults to 3 = unigrams + bigrams + trigrams. Set to 1 to
        prefer unigrams, 2 for unigrams + bigrams. This is a *soft* cap, not
        a hard filter: when ranking each sender's top-``top_n`` phrases,
        phrases meeting the length cap sort before longer phrases, but if a
        sender doesn't have ``top_n`` conforming phrases the remainder is
        filled in from their longer phrases. This guarantees every sender
        with any extractable content gets at least one keyphrase edge
        (important when the output feeds bipartite graph construction).
        Longer phrases are also always used to compute the corpus-level word
        scores, so e.g. "climate" in "global climate crisis" still accrues
        degree from the 3-word phrase even at ``max_phrase_length=1``.
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
        keyword, invalid stem/lemma argument shape, unknown ``method``, or any
        of the RAKE-mode incompatibilities (``output != "aggregated"``,
        ``keywords`` non-None, missing ``stop_words``, ``top_n < 1``,
        ``max_phrase_length < 1``).
    ImportError
        If ``stop_words``/``stem``/``lemmatize``/auto-language is requested
        but the corresponding ``[nlp]`` package isn't installed. The error
        message tells you which install command to run.

    Complexity
    ----------
    ``method="all"``: tokenization + filtering is O(N · L) where N is the
    number of posts and L is the average post length. Aggregation adds an
    O(M log M) ``group_by`` step where M is the post-filter row count.
    Stemming/lemmatization adds an O(V) Python-level pass over the
    (post-aggregation) unique vocabulary V — cheap relative to tokenization.

    ``method="rake"`` adds a Python-level per-post phrase-extraction UDF
    (O(N · L) but unvectorized — slower wall-clock per row than the all-words
    regex pass) plus an O(P) word-stats sweep over the P unique candidate
    phrases. Per-sender top-N is an O(M log M) sort + group_by + head. RAKE is
    typically a few × slower than the all-words path on the same corpus; budget
    accordingly for >10M posts.

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

    RAKE — top-N corpus-scored keyphrases (up to trigrams) per sender:

    >>> extract_keywords(                                           # doctest: +SKIP
    ...     df, method="rake", top_n=5, max_phrase_length=3,
    ...     stop_words=True,  # required for RAKE; "True" auto-detects language
    ... )
    """
    _validate_input(df, sender_col, post_col, datetime_col)

    if output not in _VALID_OUTPUT_MODES:
        raise ValidationError(
            f"output must be one of {sorted(_VALID_OUTPUT_MODES)}, got {output!r}",
            field="output",
            value=output,
        )

    if method not in _VALID_METHODS:
        raise ValidationError(
            f"method must be one of {sorted(_VALID_METHODS)}, got {method!r}",
            field="method",
            value=method,
        )

    if method == "rake":
        if output != "aggregated":
            raise ValidationError(
                "method='rake' only supports output='aggregated' because the "
                "top-N-per-sender selection requires materializing the score "
                "table. Use output='aggregated' (default) for RAKE.",
                field="output",
                value=output,
            )
        if keywords is not None:
            raise ValidationError(
                "method='rake' is incompatible with an explicit keywords= "
                "filter — RAKE *derives* the vocabulary. Drop keywords= or "
                "use method='all'.",
                field="keywords",
            )
        if stop_words is False or stop_words is None:
            raise ValidationError(
                "method='rake' requires a stop-word list (phrases are split "
                "at stop words). Pass stop_words=True for auto-detection, a "
                "language code like stop_words='en', or a custom iterable.",
                field="stop_words",
            )
        if top_n < 1:
            raise ValidationError(
                f"top_n must be >= 1 for method='rake', got {top_n}",
                field="top_n",
                value=top_n,
            )
        if max_phrase_length < 1:
            raise ValidationError(
                "max_phrase_length must be >= 1 for method='rake', got "
                f"{max_phrase_length}",
                field="max_phrase_length",
                value=max_phrase_length,
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

    # ---- Dispatch on method ------------------------------------------------
    if method == "rake":
        return _extract_keywords_rake(
            df,
            sender_col=sender_col,
            post_col=post_col,
            datetime_col=datetime_col,
            keyword_col=keyword_col,
            count_col=count_col,
            first_seen_col=first_seen_col,
            case_sensitive=case_sensitive,
            top_n=top_n,
            max_phrase_length=max_phrase_length,
            min_word_length=min_word_length,
            stop_set=stop_set,
            stemmer=stemmer,
            lemmatizer=lemmatizer,
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


# ---------------------------------------------------------------------------
# RAKE path
# ---------------------------------------------------------------------------


def _extract_keywords_rake(
    df: pl.DataFrame,
    *,
    sender_col: str,
    post_col: str,
    datetime_col: Optional[str],
    keyword_col: str,
    count_col: str,
    first_seen_col: str,
    case_sensitive: bool,
    top_n: int,
    max_phrase_length: int,
    min_word_length: int,
    stop_set: FrozenSet[str],
    stemmer: Optional[Callable[[str], str]],
    lemmatizer: Optional[Callable[[str], str]],
) -> pl.DataFrame:
    """
    RAKE branch of :func:`extract_keywords`.

    See the RAKE section of ``extract_keywords``'s docstring and
    ``guidedLP.preprocessing._rake`` for algorithm details. This wrapper:

    1. Builds a per-post phrase extractor and applies it to ``post_col``.
    2. Computes corpus-level RAKE scores for each unique phrase.
    3. Aggregates to ``(sender, phrase)`` with mention counts and
       ``first_seen``.
    4. Keeps each sender's top ``top_n`` phrases (ranked by global score;
       per-sender mention count is the tiebreaker so deterministic ordering
       survives equal-score phrases).

    Returns a DataFrame with the same column shape as the ``method="all"``
    aggregated path: ``[sender_col, keyword_col, count_col, first_seen_col?]``.
    """
    from guidedLP.preprocessing._rake import (
        attach_phrase_scores,
        make_phrase_extractor,
    )

    extractor = make_phrase_extractor(
        stop_set=stop_set,
        min_word_length=min_word_length,
        case_sensitive=case_sensitive,
        stemmer=stemmer,
        lemmatizer=lemmatizer,
    )

    carry = _carry_cols(sender_col, datetime_col)
    phrased = (
        df.select(
            carry
            + [
                pl.col(post_col)
                .map_elements(extractor, return_dtype=pl.List(pl.Utf8))
                .alias(keyword_col)
            ]
        )
        .explode(keyword_col)
        .filter(
            pl.col(keyword_col).is_not_null() & (pl.col(keyword_col) != "")
        )
    )

    if phrased.height == 0:
        # No candidate phrases anywhere — return an empty frame with the
        # correct schema so downstream code doesn't crash on a missing column.
        cols = [pl.Series(sender_col, [], dtype=df.schema[sender_col]),
                pl.Series(keyword_col, [], dtype=pl.Utf8),
                pl.Series(count_col, [], dtype=pl.UInt32)]
        if datetime_col is not None:
            cols.append(
                pl.Series(first_seen_col, [], dtype=df.schema[datetime_col])
            )
        return pl.DataFrame(cols)

    score_col = "_glp_rake_score"
    phrased = attach_phrase_scores(phrased, keyword_col, score_col)

    agg_exprs: List[pl.Expr] = [
        pl.len().alias(count_col),
        pl.col(score_col).first().alias(score_col),
    ]
    if datetime_col is not None:
        agg_exprs.append(pl.col(datetime_col).min().alias(first_seen_col))

    aggregated = phrased.group_by([sender_col, keyword_col]).agg(agg_exprs)

    # max_phrase_length is treated as a soft preference, not a hard cap:
    # phrases of acceptable length sort first, longer phrases backfill if a
    # sender doesn't have ``top_n`` short candidates. This guarantees that
    # any sender who posted at least one content word gets at least one
    # keyphrase edge — the explicit no-empty-senders requirement.
    meets_col = "_glp_meets_length"
    aggregated = aggregated.with_columns(
        (pl.col(keyword_col).str.count_matches(r"\S+") <= max_phrase_length)
        .alias(meets_col)
    )

    # Sort order per sender:
    #   1. meets_length (True before False — keeps the uni/bi/tri preference)
    #   2. score (descending — RAKE informativeness)
    #   3. count (descending — frequent ties before rare ones)
    #   4. keyword (ascending — deterministic alphabetic tiebreaker)
    sort_cols = [sender_col, meets_col, score_col, count_col, keyword_col]
    descending = [False, True, True, True, False]
    aggregated = (
        aggregated.sort(sort_cols, descending=descending)
        .group_by(sender_col, maintain_order=True)
        .head(top_n)
        .drop([score_col, meets_col])
    )

    return aggregated
