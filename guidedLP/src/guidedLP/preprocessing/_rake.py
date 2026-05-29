"""
RAKE (Rapid Automatic Keyword Extraction) scoring for ``extract_keywords``.

Implements the classic Rose et al. (2010) RAKE algorithm in pure Python so the
preprocessing module doesn't pull in ``rake-nltk`` (which transitively requires
NLTK + an ``nltk.download(...)`` step). The stop-word list is sourced from the
existing ``_stop_words.resolve_stop_words`` infrastructure, so the same
``stop_words=`` argument users already know controls phrase boundaries here.

The algorithm:

1. **Phrase extraction.** Each post is split on punctuation, then within each
   chunk the token stream is cut at every stop word, producing candidate
   phrases (sequences of content words). Phrases longer than
   ``max_phrase_length`` are discarded.
2. **Word scoring** (corpus-level). For every candidate phrase occurrence we
   accumulate ``freq[w]`` (number of phrases containing ``w``) and
   ``degree[w]`` (sum of phrase lengths for phrases containing ``w``). The
   per-word score is ``degree[w] / freq[w]`` — words that tend to appear in
   longer phrases score higher.
3. **Phrase scoring.** Each candidate phrase's score is the sum of its word
   scores. Unigrams that only ever appear alone get score 1; words that
   regularly co-occur in longer phrases get higher scores and lift the phrase
   they appear in.
4. **Per-sender top-N.** Stats are computed across the whole corpus so the
   scoring reflects corpus-wide informativeness, but the *output* keeps the
   top ``top_n`` phrases each sender actually used (ranked by global score,
   ties broken by per-sender mention count). This guarantees no sender ends
   up with an empty keyword list.

The phrase extraction step is a per-row Python UDF rather than a Polars regex
because RAKE's phrase boundaries depend on a (potentially large) stop-word
set whose membership test is much faster as a Python ``frozenset`` lookup than
as a long regex alternation. The downside is the GIL-bound loop over posts —
expect a noticeably slower wall-clock than the ``method="all"`` path on large
corpora.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, FrozenSet, List, Optional

import polars as pl


# Sentence/clause boundary: any run of one or more non-word, non-whitespace
# characters (punctuation, symbols). Matches the original RAKE paper's
# treatment of punctuation as a hard phrase boundary.
_PUNCT_SPLIT_RE = re.compile(r"[^\w\s]+", re.UNICODE)

# Word token pattern — same as the all-words path. Unicode-aware via ``\w``.
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def make_phrase_extractor(
    stop_set: FrozenSet[str],
    min_word_length: int,
    case_sensitive: bool,
    stemmer: Optional[Callable[[str], str]],
    lemmatizer: Optional[Callable[[str], str]],
) -> Callable[[Optional[str]], List[str]]:
    """
    Build a per-post phrase extractor for use as a Polars ``map_elements`` UDF.

    The returned callable takes a post string and returns a list of phrase
    strings (each phrase is a space-joined sequence of content words). Stop
    words, punctuation, and tokens shorter than ``min_word_length`` all act as
    phrase boundaries. Phrases of all lengths are emitted — the
    ``max_phrase_length`` cap is applied *after* scoring (in
    :func:`_extract_keywords_rake`) so long phrases still contribute to the
    corpus-level word stats. This matters most for ``max_phrase_length=1``
    (unigrams only): a word like "climate" in "global climate crisis" still
    accrues degree from the bigger phrase even though only the unigram is
    emitted.

    Stemming/lemmatization is applied at the token level, *before* the
    stop-word check, so that "climates" (which a user-supplied list may not
    contain) gets stemmed to "climat" and "climate" canonicalizes too. This
    means stemmed/lemmatized forms also end up in the emitted phrase strings,
    which is consistent with the rest of the extractor.
    """

    def transform(word: str) -> str:
        if not case_sensitive:
            word = word.lower()
        if stemmer is not None:
            word = stemmer(word)
        if lemmatizer is not None:
            word = lemmatizer(word)
        return word

    def extract(text: Optional[str]) -> List[str]:
        if text is None or not text:
            return []

        phrases: List[str] = []
        # Punctuation is a hard phrase boundary, so split first then tokenize.
        for chunk in _PUNCT_SPLIT_RE.split(text):
            if not chunk:
                continue
            current: List[str] = []
            for tok in _WORD_RE.findall(chunk):
                w = transform(tok)
                # A stop word OR a too-short token ends the current phrase.
                if w in stop_set or len(w) < min_word_length:
                    if current:
                        phrases.append(" ".join(current))
                    current = []
                else:
                    current.append(w)
            if current:
                phrases.append(" ".join(current))
        return phrases

    return extract


def compute_phrase_scores(
    phrases: List[str],
    counts: List[int],
) -> List[float]:
    """
    RAKE phrase scoring given per-phrase corpus occurrence counts.

    Parameters
    ----------
    phrases : list[str]
        Unique candidate phrases (each a space-joined word sequence).
    counts : list[int]
        Number of occurrences of each phrase in the corpus, aligned with
        ``phrases`` index-for-index.

    Returns
    -------
    list[float]
        Phrase scores in the same order. Sum-of-word-scores per RAKE's
        original definition.

    Notes
    -----
    Word stats are weighted by phrase occurrence count, not unique-phrase
    count. A phrase that appears 50 times contributes 50× to its words'
    freq/degree, so common phrases lift their constituent words' scores.
    This matches the original Rose et al. formulation.
    """
    word_freq: Counter = Counter()
    word_degree: Counter = Counter()

    # Pre-split phrases once; reused in the scoring pass below.
    split_cache: List[List[str]] = [p.split() for p in phrases]

    for words, count in zip(split_cache, counts):
        n = len(words)
        for w in words:
            word_freq[w] += count
            word_degree[w] += count * n

    # Word score is degree/freq. freq is always > 0 here because every word
    # in the loop above contributed to its own freq.
    word_score = {w: word_degree[w] / word_freq[w] for w in word_freq}

    return [sum(word_score[w] for w in words) for words in split_cache]


def attach_phrase_scores(
    phrased: pl.DataFrame,
    keyword_col: str,
    score_col: str,
) -> pl.DataFrame:
    """
    Compute corpus-level RAKE phrase scores and join them onto ``phrased``.

    ``phrased`` is the exploded ``(sender, keyword(, datetime))`` table where
    each row is one phrase occurrence. We group to count occurrences per
    unique phrase, score those, and broadcast the score back onto every row.
    """
    counts_df = (
        phrased.group_by(keyword_col)
        .agg(pl.len().alias("_glp_rake_count"))
    )

    phrases = counts_df[keyword_col].to_list()
    counts = counts_df["_glp_rake_count"].to_list()
    scores = compute_phrase_scores(phrases, counts)

    score_df = pl.DataFrame(
        {keyword_col: phrases, score_col: scores}
    )
    return phrased.join(score_df, on=keyword_col)
