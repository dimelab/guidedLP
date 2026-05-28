"""
Majority-language detection for preprocessing.

When a caller asks ``extract_keywords`` to apply language-dependent
preprocessing (stop-word filtering, stemming, lemmatization) without
specifying which language, this module is consulted to infer a single
language code for the whole corpus from a random sample of the posts.

This is intentionally a *coarse* detector: we sample, run langdetect on each
post, and return the most common ISO 639-1 code. A multilingual corpus will
collapse to whatever language is most represented — that's the documented
trade-off the caller makes by asking for auto-detection rather than
specifying a language explicitly.

``langdetect`` is an optional dependency installed via the ``[nlp]`` extra.
A missing-dep ImportError is converted into a clear, actionable error that
points the user at the install command.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import polars as pl

from guidedLP.common.exceptions import ValidationError

if TYPE_CHECKING:  # pragma: no cover
    pass


_INSTALL_HINT = (
    "Automatic language detection requires the 'langdetect' package. "
    "Install with: pip install 'guidedLP[nlp]'"
)


def is_available() -> bool:
    """Return True if langdetect is importable in the current environment."""
    try:
        import langdetect  # noqa: F401

        return True
    except ImportError:
        return False


def detect_majority_language(
    df: pl.DataFrame,
    post_col: str,
    sample_size: int = 1000,
    seed: int = 42,
) -> str:
    """
    Detect the most common ISO 639-1 language in a random sample of posts.

    Parameters
    ----------
    df : pl.DataFrame
        Input post table. Only ``post_col`` is read.
    post_col : str
        Name of the string column containing the post text.
    sample_size : int, default 1000
        Maximum number of posts to sample. If the DataFrame has fewer rows,
        all are used. Larger samples give a more reliable estimate at the
        cost of detection time (~1ms per post).
    seed : int, default 42
        RNG seed used both for the sample selection and for langdetect's
        internal randomness — making detection reproducible across runs.

    Returns
    -------
    str
        Detected ISO 639-1 code (e.g. ``"en"``, ``"da"``, ``"de"``).

    Raises
    ------
    ImportError
        If ``langdetect`` is not installed.
    ValidationError
        If the DataFrame is empty, or every sampled post fails detection
        (e.g. all are empty/whitespace/too short for langdetect).
    """
    try:
        import langdetect
    except ImportError as e:  # pragma: no cover
        raise ImportError(_INSTALL_HINT) from e

    langdetect.DetectorFactory.seed = seed

    if df.height == 0:
        raise ValidationError("Cannot detect language from an empty DataFrame")

    n = min(sample_size, df.height)
    sample = (
        df.sample(n=n, seed=seed, with_replacement=False)
        .get_column(post_col)
        .drop_nulls()
        .to_list()
    )

    counts: Counter = Counter()
    for text in sample:
        if not text or not text.strip():
            continue
        try:
            counts[langdetect.detect(text)] += 1
        except langdetect.LangDetectException:
            # Too short, no detectable features — skip
            continue

    if not counts:
        raise ValidationError(
            f"Could not detect language from any of {len(sample)} sampled posts. "
            "Posts may be too short or contain only non-text characters. "
            "Pass an explicit language code (e.g. language='en') to skip detection."
        )

    return counts.most_common(1)[0][0]
