"""
Stop-word list resolution.

Thin wrapper around the ``stopwordsiso`` package which ships ISO 639-1-keyed
stop-word lists for 60+ languages. Centralizing the lookup here means the
extractor doesn't need to know whether the user supplied a language code, a
custom Python iterable, or asked for auto-detection — it just calls
``resolve_stop_words`` and gets back a frozenset.

``stopwordsiso`` is part of the ``[nlp]`` extra; missing-dep ImportError is
converted to a clear actionable message.
"""

from __future__ import annotations

from typing import Iterable, FrozenSet, Union

from guidedLP.common.exceptions import ValidationError


_INSTALL_HINT = (
    "Stop-word filtering requires the 'stopwordsiso' package. "
    "Install with: pip install 'guidedLP[nlp]'"
)


def is_available() -> bool:
    """Return True if stopwordsiso is importable in the current environment."""
    try:
        import stopwordsiso  # noqa: F401

        return True
    except ImportError:
        return False


def get_stop_words(language: str) -> FrozenSet[str]:
    """
    Return a frozenset of stop words for the given ISO 639-1 language code.

    Parameters
    ----------
    language : str
        ISO 639-1 code such as ``"en"``, ``"da"``, ``"de"``. Case-insensitive
        on input — normalized to lowercase before lookup.

    Returns
    -------
    FrozenSet[str]
        Lowercased stop words. Frozenset so it's safe to share across calls
        without worrying about accidental mutation.

    Raises
    ------
    ImportError
        If ``stopwordsiso`` is not installed.
    ValidationError
        If the language code is not supported by stopwordsiso.
    """
    try:
        import stopwordsiso
    except ImportError as e:  # pragma: no cover
        raise ImportError(_INSTALL_HINT) from e

    code = language.lower()
    if not stopwordsiso.has_lang(code):
        raise ValidationError(
            f"No built-in stop-word list for language '{language}'. "
            "Supported codes are ISO 639-1 (e.g. 'en', 'da', 'de'). "
            "Pass a custom list via stop_words=[...] for unsupported languages.",
            field="language",
            value=language,
        )
    # stopwordsiso.stopwords returns a set of already-lowercase tokens
    return frozenset(stopwordsiso.stopwords(code))


def resolve_stop_words(
    stop_words: Union[bool, str, Iterable[str], None],
    language: str,
) -> FrozenSet[str]:
    """
    Resolve the user's ``stop_words`` argument into a concrete set.

    Encapsulates the four supported shapes:

    - ``False``/``None`` → empty set (caller should skip filtering).
    - ``True``           → built-in list for ``language`` (which is the
                           auto-detected or caller-specified code).
    - ``str``            → built-in list for that ISO 639-1 code.
    - ``Iterable[str]``  → exactly those tokens, lowercased.

    The ``language`` argument is only consulted in the ``True`` case.

    Returns
    -------
    FrozenSet[str]
        Possibly empty; caller filters using ``.is_in(set)`` or similar.
    """
    if stop_words is None or stop_words is False:
        return frozenset()

    if stop_words is True:
        return get_stop_words(language)

    if isinstance(stop_words, str):
        return get_stop_words(stop_words)

    # Iterable[str] — user-supplied custom list.
    return frozenset(s.lower() for s in stop_words)
