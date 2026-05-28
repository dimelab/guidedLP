"""
Stemmer and lemmatizer factories.

Both stemming and lemmatization need per-word callables that ``extract_keywords``
can apply via ``map_elements``. This module centralizes:

1. Lazy import of the optional deps (``snowballstemmer``, ``simplemma``) with
   actionable install messages.
2. ISO-639-1 → library-specific language code mapping (snowball uses full
   English names like ``"english"``, simplemma uses ISO codes natively).
3. Returning a uniform ``Callable[[str], str]`` regardless of which library
   sits behind it, so the extraction pipeline doesn't branch on the choice.
"""

from __future__ import annotations

from typing import Callable

from guidedLP.common.exceptions import ValidationError


_STEM_INSTALL_HINT = (
    "Stemming requires the 'snowballstemmer' package. "
    "Install with: pip install 'guidedLP[nlp]'"
)

_LEMMA_INSTALL_HINT = (
    "Lemmatization requires the 'simplemma' package. "
    "Install with: pip install 'guidedLP[nlp]'"
)

# ISO 639-1 → snowballstemmer language name. Snowball doesn't accept ISO codes
# directly so we maintain this mapping ourselves. Languages snowball doesn't
# support (e.g. Polish) raise a ValidationError below.
_SNOWBALL_LANG_MAP = {
    "ar": "arabic",
    "da": "danish",
    "de": "german",
    "en": "english",
    "es": "spanish",
    "fi": "finnish",
    "fr": "french",
    "hu": "hungarian",
    "it": "italian",
    "nl": "dutch",
    "no": "norwegian",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sv": "swedish",
    "tr": "turkish",
}


def is_stem_available() -> bool:
    try:
        import snowballstemmer  # noqa: F401

        return True
    except ImportError:
        return False


def is_lemma_available() -> bool:
    try:
        import simplemma  # noqa: F401

        return True
    except ImportError:
        return False


def make_stemmer(language: str) -> Callable[[str], str]:
    """
    Return a single-word stemmer for ``language`` (ISO 639-1 code).

    The returned callable lowercases its input before stemming because the
    snowball algorithms are case-sensitive in unhelpful ways (``"Climate"``
    would stem to ``"Climat"`` while ``"climate"`` stems to ``"climat"`` —
    we don't want surface-form variants to survive stemming).
    """
    try:
        import snowballstemmer
    except ImportError as e:  # pragma: no cover
        raise ImportError(_STEM_INSTALL_HINT) from e

    code = language.lower()
    if code not in _SNOWBALL_LANG_MAP:
        raise ValidationError(
            f"Snowball has no stemmer for language '{language}'. "
            f"Supported: {sorted(_SNOWBALL_LANG_MAP)}",
            field="language",
            value=language,
        )

    stemmer = snowballstemmer.stemmer(_SNOWBALL_LANG_MAP[code])

    def _stem(word: str) -> str:
        return stemmer.stemWord(word.lower())

    return _stem


def make_lemmatizer(language: str) -> Callable[[str], str]:
    """
    Return a single-word lemmatizer for ``language`` (ISO 639-1 code).

    Simplemma accepts ISO codes directly. It returns the input unchanged when
    a word can't be lemmatized, which is exactly the fall-back behavior we
    want for the extraction pipeline.
    """
    try:
        import simplemma
    except ImportError as e:  # pragma: no cover
        raise ImportError(_LEMMA_INSTALL_HINT) from e

    code = language.lower()
    # simplemma raises on unsupported codes when called; we don't pre-check
    # because its supported-language list expands across versions.

    def _lemmatize(word: str) -> str:
        try:
            return simplemma.lemmatize(word, lang=code)
        except (ValueError, KeyError):  # pragma: no cover
            return word

    return _lemmatize
