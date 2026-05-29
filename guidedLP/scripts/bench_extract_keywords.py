"""
Synthetic performance benchmark for ``preprocessing.extract_keywords``.

Generates a synthetic ``[sender, post, datetime]`` corpus with a Zipfian
word distribution (mimicking the heavy-tail shape of real social-media
text) and runs three sweeps against it:

  1. ``scale``    -- fix flags, vary n_posts (10k -> 1M)
  2. ``features`` -- fix scale, toggle case_sensitive / min_word_length /
                     keywords= / stop_words / stem / lemmatize
  3. ``output``   -- fix scale + flags, vary output={aggregated,long,lazy}
                     (lazy is sunk to a tmp parquet for a fair end-to-end
                     measurement)

Each run reports wall time + tracemalloc peak + RSS delta + output row
count, matching the style of ``scripts/bench_projection.py``.

Usage
-----
    python3 scripts/bench_extract_keywords.py                      # default sweep
    python3 scripts/bench_extract_keywords.py --mode scale
    python3 scripts/bench_extract_keywords.py --mode features --n-posts 250000
    python3 scripts/bench_extract_keywords.py --mode all --quick   # very small N

Defaults are tuned so the full sweep finishes in well under a minute on
a modern laptop.
"""

from __future__ import annotations

import argparse
import gc
import resource
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guidedLP.preprocessing import extract_keywords  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic corpus generation
# ---------------------------------------------------------------------------

# Compact English mini-vocabulary used as the "content" pool. Includes common
# function words (so stop_words= has real targets), morphologically-marked
# forms (so stem/lemmatize do real work: "running"->"run", "ran"->"run"), and
# enough nouns/verbs/adjectives for posts to look textually plausible. ~200
# tokens is small enough that a Zipfian sampler concentrates on the head but
# large enough that ``vocab_size`` post-filter is meaningful.
_ENGLISH_VOCAB: Tuple[str, ...] = (
    # Function / closed-class (stop-words bucket)
    "the", "a", "an", "of", "and", "or", "but", "if", "then", "so",
    "to", "from", "in", "on", "at", "by", "for", "with", "without",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
    "did", "have", "has", "had", "i", "you", "he", "she", "it", "we",
    "they", "this", "that", "these", "those", "not", "no", "yes", "all",
    "some", "any", "very", "just", "only", "also", "as", "than", "more",
    # Verbs (regular + irregular for stem/lemma to chew on)
    "run", "runs", "running", "ran", "walk", "walking", "walked",
    "talk", "talks", "talking", "talked", "say", "says", "said",
    "go", "goes", "going", "went", "make", "makes", "making", "made",
    "see", "sees", "seeing", "saw", "think", "thinks", "thinking", "thought",
    "know", "knows", "knowing", "knew", "want", "wants", "wanting", "wanted",
    "give", "gives", "giving", "gave", "find", "finds", "finding", "found",
    "use", "uses", "using", "used", "work", "works", "working", "worked",
    # Content nouns
    "climate", "climates", "policy", "policies", "vaccine", "vaccines",
    "election", "elections", "government", "governments", "school", "schools",
    "system", "systems", "country", "countries", "city", "cities", "world",
    "people", "person", "child", "children", "family", "families",
    "study", "studies", "research", "data", "science", "scientist",
    "model", "models", "method", "methods", "result", "results",
    "report", "reports", "story", "stories", "news", "article", "articles",
    # Adjectives / adverbs
    "good", "bad", "new", "old", "big", "small", "high", "low",
    "important", "different", "same", "right", "wrong", "true", "false",
    "early", "late", "fast", "slow", "easy", "hard", "long", "short",
    # Topical noise (so vocab feels broad enough)
    "health", "energy", "economy", "trade", "tax", "taxes", "war", "peace",
    "market", "markets", "price", "prices", "cost", "value", "growth",
    "rate", "rates", "law", "laws", "rule", "rules", "court", "judge",
)


def make_synthetic_posts(
    n_posts: int,
    n_senders: int,
    words_per_post: int,
    vocab_size: int,
    zipf_alpha: float = 1.3,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate a synthetic [sender, post, datetime] table.

    Vocab is drawn from a curated English mini-list (truncated to
    ``vocab_size`` tokens). Word ranks follow Zipf with parameter
    ``zipf_alpha`` so head words are very frequent and the tail is long -
    the same shape real social-media corpora exhibit. Post length is
    Poisson(words_per_post), clipped to >=1.
    """
    rng = np.random.default_rng(seed)

    if vocab_size > len(_ENGLISH_VOCAB):
        # Pad with synthetic tokens so callers can request bigger vocab. These
        # are stem/lemma-inert ("token0001"), which is intentional - we want
        # the NLP-effect comparisons to come from the real-English portion.
        extra = [f"token{i:05d}" for i in range(vocab_size - len(_ENGLISH_VOCAB))]
        vocab = list(_ENGLISH_VOCAB) + extra
    else:
        vocab = list(_ENGLISH_VOCAB[:vocab_size])

    # Zipfian rank weights normalized to a probability vector.
    ranks = np.arange(1, len(vocab) + 1)
    weights = ranks ** (-zipf_alpha)
    weights /= weights.sum()

    # Per-post length, Poisson around the mean, clipped.
    lengths = rng.poisson(words_per_post, size=n_posts).clip(min=1)
    total_tokens = int(lengths.sum())

    # One big sample, then split by per-post length. ~10x faster than per-row
    # sampling because numpy/sampling dispatch is the per-call overhead.
    flat_tokens = rng.choice(len(vocab), size=total_tokens, p=weights)
    vocab_arr = np.array(vocab, dtype=object)
    flat_words = vocab_arr[flat_tokens]

    # Reconstruct posts: split flat array by cumulative lengths, then join
    # with spaces. Doing the join in numpy/Python is fine here because it's
    # one-time corpus setup, not part of the measured path.
    boundaries = np.concatenate(([0], np.cumsum(lengths)))
    posts: List[str] = [
        " ".join(flat_words[boundaries[i] : boundaries[i + 1]].tolist())
        for i in range(n_posts)
    ]

    senders = [f"u_{int(s)}" for s in rng.integers(0, n_senders, size=n_posts)]

    # Synthetic timestamps spaced ~1 minute apart starting at 2024-01-01.
    base = np.datetime64("2024-01-01T00:00:00")
    deltas = np.arange(n_posts).astype("timedelta64[m]")
    dts = (base + deltas).astype("datetime64[us]")

    return pl.DataFrame(
        {
            "sender": senders,
            "post": posts,
            "datetime": pl.Series("datetime", dts).cast(pl.Datetime),
        }
    )


# ---------------------------------------------------------------------------
# Measurement helpers (same shape as scripts/bench_projection.py)
# ---------------------------------------------------------------------------


def _rss_kb() -> int:
    """RSS in kB. ru_maxrss is bytes on macOS, kB on Linux."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw // 1024 if sys.platform == "darwin" else raw


def _measure(
    label: str,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run ``fn`` once, capturing wall time + tracemalloc peak + RSS delta.

    Returns a dict suitable for tabulation.
    """
    gc.collect()
    rss_before = _rss_kb()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _rss_kb()

    # Output-row count, where defined. For a LazyFrame we collect to count -
    # but we already consumed the lazy plan inside fn via sink_parquet, so the
    # `result` returned in that case is the parquet path string.
    if isinstance(result, pl.DataFrame):
        out_rows: Optional[int] = result.height
    elif isinstance(result, pl.LazyFrame):
        out_rows = result.collect().height
    elif isinstance(result, dict) and "rows" in result:
        out_rows = int(result["rows"])
    else:
        out_rows = None

    row = {
        "label": label,
        "wall_s": round(dt, 3),
        "peak_mb": round(peak_bytes / 1e6, 1),
        "rss_delta_mb": round((rss_after - rss_before) / 1024, 1),
        "out_rows": out_rows,
    }
    print(
        f"  {label:<28s} {row['wall_s']:7.2f}s  "
        f"peak {row['peak_mb']:7.1f} MB  RSS Δ {row['rss_delta_mb']:7.1f} MB  "
        f"rows={out_rows if out_rows is not None else '-'}"
    )
    return row


# ---------------------------------------------------------------------------
# Benchmark drivers
# ---------------------------------------------------------------------------


def _call(df: pl.DataFrame, **kw: Any) -> pl.DataFrame:
    """Wrapper that materializes lazy/long/aggregated uniformly for measurement."""
    out = extract_keywords(df, **kw)
    return out if isinstance(out, pl.DataFrame) else out.collect()


def _call_sink(df: pl.DataFrame, **kw: Any) -> Dict[str, Any]:
    """For output='lazy': sink_parquet to a tmpfile, return row count via metadata."""
    lf = extract_keywords(df, output="lazy", **{k: v for k, v in kw.items() if k != "output"})
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as tf:
        path = tf.name
    lf.sink_parquet(path)
    # Cheap row count via the parquet footer (no full materialization).
    rows = pl.scan_parquet(path).select(pl.len()).collect().item()
    Path(path).unlink(missing_ok=True)
    return {"rows": rows, "path": path}


def sweep_scale(args: argparse.Namespace) -> List[Dict[str, Any]]:
    print("\n=== Sweep 1: scale (vary n_posts; default flags) ===")
    sizes = args.scale_sizes
    rows: List[Dict[str, Any]] = []
    for n in sizes:
        df = make_synthetic_posts(
            n_posts=n,
            n_senders=max(1, n // 10),
            words_per_post=args.words_per_post,
            vocab_size=args.vocab_size,
            zipf_alpha=args.zipf_alpha,
            seed=args.seed,
        )
        print(f"\n  corpus: {n:,} posts × ~{args.words_per_post} words = "
              f"~{n * args.words_per_post:,} tokens, vocab={args.vocab_size}")
        rec = _measure(f"n_posts={n:,}", _call, df)
        rec["n_posts"] = n
        rows.append(rec)
        del df
    return rows


def sweep_features(args: argparse.Namespace) -> List[Dict[str, Any]]:
    print("\n=== Sweep 2: features (fix n_posts, toggle one knob at a time) ===")
    n = args.features_n_posts
    df = make_synthetic_posts(
        n_posts=n,
        n_senders=max(1, n // 10),
        words_per_post=args.words_per_post,
        vocab_size=args.vocab_size,
        zipf_alpha=args.zipf_alpha,
        seed=args.seed,
    )
    print(f"  corpus: {n:,} posts × ~{args.words_per_post} words "
          f"= ~{n * args.words_per_post:,} tokens, vocab={args.vocab_size}")

    # Small target vocab so the keywords= filter actually prunes.
    target_kw = list(_ENGLISH_VOCAB[60:80])  # ~20 content words

    variants: List[Tuple[str, Dict[str, Any]]] = [
        ("baseline",                {}),
        ("case_sensitive=True",     {"case_sensitive": True}),
        ("min_word_length=3",       {"min_word_length": 3}),
        ("keywords=20 words",       {"keywords": target_kw}),
        ("stop_words='en'",         {"stop_words": "en"}),
        ("stem='en'",               {"stem": "en"}),
        ("lemmatize='en'",          {"lemmatize": "en"}),
        ("stop+stem+lemma (en)",    {"stop_words": "en", "stem": "en", "lemmatize": "en"}),
    ]
    rows: List[Dict[str, Any]] = []
    for label, kw in variants:
        rec = _measure(label, _call, df, **kw)
        rec["variant"] = label
        rows.append(rec)
    del df
    return rows


def sweep_output(args: argparse.Namespace) -> List[Dict[str, Any]]:
    print("\n=== Sweep 3: output mode (aggregated vs long vs lazy→sink_parquet) ===")
    n = args.features_n_posts
    df = make_synthetic_posts(
        n_posts=n,
        n_senders=max(1, n // 10),
        words_per_post=args.words_per_post,
        vocab_size=args.vocab_size,
        zipf_alpha=args.zipf_alpha,
        seed=args.seed,
    )
    print(f"  corpus: {n:,} posts × ~{args.words_per_post} words")

    rows: List[Dict[str, Any]] = []
    rows.append({**_measure("output=aggregated", _call, df, output="aggregated"), "variant": "aggregated"})
    rows.append({**_measure("output=long",       _call, df, output="long"),       "variant": "long"})
    rows.append({**_measure("output=lazy→parquet", _call_sink, df),               "variant": "lazy_sink"})
    del df
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _print_table(title: str, rows: List[Dict[str, Any]], key: str) -> None:
    if not rows:
        return
    print(f"\n--- {title} ---")
    cols = [key, "wall_s", "peak_mb", "rss_delta_mb", "out_rows"]
    print(pl.DataFrame([{c: r.get(c) for c in cols} for r in rows]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=["scale", "features", "output", "all"],
        default="all",
        help="Which sweep to run.",
    )
    p.add_argument("--quick", action="store_true",
                   help="Use tiny scales so the whole script finishes in seconds.")
    p.add_argument("--n-posts", type=int, default=None,
                   help="Override the per-sweep n_posts. Applies to features+output sweeps "
                        "and replaces the scale sweep with a single point.")
    p.add_argument("--scale-sizes", type=int, nargs="+", default=None,
                   help="Custom list of n_posts values for the scale sweep.")
    p.add_argument("--features-n-posts", type=int, default=100_000,
                   help="n_posts used by the features+output sweeps (default 100k).")
    p.add_argument("--words-per-post", type=int, default=30,
                   help="Mean tokens per post (Poisson).")
    p.add_argument("--vocab-size", type=int, default=len(_ENGLISH_VOCAB),
                   help="Vocab size (extends with synthetic tokens if > built-in list).")
    p.add_argument("--zipf-alpha", type=float, default=1.3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.scale_sizes is None:
        args.scale_sizes = (
            [2_000, 10_000, 50_000] if args.quick
            else [10_000, 50_000, 250_000, 1_000_000]
        )
    if args.quick:
        args.features_n_posts = min(args.features_n_posts, 10_000)

    if args.n_posts is not None:
        args.scale_sizes = [args.n_posts]
        args.features_n_posts = args.n_posts

    print(f"Python {sys.version.split()[0]} | polars {pl.__version__}")
    print(f"mode={args.mode}  quick={args.quick}  "
          f"words_per_post={args.words_per_post}  vocab_size={args.vocab_size}  "
          f"zipf_alpha={args.zipf_alpha}")

    scale_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    output_rows: List[Dict[str, Any]] = []

    if args.mode in ("scale", "all"):
        scale_rows = sweep_scale(args)
    if args.mode in ("features", "all"):
        feature_rows = sweep_features(args)
    if args.mode in ("output", "all"):
        output_rows = sweep_output(args)

    print("\n\n================ Summary ================")
    _print_table("Scale sweep",    scale_rows,   "n_posts")
    _print_table("Feature sweep",  feature_rows, "variant")
    _print_table("Output sweep",   output_rows,  "variant")
    return 0


if __name__ == "__main__":
    sys.exit(main())
