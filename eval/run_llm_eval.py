"""Measure the model tier: how well it does, how often it is actually called.

Two questions, one of which can be answered offline and one of which cannot.

  How often does the model get called at all?  Answerable here, and it is the
  scalability claim: diagnosis is a low-cardinality function, so cost scales
  with the gateway's vocabulary rather than with traffic.

  How accurate is the real model?  Not answerable without an API key. This
  script runs it when one is present and says so plainly when it is not, rather
  than quietly reporting the stand-in's numbers under the model's name.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reclaim.diagnose import (CachedDiagnoser, ClaudeDiagnoser, MockLLMDiagnoser,
                              RulesDiagnoser, TieredDiagnoser)
from reclaim.models import RootCause
from run_eval import load  # noqa: E402


def cache_report(rows) -> None:
    rules = RulesDiagnoser()
    reached_model = sum(1 for p in rows
                        if rules.diagnose(p).cause is RootCause.UNKNOWN)
    with_notes = sum(1 for p in rows if p.merchant_note)

    print("CALL VOLUME")
    print(f"  payments processed          {len(rows):,}")
    print(f"  answered by the rules table {len(rows) - reached_model:,} "
          f"({1 - reached_model/len(rows):.0%}, no model involved)")
    print(f"  carrying a merchant note    {with_notes:,} ({with_notes/len(rows):.0%})\n")

    print(f"{'cache key':>12}{'distinct':>11}{'calls':>9}{'hits':>8}{'hit rate':>11}")
    print("  " + "-" * 49)
    caches = {}
    for strategy in ("strict", "note"):
        c = CachedDiagnoser(TieredDiagnoser(), key_strategy=strategy)
        for p in rows:
            c.diagnose(p)
        caches[strategy] = c
        print(f"{strategy:>12}{len(c._cache):>11,}{c.calls:>9,}{c.hits:>8,}"
              f"{c.hit_rate:>11.1%}")
    print("  " + "-" * 49)
    print(f"""
  The strict key was the first thing measured and it came in at
  {caches['strict'].hit_rate:.1%}, not the 99.98% the projection implied -- because {with_notes/len(rows):.0%} of these
  payments carry a note and every one of them bypassed the cache. The
  projection was a bound on note-free traffic being quoted as if it were the
  rate. Folding the normalised note into the key fixes it honestly, since
  merchant notes are templated and identical text really does recur.""")

    cached = caches["note"]
    sigs = len(cached._cache)
    print(f"\n  The signature space is bounded by the gateway's vocabulary and the")
    print(f"  merchant's note habits, neither of which grows with traffic. Projected")
    print(f"  from the {sigs} distinct signatures in this set:\n")
    print(f"{'payments':>14}{'model calls':>14}{'hit rate':>12}")
    print("  " + "-" * 38)
    for n in (10_000, 1_000_000, 50_000_000):
        print(f"{n:>14,}{sigs:>14,}{1 - sigs/n:>11.4%}")
    print("""
  This projection assumes note text keeps recurring rather than being freshly
  written every time. A merchant whose staff type genuinely unique prose on
  every failure would approach one call per payment, and the honest thing to do
  there is measure it rather than assume this curve.""")


def accuracy_report(rows) -> None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    rules = RulesDiagnoser()
    novel = [p for p in rows if rules.diagnose(p).cause is RootCause.UNKNOWN]

    print(f"\nACCURACY ON THE {len(novel)} ROWS THE RULES TABLE CANNOT ANSWER")

    mock = MockLLMDiagnoser()
    mock_right = sum(1 for p in novel if mock.diagnose(p).cause is p.true_root_cause)
    print(f"  offline stand-in   {mock_right}/{len(novel)} = {mock_right/len(novel):.3f}")

    if not key:
        print("""
  ClaudeDiagnoser was NOT run: ANTHROPIC_API_KEY is not set.

  Every accuracy figure in this repo is therefore the stand-in's, and the
  stand-in is a keyword matcher tuned against the same vocabulary the generator
  uses. It is a lower bound on plumbing correctness and NOT evidence about the
  model. Reporting it as though it were would be the single easiest way to
  mislead a reader of this repo.

  To measure the real thing:

      pip install anthropic
      export ANTHROPIC_API_KEY=...
      python3 eval/run_llm_eval.py

  Cost, from the call volume above: one call per distinct signature.""")
        return

    print("  running ClaudeDiagnoser...")
    claude = ClaudeDiagnoser()
    cached = CachedDiagnoser(claude)
    right = agree = fallbacks = 0
    for p in novel:
        d = cached.diagnose(p)
        right += d.cause is p.true_root_cause
        agree += d.cause is mock.diagnose(p).cause
        fallbacks += d.source == "llm_fallback"
    print(f"  ClaudeDiagnoser    {right}/{len(novel)} = {right/len(novel):.3f}")
    print(f"  agreement with the stand-in: {agree}/{len(novel)} = {agree/len(novel):.1%}")
    print(f"  calls actually made: {cached.calls} (fallbacks: {fallbacks})")


def main() -> None:
    rows = load("train") + load("test")
    print(f"Model tier -- {len(rows):,} payments\n")
    cache_report(rows)
    accuracy_report(load("test"))


if __name__ == "__main__":
    main()
