"""Integrity checks for the human-AI co-authorship dataset release.

Verifies record counts, the low/high/fully triplet structure, agreement between
the JSON files and the published document-ID lists, and the absence of empty
text fields.

    python scripts/verify_dataset.py
"""

from __future__ import annotations

import sys
from collections import Counter

from load_dataset import FILES, ID_FIELD, LEVELS, TEXT_FIELD, load_ids, load_split

EXPECTED_DOCS = {"xsum": 600, "arxiv": 204, "rocstories": 599}

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        failures.append(message)


def main() -> int:
    for (source, generator) in FILES:
        records = load_split(source, generator)
        label = f"{source}/{generator}"
        print(f"\n{label}: {len(records)} records")

        doc_ids = {r["doc_id"] for r in records}
        expected = EXPECTED_DOCS[source]
        check(len(doc_ids) == expected, f"{label}: {len(doc_ids)} documents (expected {expected})")
        check(len(records) == expected * len(LEVELS),
              f"{label}: {len(records)} records (expected {expected * len(LEVELS)})")

        levels = Counter(r["ai_involvement"] for r in records)
        check(set(levels) == set(LEVELS) and len(set(levels.values())) == 1,
              f"{label}: balanced involvement levels {dict(levels)}")

        pairs = Counter((r["doc_id"], r["ai_involvement"]) for r in records)
        check(all(c == 1 for c in pairs.values()),
              f"{label}: every (doc_id, level) pair is unique")

        gens = {r["generator"] for r in records}
        check(gens == {generator}, f"{label}: generator field is {gens}")
        sources = {r["source_dataset"] for r in records}
        check(sources == {source}, f"{label}: source_dataset field is {sources}")

        missing = [(r["doc_id"], f) for r in records for f in TEXT_FIELD.values()
                   if not r.get(f) or not r[f].strip()]
        check(not missing, f"{label}: no empty text fields ({len(missing)} found)")

        id_field = ID_FIELD[source]
        check(all(id_field in r for r in records), f"{label}: every record carries `{id_field}`")

        published = set(load_ids(source))
        check(doc_ids == published,
              f"{label}: document IDs match {source} ID list "
              f"(+{len(doc_ids - published)} / -{len(published - doc_ids)})")

    print("\n" + ("All checks passed." if not failures else f"{len(failures)} check(s) FAILED."))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
