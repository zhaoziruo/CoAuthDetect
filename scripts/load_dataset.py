"""Loading helpers for the human-AI co-authorship dataset.

The three source corpora use different document-ID field names (``bbcid``,
``arxivid``, ``storyid``); every loader here adds a normalised ``doc_id`` field
so downstream code can treat the splits uniformly.

    python scripts/load_dataset.py --stats
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ID_FIELD = {"xsum": "bbcid", "arxiv": "arxivid", "rocstories": "storyid"}

FILES = {
    ("xsum", "gpt4o"): "xsum/xsum_gpt.json",
    ("xsum", "sonnet4"): "xsum/xsum_claude.json",
    ("arxiv", "gpt4o"): "arxiv/arxiv_intro_gpt.json",
    ("arxiv", "sonnet4"): "arxiv/arxiv_intro_claude.json",
    ("rocstories", "gpt4o"): "rocstories/story_gpt.json",
    ("rocstories", "sonnet4"): "rocstories/story_claude.json",
}

ID_LIST = {
    "xsum": "xsum/xsum_ids.txt",
    "arxiv": "arxiv/arxiv_ids.txt",
    "rocstories": "rocstories/story_ids.txt",
}

LEVELS = ("low_ai", "high_ai", "fully_ai")

#: text field for each adversarial level
TEXT_FIELD = {"benign": "text", "level1": "dipper", "level2": "dipper_dipper"}


def load_split(source: str, generator: str, root: Path = ROOT) -> list[dict]:
    """Load one (source, generator) split with a normalised ``doc_id`` field."""
    path = root / FILES[(source, generator)]
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    id_field = ID_FIELD[source]
    for record in records:
        record["doc_id"] = str(record[id_field])
    return records


def load_all(root: Path = ROOT) -> list[dict]:
    """Load every split in the release."""
    return [r for source, gen in FILES for r in load_split(source, gen, root)]


def load_ids(source: str, root: Path = ROOT) -> list[str]:
    """Document IDs of the human-written source documents for one corpus."""
    with open(root / ID_LIST[source], encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def as_examples(records: list[dict], level: str = "benign") -> list[tuple[str, str]]:
    """Flatten records into ``(text, label)`` pairs at one adversarial level."""
    field = TEXT_FIELD[level]
    return [(r[field], r["ai_involvement"]) for r in records]


def _stats(root: Path = ROOT) -> None:
    for (source, generator), rel in FILES.items():
        records = load_split(source, generator, root)
        print(f"\n{rel}  ({len(records)} records, {len({r['doc_id'] for r in records})} documents)")
        for level in LEVELS:
            subset = [r for r in records if r["ai_involvement"] == level]
            row = [f"  {level:<9} n={len(subset):<5}"]
            for name, field in TEXT_FIELD.items():
                lengths = [len(r[field].split()) for r in subset]
                row.append(f"{name}: {statistics.mean(lengths):6.1f} mean words")
            print("  |  ".join(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", action="store_true", help="print per-split length statistics")
    args = parser.parse_args()
    if args.stats:
        _stats()
    else:
        records = load_all()
        print(f"{len(records)} records loaded")
        print(json.dumps({k: (v[:200] + "..." if isinstance(v, str) and len(v) > 200 else v)
                          for k, v in records[0].items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
