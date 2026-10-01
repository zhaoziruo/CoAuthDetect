"""Build train/validation/test CSVs for the 4-class detection task.

This is the bridge between the release and a trainable dataset. The release
ships AI-involved texts only; the human-written class is not redistributed, so
you supply it yourself from the published document IDs (see ``--human`` below
and the "Human-written data" section of the README).

    python scripts/build_splits.py --source xsum --generator gpt4o \
        --human path/to/xsum_human.json --adversarial level1 --out-dir splits/

Writes to ``--out-dir``:

    train.csv, val.csv          at the requested adversarial level
    test-benign.csv             the three evaluation conditions, always all
    test-level1.csv               three, so the robustness table can be filled
    test-level2.csv               from one invocation
    manifest.json               what was built, for the record

Each CSV has two columns, ``text`` and ``label``:

    0 = human      1 = low_ai      2 = high_ai      3 = fully_ai

Adversarial levels follow the paper. A DIPPER paraphrase keeps the label of the
text it was derived from -- paraphrasing is a post-hoc transformation, not a new
authoring process -- so adding a level adds rows without adding classes:

    benign   human + {low,high,fully}@text
    level1   benign + {low,high,fully}@dipper          (one DIPPER pass)
    level2   level1 + {low,high,fully}@dipper_dipper   (two DIPPER passes)

Documents are partitioned *before* being expanded into rows, so no source
document ever appears in more than one split. This is asserted, not assumed: a
document leaking from train into test would let the model match on content and
inflate every number in the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from load_dataset import FILES, LEVELS, TEXT_FIELD, load_split

ROOT = Path(__file__).resolve().parent.parent

#: integer label per class, in increasing order of AI involvement
LABEL = {"human": 0, "low_ai": 1, "high_ai": 2, "fully_ai": 3}

#: which text fields enter *training* at each adversarial level
LEVEL_FIELDS = {
    "benign": ["text"],
    "level1": ["text", "dipper"],
    "level2": ["text", "dipper", "dipper_dipper"],
}

#: paper split proportions, by document
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15

HUMAN_ID_CANDIDATES = ("doc_id", "bbcid", "arxivid", "storyid", "id")
HUMAN_TEXT_CANDIDATES = ("text", "document", "body", "content")


# ---------------------------------------------------------------------------
# human-written texts (not redistributed -- supplied by the user)
# ---------------------------------------------------------------------------

def load_human(path: Path, id_field: str | None, text_field: str | None) -> dict[str, str]:
    """Read human texts as ``{doc_id: text}`` from JSON, JSONL, or CSV.

    Any of these is accepted, as long as one column holds the document ID
    exactly as it appears in the published ID list and another holds the text:

        [{"bbcid": 25265945, "text": "..."}, ...]        .json
        {"bbcid": 25265945, "text": "..."}               .jsonl, one per line
        bbcid,text                                       .csv
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as fh:
            records = list(csv.DictReader(fh))
    elif suffix == ".jsonl":
        with open(path, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
    else:
        with open(path, encoding="utf-8") as fh:
            records = json.load(fh)
        if isinstance(records, dict):  # already {id: text}
            return {str(k): str(v) for k, v in records.items()}

    if not records:
        sys.exit(f"{path} is empty")

    keys = list(records[0])
    id_field = id_field or next((k for k in HUMAN_ID_CANDIDATES if k in keys), None)
    text_field = text_field or next((k for k in HUMAN_TEXT_CANDIDATES if k in keys), None)
    if id_field is None or text_field is None:
        sys.exit(
            f"Could not infer the ID/text columns of {path} (columns: {keys}).\n"
            f"Pass --human-id-field and --human-text-field explicitly."
        )

    return {str(r[id_field]): str(r[text_field]) for r in records if str(r.get(text_field, "")).strip()}


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------

def document_order(records: list[dict], seed: int | None) -> list[str]:
    """Deterministic document ordering.

    Default is order of first appearance in the release file, which is the
    order the paper's splits were cut in. ``--shuffle-seed`` re-partitions
    instead -- useful for a seed sweep, but it will not reproduce the paper.
    """
    seen: list[str] = []
    for record in records:
        if record["doc_id"] not in seen:
            seen.append(record["doc_id"])
    if seed is not None:
        random.Random(seed).shuffle(seen)
    return seen


def partition(doc_ids: list[str]) -> dict[str, set[str]]:
    n = len(doc_ids)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * (TRAIN_FRAC + VAL_FRAC))
    return {
        "train": set(doc_ids[:n_train]),
        "val": set(doc_ids[n_train:n_val]),
        "test": set(doc_ids[n_val:]),
    }


def build_rows(
    records: list[dict],
    docs: set[str],
    human: dict[str, str],
    fields: list[str],
    dipper_ratio: float,
    rng: random.Random,
) -> list[tuple[str, int]]:
    """Expand the documents in `docs` into (text, label) rows.

    One human row per document; for each AI involvement level, one row per
    requested text field.
    """
    rows: list[tuple[str, int]] = []

    for doc_id in sorted(docs):
        text = human.get(doc_id, "").strip()
        if text:
            rows.append((text, LABEL["human"]))

    for level in LEVELS:
        subset = [r for r in records if r["ai_involvement"] == level and r["doc_id"] in docs]
        for field in fields:
            candidates = [(r[field].strip(), LABEL[level]) for r in subset if r[field].strip()]
            if field != "text" and dipper_ratio < 1.0:
                # Ablation: keep only a fraction of the adversarial rows.
                keep = int(len(candidates) * dipper_ratio)
                candidates = rng.sample(candidates, keep) if keep else []
            rows.extend(candidates)

    return rows


def write_csv(path: Path, rows: list[tuple[str, int]], rng: random.Random) -> dict[str, int]:
    rows = rows[:]
    rng.shuffle(rows)  # interleave classes so a batch is not single-label
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for _, label in rows:
        name = next(k for k, v in LABEL.items() if v == label)
        counts[name] = counts.get(name, 0) + 1
    return counts


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", required=True, choices=sorted({s for s, _ in FILES}))
    parser.add_argument("--generator", required=True, choices=sorted({g for _, g in FILES}))
    parser.add_argument("--human", required=True, type=Path,
                        help="Human-written texts keyed by document ID (JSON/JSONL/CSV). "
                             "Not part of the release -- recover it from the published ID "
                             "list; see the README.")
    parser.add_argument("--human-id-field", default=None)
    parser.add_argument("--human-text-field", default=None)
    parser.add_argument("--adversarial", default="benign", choices=list(LEVEL_FIELDS),
                        help="Which adversarial levels to mix into TRAINING. All three "
                             "test conditions are written regardless.")
    parser.add_argument("--dipper-ratio", type=float, default=1.0,
                        help="Ablation: fraction of DIPPER rows to keep in training")
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="Re-partition documents with this seed instead of using the "
                             "release order. Will NOT reproduce the paper's splits.")
    parser.add_argument("--seed", type=int, default=42629309, help="Row shuffling / sampling seed")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if not 0.0 <= args.dipper_ratio <= 1.0:
        sys.exit("--dipper-ratio must be in [0, 1]")

    records = load_split(args.source, args.generator, ROOT)
    human = load_human(args.human, args.human_id_field, args.human_text_field)
    rng = random.Random(args.seed)

    doc_ids = document_order(records, args.shuffle_seed)
    covered = sum(1 for d in doc_ids if d in human)
    print(f"{args.source}/{args.generator}: {len(records)} records over {len(doc_ids)} documents")
    print(f"human texts supplied for {covered}/{len(doc_ids)} documents")
    if covered == 0:
        sys.exit(
            f"None of the document IDs in {args.human} match the release.\n"
            f"  release IDs look like: {doc_ids[:3]}\n"
            f"  supplied IDs look like: {list(human)[:3]}"
        )
    if covered < len(doc_ids):
        missing = len(doc_ids) - covered
        print(f"[warn] {missing} documents have no human text; those documents still "
              f"contribute their AI rows, so the human class will be under-represented. "
              f"Recover the rest before training.", file=sys.stderr)

    parts = partition(doc_ids)
    # A document in two splits would let the model match on content rather than
    # provenance, so this is a hard check rather than a comment.
    assert not (parts["train"] & parts["val"]), "train/val document overlap"
    assert not (parts["train"] & parts["test"]), "train/test document overlap"
    assert not (parts["val"] & parts["test"]), "val/test document overlap"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "source": args.source,
        "generator": args.generator,
        "adversarial_level": args.adversarial,
        "dipper_ratio": args.dipper_ratio,
        "label_map": LABEL,
        "documents": {name: len(ids) for name, ids in parts.items()},
        "human_coverage": f"{covered}/{len(doc_ids)}",
        "shuffle_seed": args.shuffle_seed,
        "seed": args.seed,
        "files": {},
    }

    train_fields = LEVEL_FIELDS[args.adversarial]
    for name in ("train", "val"):
        rows = build_rows(records, parts[name], human, train_fields, args.dipper_ratio, rng)
        counts = write_csv(args.out_dir / f"{name}.csv", rows, rng)
        manifest["files"][f"{name}.csv"] = {"rows": len(rows), "by_class": counts}
        print(f"  {name + '.csv':<18} {len(rows):5d} rows  {counts}")

    # Every test condition, every time: the robustness table in the paper
    # compares one model across all three, so emitting only the matching one
    # would make that table need three runs of this script.
    for condition, field in TEXT_FIELD.items():
        rows = build_rows(records, parts["test"], human, [field], 1.0, rng)
        counts = write_csv(args.out_dir / f"test-{condition}.csv", rows, rng)
        manifest["files"][f"test-{condition}.csv"] = {"rows": len(rows), "by_class": counts}
        print(f"  {'test-' + condition + '.csv':<18} {len(rows):5d} rows  {counts}")

    with open(args.out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nWrote {args.out_dir}/ (train at level '{args.adversarial}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
