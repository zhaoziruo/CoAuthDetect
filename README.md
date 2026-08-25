# Fine-grained Human–AI Co-authorship Detection — Dataset

Data release for the paper **“Towards Fine-grained and Robust Detection of Human–AI Co-authorship.”**

This repository contains the **AI-generated** and **adversarially paraphrased** texts used in the paper's
experiments, covering three writing domains (news, academic, story), two generators
(GPT-4o and Claude Sonnet 4), three degrees of AI involvement, and two strengths of DIPPER paraphrasing.

Human-written texts are **not redistributed** here (see [Human-written data](#human-written-data)); document
IDs are provided so they can be recovered from the original corpora.

---

## Task formulation

The paper formulates provenance detection as a **4-class** problem over degrees of AI involvement:

| Class | Label in files | How it was produced |
|---|---|---|
| Human-written | *(not included, see below)* | Original corpus text |
| Low AI-involved | `low_ai` | LLM rewrites the source **sentence by sentence**, strict 1-to-1 mapping, surface-level edits only |
| High AI-involved | `high_ai` | LLM heavily rewrites the source, may restructure sentences/paragraphs, facts preserved |
| Fully AI-generated | `fully_ai` | LLM is given only a short prefix/topic and writes the full document |

**Adversarial paraphrasing.** DIPPER (lexical diversity `L=60`, order diversity `O=60`) is applied to every
AI-involved sample once (Level-1) and twice (Level-2). Paraphrases **keep the label of their source text** —
paraphrasing is a post-hoc transformation, not a new authoring process.

---

## Repository layout

```
.
├── arxiv/
│   ├── arxiv_ids.txt             # 204 arXiv IDs used in the experiments
│   ├── arxiv_intro_gpt.json      # GPT-4o generations      (612 records)
│   └── arxiv_intro_claude.json   # Claude Sonnet 4         (612 records)
├── rocstories/
│   ├── story_ids.txt             # 599 ROCStories story IDs
│   ├── story_gpt.json            # 1,797 records
│   └── story_claude.json         # 1,797 records
├── xsum/
│   ├── xsum_ids.txt              # 600 BBC document IDs
│   ├── xsum_gpt.json             # 1,800 records
│   └── xsum_claude.json          # 1,800 records
└── scripts/
    ├── load_dataset.py           # loading / normalising helper + CLI preview
    └── verify_dataset.py         # integrity checks over the release
```

Total: **8,418 records × 3 text fields = 25,254 texts.**

## Record format

Each JSON file is a list of records:

```jsonc
{
  "bbcid": 25265945,            // ID field name is dataset-specific, see table below
  "source_dataset": "xsum",     // "xsum" | "arxiv" | "rocstories"
  "generator": "gpt4o",         // "gpt4o" | "sonnet4"
  "ai_involvement": "low_ai",   // "low_ai" | "high_ai" | "fully_ai"
  "text": "...",                // benign AI-involved text
  "dipper": "...",              // Level-1: one DIPPER pass over `text`
  "dipper_dipper": "..."        // Level-2: two DIPPER passes over `text`
}
```

The document-ID key differs per source corpus:

| Folder | ID field | Example | ID type |
|---|---|---|---|
| `xsum/` | `bbcid` | `25265945` | integer |
| `arxiv/` | `arxivid` | `2011.06485v2` | string |
| `rocstories/` | `storyid` | `1b66b345-30ab-4572-b33d-31422a0217a3` | string |

`scripts/load_dataset.py` normalises all three to a common `doc_id` field.

Every document ID appears exactly **three times per file** — once per `ai_involvement` level — so records can be
grouped by ID to obtain the low/high/fully triplet for the same source document.

## Dataset statistics

| Source | Domain | Documents | Records per generator | Median words (`text`) |
|---|---|---|---|---|
| XSum | News (BBC) | 600 | 1,800 | ~470–700 |
| arXiv | CS paper introductions (2020–2021) | 204 | 612 | ~570–770 |
| ROCStories | Five-sentence stories | 599 | 1,797 | ~66–168 |

Records per (source × generator) split evenly across the three involvement levels.
DIPPER paraphrasing shortens texts slightly (typically 3–20% fewer words than `text`).

## Human-written data

The human-written class is **not redistributed** here for copyright reasons. Use the provided ID lists to
recover it from the original sources:

| Source | Where to get the human texts | ID list |
|---|---|---|
| XSum | https://github.com/EdinburghNLP/XSum | `xsum/xsum_ids.txt` |
| ROCStories | https://cs.rochester.edu/nlp/rocstories/ | `rocstories/story_ids.txt` |
| arXiv | arXiv API (https://info.arxiv.org/help/api/index.html), e.g. https://arxiv.org/abs/2011.06485v2 | `arxiv/arxiv_ids.txt` |

The arXiv subset keeps the **raw LaTeX markup** of the source introductions, because the models were prompted
with raw LaTeX input.

## Not included

- Human-written texts (see above).
- The held-out **GPT-5.1** adversarial evaluation sets described in the paper (§5.3), which are used only for
  robustness evaluation and are not part of this release.
- Model checkpoints and training code.

## Usage

```bash
python scripts/verify_dataset.py          # sanity-check the release
python scripts/load_dataset.py --stats    # per-file counts and length statistics
```

```python
from scripts.load_dataset import load_all, load_split

records = load_all()                              # all 8,418 records, with normalised doc_id
xsum_gpt = load_split("xsum", "gpt4o")            # one source/generator split

# Build a training example at adversarial Level-1
r = xsum_gpt[0]
text, label = r["dipper"], r["ai_involvement"]
```

## Prompts

The prompt templates used for `low_ai`, `high_ai`, and `fully_ai` generation are given in Appendix A of the
paper.

## License

- **Data** (`arxiv/`, `rocstories/`, `xsum/`): [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE).
- **Code** (`scripts/`): [MIT](LICENSE-CODE).

The generated texts are derived from the XSum, ROCStories, and arXiv corpora and from GPT-4o / Claude Sonnet 4
outputs. When using this data, please also respect the terms of the original corpora and of the model
providers, and cite the paper below.

## Citation

> **Note:** the paper is in the camera-ready stage. The entry below is provisional — the full author
> list, venue, pages and DOI will be filled in once the final record is available.

```bibtex
@inproceedings{coauthorship2026,
  title     = {Towards Fine-grained and Robust Detection of Human--AI Co-authorship},
  author    = {Zhao, Ziruo and others},
  booktitle = {TBA},
  year      = {2026}
}
```
