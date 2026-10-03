# AFAC2026 Financial Intelligent Innovation Contest · Task 2: Complex Financial Document Restoration | 3rd Prize Solution

[中文版](README.md) | English

> **Aliyun Tianchi · AFAC2026 Financial Intelligent Innovation Contest (Challenge Group) · Task 2: Complex Financial Document Restoration**
>
> 🏆 **B-board Score: 93.876479** (table_TEDS **86.11** / reading-order **0.00813**) · **B-board Rank: 8th** · **Final Award: 3rd Prize**
>
> [Competition Link](https://tianchi.aliyun.com/competition/entrance/532490)

## Competition Overview

The financial industry accumulates massive core business documents (insurance clauses, rate tables, cash-value tables, etc.) with non-standard layouts, high content density and complex structure (multi-level headings, nested tables, multi-column layout, cross-page charts). This task requires building a multimodal document-parsing workflow — with **perception, chunking, re-organization, de-duplication and error-correction** capabilities — on top of the organizer-provided multimodal base **FinixDoc-VL**, to automatically parse **ultra-long / ultra-large financial document screenshots** into high-fidelity Markdown strictly corresponding to the original.

- **Output format**: standard CSV (UTF-8), two columns `file_name` / `ground_truth` (Markdown structured string); row count must equal the number of test images.
- **Four goals**: content completeness (100% coverage, no omission/hallucination) / structural fidelity (multi-level headings, lists, standard Markdown tables) / reading order (multi-column, cross-block) / engineering robustness (dense tables, ultra-long documents).
- **Model constraint**: only the organizer's **FinixDoc-VL** API may be called online; local lightweight models (< **10M** params, CPU-only) are allowed; any other third-party LLM API is forbidden (penetration audit).
- **Engineering constraint**: reproduction runtime **≤ 3h**, one-click script, Python + Chinese comments.

## Results

| Metric | Score |
|---|---|
| **overall (B-board)** | **93.876479** |
| table_TEDS | **86.1149** |
| reading_order_Edit_dist | **0.00813** |
| text_block_Edit_dist | **0.036724** |
| B-board rank | **8th** |
| Final award (0.8×B-board + 0.2×final defense) | **3rd Prize** |

### Score Evolution

From the official baseline (histogram-projection striping, ~65 on A-board) to the final 93.88; 8 monotonically-increasing B-board submissions in 3 days:

| Version | overall | table_TEDS | Key change |
|---|---|---|---|
| baseline | ~65 | — | Official idea: histogram-projection striping + concat |
| v1 | 83.24 | 65.05 | Structure-aware splitting foundation |
| v3 | 87.79 | ~72 | Staircase sub-table split; heading-level finalized; two-column gating |
| v4 | 90.67 | 77.99 | Tighter row-completion threshold; misaligned label-row merge |
| v5 | 92.21 | 82.00 | Six fixes for large tables |
| v6 | 92.00 | — | Threshold probe dropped score; rolled back same day |
| v7 | 93.84 | 86.05 | **Banner-header separation & reconstruction** (largest single gain) |
| **v8** | **93.88** | **86.11** | Pure-text degeneration detection & re-split |

## Solution Architecture

### Overall Workflow

```
                        Input image
                           |
          +----------------+----------------
          | table-type                     | long-doc-type
          v                                v
+-- Level1 region split -----+   +-- Long-doc chunking -------+
| page -> text/table regions |   | 2-col detect / TOC whole /  |
+------------+---------------+   | blank-band split            |
+-- Level1.5 subtable split -+   +-------------+---------------+
| staircase by right-reset   |                 |
+------------+---------------+                 |
+-- Level2 table chunking ---+                 |
| row/col analysis -> draw   |                 |
| lines -> rowband x colgrp  |                 |
+------------+---------------+                 |
           v                                v
+------ FinixDoc-VL API concurrent calls (retry / rate-limit / MD5 cache) ------+
           v                                v
+-- Reactive error correction+   +-- Vertical merge ----------+
+-- Banner header reconstruct-+   +-- Heading-level normalize -+
+-- Merge + doc-level postproc+   +-------------+---------------+
           +----------------+-------------------+
                            v
              submission CSV (file_name, ground_truth)
```

### Key Innovations

#### 1. Three-level structure-aware splitting + input modulation
Chunk by **row/column structure** rather than fixed pixels: Level1 region split (text/table separation, fake-table downgrade) → Level1.5 sub-table split (staircase right-edge reset) → Level2 row-band × column-group chunking (each chunk ≤12 rows ×12 cols, side ≤3000px). Input modulation: **draw grid lines** for borderless/semi-bordered tables to supply structural cues; **2x upscale** when short side <80px to prevent hallucination — pushing every chunk into the model's stable working zone.

#### 2. Row/column prior (the cross-cutting foundation)
At chunking time, each chunk's **row/column range is encoded into its filename** (e.g. `chunk_007_r13-24_c13-24`). Merging aligns by the prior; detection knows "returned 10 cols but prior says 12" immediately. Originally for debugging, it became the foundation of all auto-correction.

#### 3. Reactive error correction & re-split (model-agnostic self-healing)
Four symptom detections on API output (over-read / missing-col / missing-row / pure-text degeneration); on trigger, **halve and re-call**, gated by a raggedness score — **accept only if strictly better**. **160 triggers, 0 regressions** on B-board.

#### 4. Banner-header separation & reconstruction
A column-spanning banner header (e.g. "end of policy year" spanning 46 cols) is an empty row in mid column-groups; the model drops empty rows, shifting the whole column-group up by one ("silent error"). Separate the header band from the body and **programmatically reconstruct rowspan/colspan**; 7 banner tables fully aligned, single-version TEDS **+4.04**.

#### 5. Long documents: two-column / cross-page / heading levels
Two-column TOC sent **as a whole block** to preserve reading order + double gating against misjudgment; split points **avoid tables** to keep cross-block tables intact; heading levels **GT-recalibrated** by numbering pattern + dot-depth (capped at L3). Reading-order edit distance **0.008**.

## Tech Stack

| Component | Usage |
|---|---|
| Python 3.11 | Main language |
| FinixDoc-VL API | The only model call (small-block recognition) |
| OpenCV / NumPy | Row/col structure analysis, line drawing, projection |
| Pillow | Image crop / upscale / encoding |
| requests | Concurrent API client (retry / rate-limit / MD5 cache) |

## Repository Structure

```
├── src/
│   ├── pipeline_v2.py        # End-to-end pipeline (table / long-doc dual track)
│   ├── region_splitter.py    # Level1 region split
│   ├── subtable_splitter.py  # Level1.5 sub-table split
│   ├── table_chunker.py      # Level2 table chunking / merge
│   ├── image_chunker.py      # Long-doc chunking
│   ├── merger.py             # Long-doc vertical merge
│   ├── heading_normalizer.py # Heading-level normalization
│   └── api_client.py         # FinixDoc-VL client (only model call + content-MD5 cache)
├── scripts/
│   ├── run_pipeline_v2.py    # Batch entry
│   └── gen_index_html.py     # Intermediate-artifact index page
├── configs/config.py         # Parameters
├── cache/                    # API response cache (keyed by content MD5, 6849 entries)
├── assets/                   # Figures
├── docs/方法描述文档.md       # Full technical report (Chinese)
├── reference_submission_B_v8.csv  # B-board submission (93.876479)
├── run.sh / setup_env.sh / requirements.txt
```

## Data Download

Evaluation data must be downloaded from Tianchi (login required): [competition page](https://tianchi.aliyun.com/competition/entrance/532490) → Data Download. Place as:

```
data/
├── finix_huge_table_rest_B/images/   # B-board table-type, 50 images
└── finix_huge_long_rest_B/images/    # B-board long-doc-type, 50 images
```

> The repo ships `cache/` (all raw FinixDoc-VL responses during the B-board run, keyed by chunk-image **content MD5**). With the data downloaded, the cache enables **byte-identical reproduction** of the B-board submission.

## Quick Start

### Environment

```bash
bash setup_env.sh     # create conda env afac2026 and install deps
```

### Run

```bash
# Mode 1: with cache (default, ~3-10 min, byte-identical reproduction)
bash run.sh

# Mode 2: without cache, all real API calls (needs FinixDoc-VL open, ~1.5 h)
rm -rf cache/ && bash run.sh
```

Outputs: `submission_B.csv` (100 rows) + `output/pipeline_B/index.html` (per-sample visualization).

### Runtime

| Mode | Time | Note |
|---|---|---|
| cache | ~3-10 min | measured 216s; output MD5-identical to B-board submission |
| real API | ~1.5 h | measured 5566s; well under the 3h limit |

## Key Parameters

| Parameter | Value | Note |
|---|---|---|
| Table chunk cap | ≤12 rows ×12 cols, side ≤3000px | avoid length-cap truncation |
| Small-image upscale | short side <80px → 2x | calibrated by full scan (hallucinated samples ≤41px) |
| Fake-table downgrade | height <110px | guard against title-box misjudge |
| Image-level / chunk-level concurrency | 3 / 8 | within official rate limit |
| Correction gate | accept only if strictly better | 160 triggers, 0 regressions |
| Heading-level cap | L3 | recalibrated on 100 training GTs |

## Documentation

Full technical report (problem analysis, key techniques, iteration history, parameters, reflection): [docs/方法描述文档.md](docs/方法描述文档.md) (Chinese).

## Notes

- **Compliance**: the only model call in the whole project is the official FinixDoc-VL API (`api_client.py`); no local model, no prompt engineering (the interface has no prompt parameter), CPU-only.
- **Cache ≠ hardcoding**: each file in `cache/` is one raw API return; its filename is the content MD5 of the chunk image. Deleting the cache makes every call real.
- **Extraction tip**: data images have Chinese filenames; old `unzip` may report `File name too long` — use `LANG=C.UTF-8 unzip xxx.zip` or Python `zipfile`.
- **Memory** ≥ 8GB (largest single table 22928×16223 ≈ 372MP).

## License

This project is for learning and reference only; commercial use is not permitted. Data and evaluation originate from the Tianchi AFAC2026 competition platform.
