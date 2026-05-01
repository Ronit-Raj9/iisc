# IISc Healthcare Hackathon — PS1 Solution (NHA AB PM-JAY Auto-Adjudication)

Per-page Gemma 3 12B classification + STG rule extraction for four PM-JAY package codes:

| Code   | Condition                  |
| ------ | -------------------------- |
| MG064A | Severe Anemia              |
| SG039C | Cholecystectomy            |
| MG006A | Enteric Fever              |
| SB039A | Total Knee Replacement     |

## Best submission so far

| Component        | Score    |
| ---------------- | -------- |
| `mandatory_f1`   | 0.9154   |
| `clinical_f1`    | 0.7909   |
| `extra_f1`       | 0.7684   |
| `rank_score`     | 0.1066   |
| **`final_score`**| **0.7662** |

(Best per-package row counts: MG064A=351, SG039C=331, MG006A=456, SB039A=459.)

### Pipeline iterations & live-eval scores

| # | Change | clinical_f1 | extra_f1 | rank_score | final |
| - | ------ | ----------- | -------- | ---------- | ----- |
| 1 | Per-page evidence + per-page rank from RANK_MAP                       | 0.7838 | 0.7683 | 0.107  | 0.7641 |
| 2 | Pure Gemma override on clinical fields                                | 0.7567 | 0.7683 | 0.106  | 0.7554 |
| 3 | INTERSECTION (kw AND gem) on all clinical fields                      | **0.7909** | 0.7683 | 0.106  | 0.7661 |
| 4 | Hybrid (intersect on diagnosis fields, union on signs)                | 0.7725 | 0.7683 | 0.106  | 0.7604 |
| 5 | INTERSECTION + SG039C-only continuity / admin-extras / kw expansions  | 0.7909 | **0.7684** | **0.1066** | **0.7662** |

INTERSECTION wins — labels are stricter than Gemma; keyword-AND-Gemma filters out keyword false positives. SG039C-only refinements come from manual review of all 10 SG039C cases.

## Repo layout

```
.
├── solution_notebook.ipynb            # Latest end-to-end notebook (upload to NHA jupyter server)
├── best_submission_ps1.ipynb          # Notebook that produced the 0.7641 result
├── best_submission_outputs/           # The 4 JSON files from the 0.7641 submission
│   ├── MG064A.json
│   ├── SG039C.json
│   ├── MG006A.json
│   └── SB039A.json
│
├── HI.txt                             # Problem statement / output spec (organizer)
├── STG_RULES_*.md                     # 4 STG rule references (per package)
├── STG Rules PDF's/                   # Original STG PDFs
├── nha_ps1_skeletal_notebook_main (1).ipynb   # Organizer-provided skeleton
├── full (1).py                        # Reference impl (credentials redacted)
│
├── nha_ps1/                           # Early modular package (pre-notebook approach)
├── scripts/
│   ├── build_solution_notebook.py     # Generator: produces solution_notebook.ipynb from CELL_OVERRIDES
│   ├── test_solution_notebook.py      # Local end-to-end test with mocked NHAclient
│   └── smoke_test.py                  # Unit tests for nha_ps1/ package
└── requirements.txt
```

## How to run on the NHA Jupyter server

1. **Upload `solution_notebook.ipynb`** to the platform.
2. Open the **NHAclient init cell** and paste your `clientId` / `clientSecret`.
3. Run the **databank widget cell** — Databank ID `c110a5f8-6e79-43bd-bd7a-979677354958` — wait for download to finish.
4. Set `MAX_VLM_CALLS = None` in the **CONFIG cell**.
5. Run all cells.

The pipeline:
- auto-detects the dataset under `/home/jovyan/<databank-id>/Claims`
- fires one Gemma call per page for OCR + doc-type + entities (cached at `vlm_cache/`)
- fires a second Gemma call per page for clinical-field assessment (cached at `vlm_clinical_cache/`)
- writes per-package JSON to `output/<PACKAGE>.json`

Subsequent runs hit the caches → 0 fresh tokens → instant.

## Architecture (pipeline stages)

```
PDF/JPG file
  → extract_pages         (PyMuPDF rasterize at zoom=1.5)
  → analyze_page_with_gemma   (cache-1)  → doc_type, ocr_snippet, entities, visual flags
  → analyze_clinical_with_gemma (cache-2) → per-package clinical-field flags
  → detect_visual_elements    (pyzbar QR/barcode + VLM signals)
  → estimate_page_quality     (Laplacian blur)
  → classify_document_type    (VLM-first; keyword fallback)
  → populate_row_for_package  (per-page evidence + clinical override from cache-2)
  → _propagate_case_dates     (pre_date / post_date / doa / dod copied case-wide)
  → assign_document_ranks     (per-page rank from RANK_MAP)
  → _strip_to_schema          (output keys exactly = HI.txt schema)
  → output/<PACKAGE>.json
```

## What we tried (rank scheme experiments)

| Scheme                                       | rank_score | final_score |
| -------------------------------------------- | ---------- | ----------- |
| Per-page from RANK_MAP (typical-order ties)  | **0.107**  | **0.7641**  |
| Global per-page sequential within case       | 0.0999     | 0.7632      |
| File-level sequential by typical_rank        | 0.045      | 0.7557      |
| File-level sequential by filename            | 0.0075     | 0.7515      |

Per-page from RANK_MAP wins.

## Token-budget tracking

| Limit (PS1)      | Used (best run) | Used (clinical re-run, projected) |
| ---------------- | --------------- | --------------------------------- |
| Input  24M       | ~2.1M (8.7%)    | ~4.5M (19%)                       |
| Output 1.5M      | ~690K (46%)     | ~850K (57%)                       |
| Total  25.5M     | ~2.8M (11%)     | ~5.4M (21%)                       |

## Credits

Built during the IISc Healthcare Hackathon, April 2026. Pipeline iterated against the live evaluator. Currently #1 leaderboard position, working on extending the lead.
