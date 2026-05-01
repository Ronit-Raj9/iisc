# 🏥 ABPMJAY Auto Adjudication Hackathon — PS1 Complete Intel

> [!CAUTION]
> **Deadline: May 1, 2026, 11:59 PM IST** — Your **latest submission** at this time is your final submission.

---

## 1. Platform & Access

| Item | Detail |
|------|--------|
| **Platform URL** | https://aaehackathon.nhaad.in/ |
| **Discord** | https://discord.gg/jyyeuauG |
| **Client ID / Secret** | Profile → Security and Access → "Reset Credentials" → Downloads a file |
| **Team Registration** | One person registers, shares credentials with teammates. No separate team registration portal. Contact @Hardik to update team details after registration. |
| **STGs Location** | Platform → Resources → Knowledge Hub |
| **Notebook Creation** | Hackathon tab → PS1 → "Open in Sandbox" → Create tab → Create notebook. **Do NOT** create from Profile → Dashboard → My Projects (gives wrong template). |
| **Notebook Name** | `nha_ps1_skeletal_notebook_main.ipynb` — **Do NOT change the prefilled name** |

> [!IMPORTANT]
> **Keep your code in `nha_ps1_skeletal_notebook_main.ipynb`** — Recommended for better scores. This is what the evaluators will run.

---

## 2. Models Allowed

### Provided Models (via NHA API)
You can use **all** provided models — their combined token budget is available to you.

| Model | PS1 Input Tokens | PS1 Output Tokens | PS1 Total Tokens |
|-------|-----------------|-------------------|------------------|
| **Ministral 3B** | 24,000,000 | 1,800,000 | 25,800,000 |
| **Ministral 8B** | 24,000,000 | 1,500,000 | 25,500,000 |
| **Nemotron Nano 30B** | 24,000,000 | 1,500,000 | 25,500,000 |
| **Gemma 3 12B** | 24,000,000 | 1,500,000 | 25,500,000 |
| **Gemma 3 4B** | 24,000,000 | 1,800,000 | 25,800,000 |

> [!WARNING]
> - **No per-day token limits** — only total limits per model.
> - Copy model names **exactly** from the Models tab on the hackathon page. Wrong names give `403 Forbidden`.
> - Must include `metadata={"problem_statement": 1}` in API calls or you get 403.
> - **No GPU access** — everything runs on CPU.

### Custom / External Models
- ✅ **HuggingFace open-source models are allowed** — must run **offline** within the sandbox.
- ✅ Must be trained on **open data**. Provide references in notebook.
- ✅ **Sarvam Vision 3B** — Allowed if it fits and runs in sandbox.
- ✅ **Pytesseract / Python packages** — Allowed as long as they **do NOT make external requests**.
- ❌ **No proprietary models**.
- ❌ **No network calls** during final evaluation — internet will be cut off.
- ❌ **No external API calls** (3rd party).
- Upload model into notebook, install dependencies via terminal (`+` tab → terminal, can use Conda Forge).

> [!CAUTION]
> **Final evaluation will run your notebook with internet cut off. Only the provided NHA models will be allowed. Everything else must run offline/locally.**

---

## 3. Input Data

- **4 Packages**: `MG006A` (Fever), `MG064A` (Anemia), `SB039A` (Knee Replacement), `SG039C` (Cholecystectomy)
- **10 claims per package** → **40 case IDs total**
- **Case ID** = Folder name (usually starts with `PMJAY_`)
- Same case_id repeats for all pages of that case
- Data format: **PDFs** — you must **convert PDF to images (JPG)** before sending to LLM
- Data is accessed via **Databank ID** + **Client ID/Secret**
- **PDF quality is poor** — known issue, deal with it
- **S3Link / link value** = Relative path from Samples folder (use format: `package/claim_id/filename`)

> [!IMPORTANT]
> Hospital documents do NOT come in FHIR/HL7 format — only raw document images/PDFs. You must build OCR/extraction pipeline.

---

## 4. Output Format — CRITICAL

### Folder Structure
```
output/
├── MG006A.json
├── MG064A.json
├── SB039A.json
└── SG039C.json
```

### JSON Structure Rules
- **4 JSON files total** — one per package
- Each JSON = **single array** containing **all 10 claims** for that package
- JSON is at **file + page level** (per page)
- **Key name `"link"`** — Use this for ALL packages (NOT `S3Link`, `S3_link`, `Document`, `s3_link`)
- **Link value** = relative path: `package/claim_id/filename`
- **Key names must match output guidelines exactly** — order can differ
- **Values are binary (0 or 1)** for clinical condition flags
- Put **1** if you see enough evidence of procedure, else **0**
- For clinical conditions (e.g., severe anemia), **evaluate specific clinical criteria** (e.g., Hb < 7 + blood transfusion) before assigning 1 — don't just rely on doctor's diagnosis text
- **Follow output guidelines JSON format** — stick to JSON only, ignore summary tables/timelines in problem statement
- Naming: `<package-name>.json` (e.g., `MG006A.json`)

> [!WARNING]
> The directory name MUST be `output` (NOT `outputs`). Wrong name = evaluation fails silently.

### Sample JSON entry format (from channel):
```json
[
  {
    "case_id": "PMJAY_MN_S_2025_R3_2802202610025071",
    "link": "MG006A/PMJAY_MN_S_2025_R3_2802202610025071/000753__filename.jpg",
    "procedure_code": "MG006A",
    ...
  }
]
```

---

## 5. Document Ranking — CRITICAL

### Core Rules
1. **Rank based on dates of documents** — chronological order (earliest date = Rank 1)
2. **Rank reflects combination of chronological order AND clinical importance**
3. **Per logical document** within a PDF — NOT per physical PDF file
4. If a PDF has multiple documents, group pages by document type → each group gets its own rank
5. **Same rank for all pages of the same logical document**
6. **If multiple pages talk about the same thing → same rank**
7. **A single page can only be ONE document type** (clinical notes OR investigation OR treatment — NOT all together)
8. For the most prominent content on a page, assign that document type's rank

### Ranking Example (from Dr. Shaileja Yadav — official):
> One PDF with 3 docs:
> - Pages 1-3: Clinical Notes (3 Mar) → **Rank 2**
> - Pages 4-5: Discharge Summary (6 Mar) → **Rank 3**
> - Pages 6-7: Investigations (1 Mar) → **Rank 1**

### Rank Map
- The **rank map provided in the sample notebook** is what will be used during final evaluation too.
- Rank scoring uses **Spearman correlation** (from eval logs).
- Tied ranks (same rank for multiple pages) are acceptable when those pages contain the same document type.

---

## 6. Document Classification

- Classify images/pages using the **output guidelines JSON keys only**
- Radiology images (USG, X-Ray, MRI, endoscopy) → classify under the **document category** from output guidelines (e.g., "investigation"), NOT by radiological name
- Nurse notes, consent forms, charts → Treat as **extra documents**
- Duplicate documents in same claim → **Keep one, mark rest as duplicates** (from Hardik's response: "Consider it valid" for marginal gaps)

---

## 7. Clinical Rules & STG Interpretation

### Marginal Gaps
- If patient report is marginally off from STG (e.g., temp 100.8 vs STG requirement >101) → **Consider it VALID**

### Clinical Conditions
- Abdominal pain without exact anatomical location → Use judgment based on context
- "Cholecystitis" without acute/chronic → Assume **acute** if surgery is being performed
- For anemia: Evaluate **specific clinical conditions** (Hb < 7, blood transfusion) before marking 1

### STG Design
- **Create a rule lookup system** using the STG document — this is part of your solutioning
- You can use STG PDF as input to LLM
- Include STG files inside your project folder — evaluators will use whatever extra resources you upload
- **Same 4 STGs will be used for unseen data** in final evaluation
- **Code should be generic and scalable** — NOT hardcoded for just these 4 packages
- **Bonus for package/STG-agnostic pipeline**

> [!IMPORTANT]
> The evaluators will inspect all imports and ensure no unauthorized LLM or proprietary models are used.

---

## 8. Evaluation Process

### Auto Evaluation (40% weight — document classification)
- **Stop your notebook/sandbox FIRST** → then click Evaluate
- Takes **5-10 minutes** — you can close the popup and check results later via "View My Evaluations"
- **Limit: 20 successful submissions total** (increased from 15, then from 5)
- **Highest score** is taken (not latest)
- If evaluation shows error → it does NOT decrement your daily limit

### Manual/Subjective Evaluation (60% weight)
- **Code and approach evaluated after May 1st**
- Leaderboard rankings are NOT final — may change after code review
- Human-readable summary is part of subjective evaluation (store in notebook)
- Scalability, approach robustness, generalization to unseen data all evaluated

### Final Evaluation
- After May 1, 11:59 PM cutoff → sandbox not accessible
- Notebooks run on **same/different datasets** (unseen claims based on same 4 STGs)
- **Internet cut off** — only NHA-provided models work
- All imports inspected for unauthorized models
- **Latest submission at 11:59 PM is final** — but highest auto-eval score is kept for leaderboard

### Common Evaluation Errors
| Error | Cause | Fix |
|-------|-------|-----|
| `missing generated PS1 output file` | Output files not in `output/` folder | Create `output/` folder with 4 JSONs |
| `no participant rows matched labels` | Wrong `case_id` in JSON | Use correct folder names as case_id |
| Stale output | Old cached files | Delete all other output folders, keep only required one |
| `exit status 3` | Missing JSON files | Ensure all 4 package JSONs exist |
| Evaluation runs twice | Known bug | Doesn't affect daily limit |

---

## 9. Final Submission Checklist

- [x] Code in `nha_ps1_skeletal_notebook_main.ipynb`
- [x] `output/` folder with 4 JSON files (`MG006A.json`, `MG064A.json`, `SB039A.json`, `SG039C.json`)
- [x] Each JSON = array of all 10 claims for that package (page-level)
- [x] Use key `"link"` (not S3Link/s3_link/Document)
- [x] Link value = relative path `package/claim_id/filename`
- [x] Key names match output guidelines exactly
- [x] Binary values (0/1) for clinical condition flags
- [x] Document ranking based on dates + clinical importance
- [x] Clean up code, add documentation
- [x] Include README if there are special run instructions
- [x] Human-readable summary stored in notebook (for subjective eval)
- [x] Any custom models uploaded to project folder with references
- [x] No hardcoded solutions — must generalize to unseen data
- [x] No external API calls in code
- [x] Stop sandbox before final evaluation

> [!TIP]
> - You CAN work locally and copy JSONs to sandbox for eval — but must use same models
> - You're free to create your own code structure (don't have to follow skeleton strictly)
> - No frontend needed — just code
> - Pytesseract can be installed from Conda Forge via terminal

---

## 10. Key Moderator Contacts

| Person | Role |
|--------|------|
| **Hardik** | Organizer, domain/output questions, team updates, deadline decisions |
| **Rakshit Ramesh** | Technical moderator, platform issues, evaluation help |
| **Joel_DJ** | Technical support, evaluation debugging |
| **Amarthya** | Moderator, sandbox/notebook issues |
| **Novoneel Chakraborty** | Moderator, technical clarifications |
| **Dr. Shaileja Yadav** | Domain expert (document ranking, clinical guidance) |
| **Vamsi Naik** | Infrastructure/notebook creation issues |

---

## 11. Anti-Disqualification Rules

> [!CAUTION]
> - **Do NOT use multiple accounts** to access LLMs — actively monitored, will be disqualified
> - **Do NOT send concurrent requests** (10+ at once) — blocks other users, will affect final evaluation
> - **Do NOT use unauthorized/proprietary models**
> - **Do NOT make external network requests** in your code
> - All imports will be inspected during final evaluation

---

## 12. Miscellaneous Clarifications

| Topic | Answer |
|-------|--------|
| PASS/FAIL/Conditional in JSON? | Not required in auto-eval JSON. Only for subjective evaluation in notebook. |
| Date format | Not officially clarified. Both `12/03/2005` and `12-04-2004` were asked but no definitive answer. |
| Can we add additional rules beyond STG? | Yes, you can update rules on top of STG rules |
| Score >0.95 possible with STG? | "Yes maybe.." — Rakshit |
| PDF conversion | You must write code to convert PDF → JPG before sending to LLM |
| Can work locally? | Yes, but must use same models. Copy JSONs to sandbox for eval. |
| Token limit exhausted? | Switch models. Or work locally and upload JSONs. |
| 429 Too Many Requests | Rate limiting — add sleep/retry, switch models, don't send concurrent requests |
| rank_score in eval | Based on Spearman correlation of document_rank |
| Credits showing 0? | Don't worry, proceed normally |
| Environment variables | `DATA_ROOT`, `NHA_CLIENT_ID`, `NHA_CLIENT_SECRET` |
| Best score or latest? | **Highest score** is kept on leaderboard |
| Summary PDF submission? | Not separate — keep summary in notebook for subjective eval |
