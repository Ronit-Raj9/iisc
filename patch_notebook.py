#!/usr/bin/env python3
"""Patch Ronit.ipynb with scoring improvements.

Changes:
1. Fix LINK_FIELD and PACKAGE_SCHEMAS to use "link" everywhere
2. Improve clinical prompts (marginal values, relaxed criteria)
3. Add package-specific context hints to primary Gemma prompt
4. Fix age_valid clinical prompt for SB039A
"""
import json, shutil, sys
from pathlib import Path

NB_PATH = Path("Ronit.ipynb")
BACKUP = Path("Ronit.ipynb.backup")

# --- Load ---
shutil.copy(NB_PATH, BACKUP)
print(f"Backup saved to {BACKUP}")

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
patched = []


def patch_cell_source(cell_idx, old_str, new_str, label=""):
    """Replace old_str with new_str in the given cell's source."""
    src = "".join(cells[cell_idx]["source"])
    if old_str not in src:
        print(f"  WARNING: Could not find target string for '{label}' in cell {cell_idx}")
        return False
    src = src.replace(old_str, new_str)
    cells[cell_idx]["source"] = [src]
    patched.append(label)
    return True


# ============================================================
# PATCH 1: Fix PACKAGE_SCHEMAS — use "link" for all packages
# ============================================================
print("\n[1] Fixing PACKAGE_SCHEMAS link field...")

patch_cell_source(7,
    '        "case_id", "S3_link/DocumentName", "procedure_code", "page_number",',
    '        "case_id", "link", "procedure_code", "page_number",',
    "SG039C schema link fix")

patch_cell_source(7,
    '        "case_id", "S3_link", "procedure_code", "page_number",',
    '        "case_id", "link", "procedure_code", "page_number",',
    "MG006A schema link fix")

patch_cell_source(7,
    '        "case_id", "s3_link", "procedure_code", "page_number",',
    '        "case_id", "link", "procedure_code", "page_number",',
    "SB039A schema link fix")


# ============================================================
# PATCH 2: Fix LINK_FIELD — use "link" for all packages
# ============================================================
print("[2] Fixing LINK_FIELD...")

patch_cell_source(7,
    '''LINK_FIELD = {
    "MG064A": "link",
    "SG039C": "S3_link/DocumentName",
    "MG006A": "S3_link",
    "SB039A": "s3_link",
}''',
    '''LINK_FIELD = {
    "MG064A": "link",
    "SG039C": "link",
    "MG006A": "link",
    "SB039A": "link",
}''',
    "LINK_FIELD all packages")


# ============================================================
# PATCH 3: Fix KEY_ALIASES — still map old names to "link"
# ============================================================
print("[3] Verifying KEY_ALIASES...")
src7 = "".join(cells[7]["source"])
if '"S3_link": "link"' in src7:
    print("  KEY_ALIASES already maps old names -> 'link'. OK.")


# ============================================================
# PATCH 4: Improve clinical prompts
# ============================================================
print("[4] Improving clinical prompts...")

# MG006A — fever: accept marginal values
patch_cell_source(12,
    '  "fever": <1 if THIS PAGE documents fever, pyrexia, febrile state, OR temperature >=38.3°C/101°F, OR fever duration >2 days, else 0>,',
    '  "fever": <1 if THIS PAGE documents fever, pyrexia, febrile state, raised temperature (even slightly below 38.3°C/101°F — marginal values count as fever), temperature chart entries, OR fever duration mentioned, else 0>,',
    "MG006A fever marginal values")

# MG006A — symptoms: broaden
patch_cell_source(12,
    '  "symptoms": <1 if THIS PAGE documents enteric-fever symptoms (headache, body aches, joint pain, muscle pain, malaise, GI symptoms like vomiting/diarrhea/constipation, abdominal pain, splenomegaly, hepatomegaly, rose spots), else 0>,',
    '  "symptoms": <1 if THIS PAGE documents ANY fever-related symptoms: headache, body aches, joint/muscle pain, malaise, weakness, chills, rigors, GI symptoms (vomiting/diarrhea/constipation/nausea), abdominal pain, splenomegaly, hepatomegaly, rose spots, loss of appetite, generalized weakness, else 0>,',
    "MG006A symptoms broadened")

# SG039C — pain_present: accept any abdominal pain
patch_cell_source(12,
    '  "pain_present": <1 if THIS PAGE documents pain in the right hypochondrium, epigastrium, RUQ, biliary colic, or abdominal pain, else 0>,',
    '  "pain_present": <1 if THIS PAGE documents ANY abdominal pain, right hypochondrium pain, epigastric pain, RUQ pain, biliary colic, upper abdominal discomfort, or pain abdomen (exact anatomical location not required), else 0>,',
    "SG039C pain_present broadened")

# SG039C — clinical_condition: "cholecystitis" alone = assume acute
patch_cell_source(12,
    '  "clinical_condition": <1 if THIS PAGE documents cholelithiasis, cholecystitis, biliary colic, jaundice, gallstones, gall bladder pathology, biliary pancreatitis, or obstructive jaundice, else 0>,',
    '  "clinical_condition": <1 if THIS PAGE documents cholelithiasis, cholecystitis (acute or chronic or unspecified — all count), biliary colic, jaundice, gallstones, gall bladder pathology/disease, biliary pancreatitis, obstructive jaundice, GB calculus, or choledocholithiasis, else 0>,',
    "SG039C clinical_condition broadened")

# SB039A — age_valid: include RA/trauma exceptions
patch_cell_source(12,
    '  "age_valid": <1 if THIS PAGE explicitly documents the patient age and the age is GREATER THAN 55 years, else 0>',
    '  "age_valid": <1 if THIS PAGE documents patient age > 55, OR if patient has rheumatoid arthritis (RA), post-traumatic arthritis, avascular necrosis (AVN), or systemic inflammatory arthritis (age restriction waived for these), else 0>',
    "SB039A age_valid RA/trauma exception")


# ============================================================
# PATCH 5: Add package-specific context to primary Gemma prompt
# ============================================================
print("[5] Adding package-specific context to primary prompt...")

old_prompt_start = 'f"Analyze this single page from a PMJAY claim under package {package_code}.\\n\\n"'
new_prompt_start = '''f"Analyze this single page from a PMJAY claim under package {package_code}.\\n"
        f"{_PACKAGE_CONTEXT.get(package_code, \'\')}\\n\\n"'''

# We need to add the _PACKAGE_CONTEXT dict before the _gemma_user_prompt function
# and modify the prompt to use it
old_before_prompt = '''def _gemma_user_prompt(package_code: str) -> str:
    types = ", ".join(DOC_TYPE_VOCAB)
    return (
        f"Analyze this single page from a PMJAY claim under package {package_code}.\\n\\n"'''

new_before_prompt = '''_PACKAGE_CONTEXT = {
    "MG064A": (
        "PACKAGE MG064A = Severe Anemia (blood transfusion). "
        "Key docs: CBC/Hb lab (pre-treatment showing Hb<7), indoor case paper, "
        "blood transfusion record, post-Hb lab (after transfusion), discharge summary. "
        "cbc_hb_report = pre-transfusion Hb lab. post_hb_report = Hb AFTER transfusion. "
        "treatment_details = medication chart / transfusion record."
    ),
    "SG039C": (
        "PACKAGE SG039C = Laparoscopic Cholecystectomy (gallbladder removal). "
        "Key docs: USG showing gallstones/calculi, LFT report, operative/OT notes, "
        "pre-anaesthesia clearance (PAC), specimen/intraoperative photo, histopathology, discharge summary. "
        "photo_evidence = intraoperative photo or gallstone specimen photo."
    ),
    "MG006A": (
        "PACKAGE MG006A = Enteric Fever (typhoid/febrile illness). "
        "Key docs: pre-treatment investigations (Widal/blood culture/CBC), "
        "vitals/temperature chart (TPR), post-treatment investigations, discharge summary. "
        "investigation_pre = Widal, blood culture, CBC BEFORE treatment. "
        "investigation_post = same test types AFTER treatment (later date). "
        "vitals_treatment = daily vitals chart / TPR / temperature record / treatment chart."
    ),
    "SB039A": (
        "PACKAGE SB039A = Total Knee Replacement (TKR). "
        "Key docs: pre-op knee X-ray/CT, indoor case, OT/operative notes, "
        "implant invoice/sticker/barcode, post-op photo, post-op X-ray with implant visible, discharge summary. "
        "xray_ct_knee = PRE-OP X-ray showing degeneration. post_op_xray = AFTER surgery showing implant in situ. "
        "implant_invoice = invoice/bill/sticker with implant serial number or barcode."
    ),
}


def _gemma_user_prompt(package_code: str) -> str:
    types = ", ".join(DOC_TYPE_VOCAB)
    return (
        f"Analyze this single page from a PMJAY claim under package {package_code}.\\n"
        f"{_PACKAGE_CONTEXT.get(package_code, \'\')}\\n\\n"'''

patch_cell_source(12, old_before_prompt, new_before_prompt, "Package context + prompt")


# ============================================================
# PATCH 6: Increase image resolution for poor-quality docs
# ============================================================
print("[6] Increasing image resolution...")

patch_cell_source(12,
    'MAX_IMAGE_LONG_EDGE = 1024',
    'MAX_IMAGE_LONG_EDGE = 1280  # Increased for better OCR on poor-quality scans',
    "Image resolution increase")

# Also check for PDF_RENDER_ZOOM in config cell
for ci in range(len(cells)):
    src = "".join(cells[ci]["source"])
    if 'PDF_RENDER_ZOOM' in src and '= 1.5' in src:
        patch_cell_source(ci,
            'PDF_RENDER_ZOOM = 1.5',
            'PDF_RENDER_ZOOM = 2.0  # Higher zoom for better text extraction from poor scans',
            "PDF zoom increase")
        break


# ============================================================
# SAVE
# ============================================================
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n{'='*60}")
print(f"DONE! Applied {len(patched)} patches:")
for p in patched:
    print(f"  ✅ {p}")
print(f"\nBackup at: {BACKUP}")
print(f"Patched:   {NB_PATH}")
print(f"\nNEXT STEPS:")
print(f"  1. Delete vlm_clinical_cache/ to re-run clinical prompts")
print(f"  2. Keep vlm_cache/ intact (primary classification cache)")
print(f"  3. Run the notebook with MAX_VLM_CALLS = None")
print(f"  4. Stop sandbox -> Submit eval")
