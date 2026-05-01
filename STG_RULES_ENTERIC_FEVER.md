# STG Rules: Acute Febrile Illness / Enteric Fever / Pyrexia of Unknown Origin

## Package Codes
- **MG001A**: Acute Febrile Illness
- **MG006A**: Enteric Fever
- **MG026A**: Pyrexia of Unknown Origin

---

## 0. Document-Layout to Schema Mapping (CRITICAL — added per per-patient review)

The classifier MUST map common Indian-hospital document layouts to the
correct schema doc_type.  Misclassifying these is the dominant source of
mandatory_f1 errors on this package.

| Document layout (filename + content) | Maps to schema field |
| ------------------------------------ | -------------------- |
| **Admission Ticket / Bed-Head Ticket / BHT** (front sheet with patient demographics, chief complaints, history, exam, diagnosis) | `clinical_notes` (NOT `extra_document`) |
| **Daily Progress Notes / Doctor's Notes** (date-stamped daily notes during stay) | `vitals_treatment` (NOT `clinical_notes`) |
| **TPR Chart / Temperature Chart / Vitals Chart** (handwritten tabular grid with temp/pulse/resp values across days) | `vitals_treatment` |
| **Nurse's Notes / Nursing Chart** (medication administration, hourly observations) | `vitals_treatment` |
| **Widal / Typhidot / Salmonella / Blood Culture report** (pre-treatment, day 1-3 of admission) | `investigation_pre` |
| **Repeat Widal / Post-treatment Widal / Follow-up Blood Culture** | `investigation_post` |
| **CBC / ESR / Peripheral Smear / LFT** dated within 24 h of admission | `investigation_pre` |
| **Repeat CBC / ESR / Peripheral Smear / LFT** dated near discharge | `investigation_post` |
| **Discharge Summary / Discharge Card / DC** | `discharge_summary` |
| **Consent forms, pharmacy bills, ID cards, feedback forms, indents** | `extra_document` |

Filename patterns that should override Gemma when present:
- `ticket`, `bht`, `bed_head`, `admission_ticket` → `clinical_notes`
- `vital`, `tpr`, `temp_chart`, `temperature_chart`, `progress_note`,
  `progress_chart`, `nurse_notes` → `vitals_treatment`
- `widal`, `typhi`, `salmonella`, `blood_culture` (when not labelled
  "repeat" or "post") → `investigation_pre`
- `repeat_widal`, `post_widal`, `post_treatment` → `investigation_post`
- `_dis`, `_dc_`, `discharge` → `discharge_summary`

---

## 0.1 Date Nullification Rule (added per per-patient review)

**The schema FORCES `pre_date` and `post_date` on every page row, but the
classifier MUST NOT hallucinate dates to fill the slot.**

Strict rule:
- `pre_date` is non-null **only** on rows where `investigation_pre = 1` AND
  the page itself contains a parseable investigation date.  After population,
  the case-level value is propagated to every row of the same case (eval
  expects this).
- `post_date` follows the same rule for `investigation_post`.
- A date extracted from a page must pass a **plausibility check** (year
  within 5 years of the case folder year).  Dates outside this window are
  treated as null.

The LLM hallucination pattern observed (e.g. "22-07-2022" copied across an
otherwise-2026 case) usually arises when the model "fills" the schema with
a previously-seen date.  The remedy: ignore Gemma's `dates_found` if no
date is physically present on this page's snippet.

---

## 1. Document Classification Rules (Completeness Checks)

### Pre-authorization Phase Documents

| Document Type | Required | Validation Criteria | Missing Action |
|---------------|----------|---------------------|----------------|
| Clinical Notes | ✅ Yes | Must contain detailed patient history | Flag CONDITIONAL |
| Pre-treatment CBC | ✅ Yes | Complete Blood Count before treatment | Flag CONDITIONAL |
| Pre-treatment ESR | ✅ Yes | Erythrocyte Sedimentation Rate before treatment | Flag CONDITIONAL |
| Pre-treatment Peripheral Smear | ✅ Yes | To check for malaria and other conditions | Flag CONDITIONAL |
| Pre-treatment LFT | ✅ Yes | Liver Function Test before treatment | Flag CONDITIONAL |

### Claim Submission Phase Documents

| Document Type | Required | Validation Criteria | Missing Action |
|---------------|----------|---------------------|----------------|
| Indoor Case Papers | ✅ Yes | Records of hospital stay | Flag CONDITIONAL |
| Treatment Details | ✅ Yes | Detailed treatment records | Flag CONDITIONAL |
| Post-treatment CBC | ✅ Yes | Complete Blood Count after treatment | Flag CONDITIONAL |
| Post-treatment ESR | ✅ Yes | Erythrocyte Sedimentation Rate after treatment | Flag CONDITIONAL |
| Post-treatment Peripheral Smear | ✅ Yes | Follow-up peripheral smear | Flag CONDITIONAL |
| Post-treatment LFT | ✅ Yes | Liver Function Test after treatment | Flag CONDITIONAL |
| Discharge Summary | ✅ Yes | Must include date of follow-up | Flag CONDITIONAL |

**Total Required Documents**: 12 (5 pre-authorization + 7 claim submission)

**Critical Note**: Distinguishing between pre-treatment and post-treatment lab reports is mandatory. Both sets must be present.

---

## 2. OCR & Text Extraction Targets

### Core Vital Signs (Mandatory)

```python
VITAL_SIGNS = {
    "temperature": {
        "formats": ["°C", "°F", "celsius", "fahrenheit"],
        "threshold_celsius": 38.3,
        "threshold_fahrenheit": 101.0
    },
    "fever_duration": {
        "unit": "days",
        "minimum": 2
    }
}
```

### Clinical Keywords to Extract

#### AFI Symptom Keywords (from Clinical Notes)
```python
AFI_SYMPTOM_KEYWORDS = [
    "headache",
    "dizziness",
    "pain in muscles",
    "pain in joints",
    "muscle pain",
    "joint pain",
    "weakness",
    "body ache"
]
```

#### Enteric/Typhoid Fever Keywords (from Clinical Notes)
```python
ENTERIC_FEVER_KEYWORDS = [
    "malaise",
    "constipation",
    "diarrhea",
    "rose-colored spots",
    "rose spots",
    "enlarged spleen",
    "splenomegaly",
    "enlarged liver",
    "hepatomegaly"
]
```

#### Exclusion Keywords (Require Different Protocol)
```python
EXCLUSION_KEYWORDS = [
    "organ dysfunction",
    "sepsis",
    "septic shock",
    "multi-organ failure",
    "organ failure"
]
```

### Lab Report Extraction

#### Pre-treatment Lab Reports
```python
PRE_TREATMENT_LABS = {
    "CBC": ["hemoglobin", "WBC", "platelet", "RBC"],
    "ESR": ["erythrocyte sedimentation rate", "ESR value"],
    "Peripheral_Smear": ["malaria", "parasites", "blood film"],
    "LFT": ["bilirubin", "SGOT", "SGPT", "alkaline phosphatase"]
}
```

#### Post-treatment Lab Reports
```python
POST_TREATMENT_LABS = {
    "CBC": ["hemoglobin", "WBC", "platelet", "RBC"],
    "ESR": ["erythrocyte sedimentation rate", "ESR value"],
    "Peripheral_Smear": ["malaria", "parasites", "blood film"],
    "LFT": ["bilirubin", "SGOT", "SGPT", "alkaline phosphatase"]
}
```

### Date Extraction
- Admission Date
- Discharge Date
- Follow-up Date (from Discharge Summary)
- Calculate: Length of Stay (LOS) = Discharge Date - Admission Date

---

## 3. Computer Vision Rules (Visual Evidence)

### No Photographic Evidence Required

**Important**: Unlike surgical packages (Cholecystectomy, Knee Replacement), the Enteric Fever STG does NOT mandate:
- ❌ Intra-operative photographs
- ❌ X-rays
- ❌ Implant barcodes
- ❌ Specimen pictures

```python
def check_visual_evidence_enteric_fever(package_code):
    """
    No visual evidence required for Enteric Fever packages
    """
    if package_code in ["MG001A", "MG006A", "MG026A"]:
        return {
            "visual_check_required": False,
            "reason": "Enteric Fever STG does not mandate photographic evidence"
        }
```

**Pipeline Action**: Bypass object detection and photo-matching steps for these package codes. Rely entirely on document classification and OCR of lab reports.

---

## 4. TMS Rules Engine (Pass/Fail Logic Gates)

### Rule 1: Fever Threshold & Duration
**Requirement**: Patient must have suffered from high fever for extended period

```python
def check_fever_threshold_duration(temperature, temp_unit, fever_duration_days):
    """
    TMS Rule 1: Fever Threshold & Duration
    """
    # Convert to Celsius if needed
    if temp_unit.upper() in ["F", "FAHRENHEIT", "°F"]:
        temp_celsius = (temperature - 32) * 5/9
    else:
        temp_celsius = temperature
    
    # Check thresholds
    fever_high_enough = (temp_celsius >= 38.3) or (temperature >= 101.0 and temp_unit.upper() in ["F", "FAHRENHEIT", "°F"])
    duration_sufficient = fever_duration_days > 2
    
    if fever_high_enough and duration_sufficient:
        return {
            "status": "PASS",
            "rule": "TMS_RULE_1_FEVER",
            "reason": f"Fever {temperature}{temp_unit} for {fever_duration_days} days meets criteria"
        }
    else:
        reasons = []
        if not fever_high_enough:
            reasons.append(f"Temperature {temperature}{temp_unit} below threshold (38.3°C/101°F)")
        if not duration_sufficient:
            reasons.append(f"Duration {fever_duration_days} days insufficient (must be >2 days)")
        
        return {
            "status": "FAIL",
            "rule": "TMS_RULE_1_FEVER",
            "reason": "Fever severity and duration do not meet mandatory clinical criteria. " + "; ".join(reasons)
        }
```

**Logic**: `IF (Temperature >= 38.3°C OR Temperature >= 101°F) AND (Fever_Duration > 2 days) == TRUE → PASS`

**Failure**: Output FAIL (Reason: Fever severity and duration do not meet mandatory clinical criteria for admission)

**Examples**:
- ✅ PASS: 102°F for 3 days
- ✅ PASS: 39°C for 4 days
- ❌ FAIL: 99°F for 3 days (temperature too low)
- ❌ FAIL: 102°F for 1 day (duration too short)

---

### Rule 2: Lab Report Verification (Pre vs. Post)
**Requirement**: Hospital must have conducted both initial and follow-up blood work

```python
def check_lab_report_verification(classified_documents):
    """
    TMS Rule 2: Lab Report Verification (Pre vs. Post)
    """
    # Required pre-treatment labs
    pre_treatment_required = [
        "Pre-treatment CBC",
        "Pre-treatment ESR",
        "Pre-treatment Peripheral Smear",
        "Pre-treatment LFT"
    ]
    
    # Required post-treatment labs
    post_treatment_required = [
        "Post-treatment CBC",
        "Post-treatment ESR",
        "Post-treatment Peripheral Smear",
        "Post-treatment LFT"
    ]
    
    # Check pre-treatment labs
    missing_pre = [doc for doc in pre_treatment_required if doc not in classified_documents]
    
    # Check post-treatment labs
    missing_post = [doc for doc in post_treatment_required if doc not in classified_documents]
    
    if not missing_pre and not missing_post:
        return {
            "status": "PASS",
            "rule": "TMS_RULE_2_LAB_VERIFICATION",
            "reason": "Both pre-treatment and post-treatment lab evaluations present"
        }
    elif missing_post:
        return {
            "status": "CONDITIONAL",
            "rule": "TMS_RULE_2_LAB_VERIFICATION",
            "reason": f"Mandatory post-treatment lab evaluations missing: {', '.join(missing_post)}"
        }
    elif missing_pre:
        return {
            "status": "CONDITIONAL",
            "rule": "TMS_RULE_2_LAB_VERIFICATION",
            "reason": f"Mandatory pre-treatment lab evaluations missing: {', '.join(missing_pre)}"
        }
```

**Logic**: `IF Document_Classifier detects BOTH (Pre_Auth_CBC_ESR_LFT) AND (Claim_Post_Treatment_CBC_ESR_LFT) == TRUE → PASS`

**Failure**: Output CONDITIONAL (Reason: Mandatory post-treatment lab evaluations are missing)

---

### Rule 3: Exclusion Criteria Check
**Requirement**: Patient must NOT have organ dysfunction or sepsis (requires different protocol)

```python
def check_exclusion_criteria(clinical_notes, treatment_details):
    """
    TMS Rule 3: Exclusion Criteria Check
    """
    exclusion_keywords = [
        "organ dysfunction",
        "sepsis",
        "septic shock",
        "multi-organ failure",
        "organ failure"
    ]
    
    combined_text = f"{clinical_notes} {treatment_details}".lower()
    
    found_exclusions = [kw for kw in exclusion_keywords if kw.lower() in combined_text]
    
    if found_exclusions:
        return {
            "status": "CONDITIONAL",
            "rule": "TMS_RULE_3_EXCLUSION",
            "reason": f"Patient shows signs requiring different management protocol: {', '.join(found_exclusions)}. Manual review required."
        }
    else:
        return {
            "status": "PASS",
            "rule": "TMS_RULE_3_EXCLUSION",
            "reason": "No exclusion criteria found"
        }
```

**Logic**: `IF Clinical_Notes CONTAINS ("organ dysfunction" OR "sepsis") == FALSE → PASS`

**Note**: Finding exclusion keywords doesn't mean FAIL, but triggers CONDITIONAL for manual review as patient may need different management protocol.

---

### Rule 4: Chronology / ALOS Validation
**Requirement**: Length of stay must be within approved limits

```python
def check_alos_validation_enteric(admission_date, discharge_date):
    """
    TMS Rule 4: Average Length of Stay (ALOS) Validation for Enteric Fever
    """
    from datetime import datetime
    
    # Calculate length of stay
    los = (discharge_date - admission_date).days
    
    # ALOS for Enteric Fever: 3-5 days
    min_los = 3
    max_los = 5
    
    if los <= max_los:
        return {
            "status": "PASS",
            "rule": "TMS_RULE_4_ALOS",
            "reason": f"Length of stay ({los} days) within ALOS limit (3-5 days)"
        }
    else:
        return {
            "status": "CONDITIONAL",
            "rule": "TMS_RULE_4_ALOS",
            "reason": f"Length of stay ({los} days) exceeds standard ALOS of 3-5 days, requires review"
        }
```

**ALOS Limits**:
- **Enteric Fever (MG001A/MG006A/MG026A)**: 3-5 days

**Logic**: `IF (Discharge_Date - Admission_Date) <= 5 days → PASS`

**Failure**: Output CONDITIONAL (Reason: Length of stay exceeds standard ALOS of 3-5 days, requires review)

---

## 5. Final Decision Logic

### Decision Tree

```python
def make_final_decision_enteric(document_checks, tms_rules):
    """
    Final decision logic for Enteric Fever claims
    """
    # Check document completeness
    missing_docs = [doc for doc, present in document_checks.items() if not present]
    
    if missing_docs:
        return {
            "decision": "CONDITIONAL",
            "confidence": 0.7,
            "reasons": [f"Missing document: {doc}" for doc in missing_docs],
            "missing_documents": missing_docs
        }
    
    # Check TMS rules
    failed_rules = [rule for rule in tms_rules if rule["status"] == "FAIL"]
    conditional_rules = [rule for rule in tms_rules if rule["status"] == "CONDITIONAL"]
    
    if failed_rules:
        return {
            "decision": "FAIL",
            "confidence": 0.9,
            "reasons": [rule["reason"] for rule in failed_rules],
            "rule_flags": [rule["rule"] for rule in failed_rules]
        }
    
    if conditional_rules:
        return {
            "decision": "CONDITIONAL",
            "confidence": 0.8,
            "reasons": [rule["reason"] for rule in conditional_rules],
            "rule_flags": [rule["rule"] for rule in conditional_rules]
        }
    
    # All checks passed
    return {
        "decision": "PASS",
        "confidence": 0.95,
        "reasons": [
            "All mandatory documents present (pre and post-treatment)",
            "Fever criteria met (≥101°F/38.3°C for >2 days)",
            "All TMS rules passed"
        ],
        "rule_flags": []
    }
```

### Pass Criteria
✅ All 12 required documents present (5 pre-auth + 7 claim)
✅ Both pre-treatment AND post-treatment lab reports present
✅ TMS Rule 1 (Fever) = PASS (≥101°F/38.3°C for >2 days)
✅ TMS Rule 2 (Lab Verification) = PASS
✅ TMS Rule 3 (Exclusion) = PASS (no organ dysfunction/sepsis)
✅ TMS Rule 4 (ALOS) = PASS (≤5 days)

**Result**: Output PASS

### Conditional Criteria
⚠️ Any required document missing
⚠️ Post-treatment lab reports missing
⚠️ Pre-treatment lab reports missing
⚠️ TMS Rule 3 (Exclusion) = CONDITIONAL (organ dysfunction/sepsis found)
⚠️ TMS Rule 4 (ALOS) exceeds 5 days

**Result**: Output CONDITIONAL (Requires manual review)

### Fail Criteria
❌ TMS Rule 1 (Fever) = FAIL (Temperature <101°F/<38.3°C OR Duration ≤2 days)

**Result**: Output FAIL

---

## 6. Implementation Checklist

### Document Classification Module
- [ ] Implement classifier for 12 document types
- [ ] Distinguish between pre-treatment and post-treatment lab reports
- [ ] Track missing documents separately (pre vs. post)
- [ ] Generate CONDITIONAL flag for missing docs

### OCR & NLP Module
- [ ] Extract temperature values (handle °C and °F formats)
- [ ] Extract fever duration (in days)
- [ ] Extract AFI symptom keywords
- [ ] Extract Enteric fever keywords
- [ ] Scan for exclusion keywords (organ dysfunction, sepsis)
- [ ] Extract admission and discharge dates
- [ ] Extract follow-up date from discharge summary
- [ ] Calculate length of stay

### Computer Vision Module
- [ ] Bypass visual evidence checks for MG001A/MG006A/MG026A
- [ ] No object detection required
- [ ] No photo matching required

### Rules Engine Module
- [ ] Implement TMS Rule 1 (Fever Threshold & Duration)
- [ ] Implement TMS Rule 2 (Lab Report Verification - Pre vs. Post)
- [ ] Implement TMS Rule 3 (Exclusion Criteria Check)
- [ ] Implement TMS Rule 4 (ALOS Validation)

### Decision Module
- [ ] Implement final decision logic
- [ ] Generate confidence scores
- [ ] Generate reason strings
- [ ] Track rule flags
- [ ] List missing documents (distinguish pre vs. post)

---

## 7. Evidence Provenance Tracking

For each rule check, track:
- **Source Document**: Which document was analyzed
- **Page Number**: Where the evidence was found
- **Bounding Box**: Coordinates of relevant text
- **Confidence Score**: Model confidence in extraction
- **Extracted Text**: Actual text snippet used for rule

```python
evidence_template = {
    "rule": "TMS_RULE_1_FEVER",
    "status": "PASS",
    "source_document": "Clinical_Notes.pdf",
    "page_number": 2,
    "bounding_box": [150, 300, 450, 350],
    "confidence": 0.89,
    "extracted_text": "Patient presented with fever of 102°F for 4 days",
    "extracted_values": {
        "temperature": 102,
        "temperature_unit": "°F",
        "duration": 4,
        "duration_unit": "days"
    },
    "timestamp": "2026-04-25T10:30:00Z"
}
```

---

## 8. Special Considerations

### Temperature Extraction Challenges

```python
def extract_temperature(text):
    """
    Extract temperature with various formats
    """
    import re
    
    # Patterns to match
    patterns = [
        r'(\d+\.?\d*)\s*°?[CF]',  # 102°F, 38.5C
        r'temperature[:\s]+(\d+\.?\d*)',  # temperature: 102
        r'fever[:\s]+(\d+\.?\d*)',  # fever: 102
        r'temp[:\s]+(\d+\.?\d*)'  # temp: 102
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            temp_value = float(match.group(1))
            
            # Determine unit
            if '°F' in text or 'fahrenheit' in text.lower():
                return {"value": temp_value, "unit": "°F"}
            elif '°C' in text or 'celsius' in text.lower():
                return {"value": temp_value, "unit": "°C"}
            else:
                # Guess based on value
                if temp_value > 50:
                    return {"value": temp_value, "unit": "°F"}
                else:
                    return {"value": temp_value, "unit": "°C"}
    
    return None
```

### Duration Extraction

```python
def extract_fever_duration(text):
    """
    Extract fever duration in days
    """
    import re
    
    patterns = [
        r'(\d+)\s*days?',
        r'for\s+(\d+)\s*days?',
        r'duration[:\s]+(\d+)\s*days?',
        r'since\s+(\d+)\s*days?'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return None
```

### Pre vs. Post Treatment Lab Distinction

```python
def classify_lab_timing(document_text, document_metadata):
    """
    Distinguish between pre-treatment and post-treatment labs
    """
    pre_keywords = [
        "pre-treatment",
        "pre treatment",
        "initial",
        "admission",
        "baseline",
        "before treatment"
    ]
    
    post_keywords = [
        "post-treatment",
        "post treatment",
        "follow-up",
        "discharge",
        "after treatment",
        "repeat"
    ]
    
    text_lower = document_text.lower()
    
    # Check for explicit timing keywords
    if any(kw in text_lower for kw in pre_keywords):
        return "PRE_TREATMENT"
    elif any(kw in text_lower for kw in post_keywords):
        return "POST_TREATMENT"
    
    # Use document date vs. admission/discharge dates
    if document_metadata.get("date"):
        doc_date = document_metadata["date"]
        admission_date = document_metadata.get("admission_date")
        discharge_date = document_metadata.get("discharge_date")
        
        if admission_date and doc_date <= admission_date + timedelta(days=1):
            return "PRE_TREATMENT"
        elif discharge_date and doc_date >= discharge_date - timedelta(days=1):
            return "POST_TREATMENT"
    
    return "UNKNOWN"
```

---

## 9. Package-Specific Differences

| Package Code | Package Name | Key Differences |
|--------------|--------------|-----------------|
| MG001A | Acute Febrile Illness | General febrile illness, broader symptom set |
| MG006A | Enteric Fever | Specific to typhoid/enteric fever, look for enteric-specific symptoms |
| MG026A | Pyrexia of Unknown Origin | Fever without clear cause, may have fewer specific symptoms |

**Note**: All three packages follow the same document requirements and TMS rules. The main difference is in the clinical presentation keywords to extract.

---

**Document**: STG_RULES_ENTERIC_FEVER.md
**Package Codes**: MG001A, MG006A, MG026A
**Created**: April 25, 2026
**Status**: Complete Technical Specification
