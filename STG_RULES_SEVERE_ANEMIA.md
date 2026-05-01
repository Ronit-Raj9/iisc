# STG Rules: Severe Anemia

## Package Codes
- **MG064A**: Severe Anemia

---

## 1. Document Classification Rules (Completeness Checks)

### Pre-authorization Phase Documents

| Document Type | Required | Validation Criteria | Missing Action |
|---------------|----------|---------------------|----------------|
| Clinical Notes | ✅ Yes | Must include evaluation findings, indications for procedure, and planned line of treatment | Flag CONDITIONAL |
| Pre-treatment CBC | ✅ Yes | Complete Blood Count before treatment | Flag CONDITIONAL |
| Pre-treatment Hb Report | ✅ Yes | Hemoglobin level before treatment | Flag CONDITIONAL |

### Claim Submission Phase Documents

| Document Type | Required | Validation Criteria | Missing Action |
|---------------|----------|---------------------|----------------|
| Indoor Case Papers | ✅ Yes | Detailed records of hospital stay | Flag CONDITIONAL |
| Treatment Details | ✅ Yes | Must document treatments administered | Flag CONDITIONAL |
| Post-treatment CBC | ✅ Yes | Complete Blood Count after treatment | Flag CONDITIONAL |
| Post-treatment Hb Report | ✅ Yes | Hemoglobin level after treatment to show recovery | Flag CONDITIONAL |
| Discharge Summary | ✅ Yes | Detailed discharge summary | Flag CONDITIONAL |

**Total Required Documents**: 8 (3 pre-authorization + 5 claim submission)

**Critical Note**: Distinguishing between pre-treatment and post-treatment lab reports is mandatory. Both Hb levels must be documented.

---

## 2. OCR & Text Extraction Targets

### Core Quantitative Target (Mandatory)

```python
HEMOGLOBIN_EXTRACTION = {
    "keywords": ["hemoglobin", "Hb", "haemoglobin", "HGB"],
    "unit": "g/dl",
    "severity_threshold": 7.0,  # Must be < 7 g/dl for severe anemia
    "required_phases": ["pre_treatment", "post_treatment"]
}
```

### Treatment Keywords (Mandatory)

```python
MANDATORY_TREATMENT_KEYWORDS = [
    "blood transfusion",
    "ferrous sulphate injection",
    "ferrous sulfate injection",
    "blood transfused",
    "packed red blood cells",
    "PRBC",
    "whole blood transfusion"
]
```

### Clinical Keywords to Extract

#### Common Symptom Keywords (from Clinical Notes)
```python
ANEMIA_SYMPTOM_KEYWORDS = [
    "pallor",
    "pale conjunctivae",
    "pale mucous membranes",
    "fatigue",
    "weakness",
    "dizziness",
    "dyspnea",
    "shortness of breath",
    "tachycardia",
    "rapid heart rate",
    "palpitations"
]
```

#### Severe/Life-threatening Signs (from Clinical Notes)
```python
SEVERE_ANEMIA_SIGNS = [
    "sweating",
    "cold extremities",
    "edema",
    "respiratory distress",
    "shock",
    "bleeding",
    "hemorrhage",
    "hypotension",
    "altered consciousness",
    "syncope"
]
```

### Lab Report Extraction

#### Pre-treatment Lab Values
```python
PRE_TREATMENT_LABS = {
    "CBC": ["hemoglobin", "Hb", "WBC", "platelet", "RBC", "hematocrit"],
    "Hb": ["hemoglobin level", "Hb value", "g/dl", "g/dL"]
}
```

#### Post-treatment Lab Values
```python
POST_TREATMENT_LABS = {
    "CBC": ["hemoglobin", "Hb", "WBC", "platelet", "RBC", "hematocrit"],
    "Hb": ["hemoglobin level", "Hb value", "g/dl", "g/dL"]
}
```

### Date Extraction
- Admission Date
- Discharge Date
- Transfusion Date(s) (from Treatment Details)
- Calculate: Length of Stay (LOS) = Discharge Date - Admission Date

---

## 3. Computer Vision Rules (Visual Evidence)

### No Photographic Evidence Required

**Important**: Similar to Enteric Fever, the Severe Anemia STG does NOT mandate:
- ❌ Intra-operative photographs
- ❌ X-rays
- ❌ Implant barcodes
- ❌ Specimen pictures

```python
def check_visual_evidence_severe_anemia(package_code):
    """
    No visual evidence required for Severe Anemia package
    """
    if package_code == "MG064A":
        return {
            "visual_check_required": False,
            "reason": "Severe Anemia STG does not mandate photographic evidence"
        }
```

**Pipeline Action**: Bypass object detection and photo-matching steps for MG064A. Rely entirely on document classification and OCR of lab and treatment reports.

---

## 4. TMS Rules Engine (Pass/Fail Logic Gates)

### Rule 1: Severity Threshold (Hemoglobin Level)
**Requirement**: Patient's starting hemoglobin must be dangerously low

```python
def check_severity_threshold(pre_treatment_hb):
    """
    TMS Rule 1: Severity Threshold
    """
    severity_threshold = 7.0  # g/dl
    
    if pre_treatment_hb < severity_threshold:
        return {
            "status": "PASS",
            "rule": "TMS_RULE_1_SEVERITY",
            "reason": f"Pre-treatment hemoglobin {pre_treatment_hb} g/dl is below 7 g/dl threshold"
        }
    else:
        return {
            "status": "FAIL",
            "rule": "TMS_RULE_1_SEVERITY",
            "reason": f"Patient hemoglobin level {pre_treatment_hb} g/dl was not less than 7 g/dl, failing mandatory criteria for severe anemia admission"
        }
```

**Logic**: `IF (Pre_treatment_hemoglobin_level < 7 g/dl) == TRUE → PASS`

**Failure**: Output FAIL (Reason: Patient hemoglobin level was not less than 7 g/dl, failing mandatory criteria for severe anemia admission)

**Examples**:
- ✅ PASS: 6.5 g/dl
- ✅ PASS: 5.2 g/dl
- ❌ FAIL: 7.0 g/dl (not strictly less than 7)
- ❌ FAIL: 8.5 g/dl

---

### Rule 2: Treatment Administered
**Requirement**: Hospital must have administered required treatments for severe anemia

```python
def check_treatment_administered(treatment_details):
    """
    TMS Rule 2: Treatment Administered
    """
    treatment_text = treatment_details.lower()
    
    # Check for blood transfusion
    transfusion_keywords = [
        "blood transfusion",
        "blood transfused",
        "packed red blood cells",
        "prbc",
        "whole blood transfusion",
        "transfusion given"
    ]
    
    transfusion_found = any(kw in treatment_text for kw in transfusion_keywords)
    
    # Check for ferrous sulphate injection
    ferrous_keywords = [
        "ferrous sulphate injection",
        "ferrous sulfate injection",
        "iron injection",
        "parenteral iron"
    ]
    
    ferrous_found = any(kw in treatment_text for kw in ferrous_keywords)
    
    if transfusion_found and ferrous_found:
        return {
            "status": "PASS",
            "rule": "TMS_RULE_2_TREATMENT",
            "reason": "Both blood transfusion and ferrous sulphate injection documented"
        }
    else:
        missing = []
        if not transfusion_found:
            missing.append("blood transfusion")
        if not ferrous_found:
            missing.append("ferrous sulphate injection")
        
        return {
            "status": "FAIL",
            "rule": "TMS_RULE_2_TREATMENT",
            "reason": f"Mandatory treatment not recorded: {', '.join(missing)}"
        }
```

**Logic**: `IF Treatment_Details CONTAINS "blood transfusion" AND "ferrous sulphate injection" == TRUE → PASS`

**Failure**: Output FAIL (Reason: Mandatory blood transfusion and ferrous sulphate injection not recorded in treatment details)

---

### Rule 3: Lab Report Verification (Pre vs. Post)
**Requirement**: Hospital must have conducted both initial and follow-up blood work

```python
def check_lab_report_verification_anemia(classified_documents):
    """
    TMS Rule 3: Lab Report Verification (Pre vs. Post)
    """
    # Required pre-treatment labs
    pre_treatment_required = [
        "Pre-treatment CBC",
        "Pre-treatment Hb Report"
    ]
    
    # Required post-treatment labs
    post_treatment_required = [
        "Post-treatment CBC",
        "Post-treatment Hb Report"
    ]
    
    # Check pre-treatment labs
    missing_pre = [doc for doc in pre_treatment_required if doc not in classified_documents]
    
    # Check post-treatment labs
    missing_post = [doc for doc in post_treatment_required if doc not in classified_documents]
    
    if not missing_pre and not missing_post:
        return {
            "status": "PASS",
            "rule": "TMS_RULE_3_LAB_VERIFICATION",
            "reason": "Both pre-treatment and post-treatment Hb reports present"
        }
    elif missing_post:
        return {
            "status": "CONDITIONAL",
            "rule": "TMS_RULE_3_LAB_VERIFICATION",
            "reason": f"Mandatory post-treatment Hb report is missing: {', '.join(missing_post)}"
        }
    elif missing_pre:
        return {
            "status": "CONDITIONAL",
            "rule": "TMS_RULE_3_LAB_VERIFICATION",
            "reason": f"Mandatory pre-treatment Hb report is missing: {', '.join(missing_pre)}"
        }
```

**Logic**: `IF Document_Classifier detects BOTH (Pre_Auth_CBC_Hb) AND (Claim_Post_Treatment_CBC_Hb) == TRUE → PASS`

**Failure**: Output CONDITIONAL (Reason: Mandatory post-treatment Hb report is missing)

---

### Rule 4: Chronology / ALOS Validation
**Requirement**: Length of stay must be within approved limits

```python
def check_alos_validation_anemia(admission_date, discharge_date):
    """
    TMS Rule 4: Average Length of Stay (ALOS) Validation for Severe Anemia
    """
    from datetime import datetime
    
    # Calculate length of stay
    los = (discharge_date - admission_date).days
    
    # ALOS for Severe Anemia: 3 days
    max_los = 3
    
    if los <= max_los:
        return {
            "status": "PASS",
            "rule": "TMS_RULE_4_ALOS",
            "reason": f"Length of stay ({los} days) within ALOS limit (3 days)"
        }
    else:
        return {
            "status": "CONDITIONAL",
            "rule": "TMS_RULE_4_ALOS",
            "reason": f"Length of stay ({los} days) exceeds standard ALOS of 3 days, requires manual review"
        }
```

**ALOS Limits**:
- **Severe Anemia (MG064A)**: ≤ 3 days

**Logic**: `IF (Discharge_Date - Admission_Date) <= 3 days → PASS`

**Failure**: Output CONDITIONAL (Reason: Length of stay exceeds standard ALOS of 3 days, requires manual review)

---

## 5. Final Decision Logic

### Decision Tree

```python
def make_final_decision_anemia(document_checks, tms_rules):
    """
    Final decision logic for Severe Anemia claims
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
            "Pre-treatment Hb < 7 g/dl confirmed",
            "Blood transfusion and ferrous sulphate injection documented",
            "All TMS rules passed"
        ],
        "rule_flags": []
    }
```

### Pass Criteria
✅ All 8 required documents present (3 pre-auth + 5 claim)
✅ Both pre-treatment AND post-treatment Hb reports present
✅ TMS Rule 1 (Severity) = PASS (Pre-treatment Hb < 7 g/dl)
✅ TMS Rule 2 (Treatment) = PASS (Blood transfusion + Ferrous sulphate injection)
✅ TMS Rule 3 (Lab Verification) = PASS
✅ TMS Rule 4 (ALOS) = PASS (≤3 days)

**Result**: Output PASS

### Conditional Criteria
⚠️ Any required document missing
⚠️ Post-treatment Hb report missing
⚠️ Pre-treatment Hb report missing
⚠️ TMS Rule 3 (Lab Verification) = CONDITIONAL
⚠️ TMS Rule 4 (ALOS) exceeds 3 days

**Result**: Output CONDITIONAL (Requires manual review)

### Fail Criteria
❌ TMS Rule 1 (Severity) = FAIL (Pre-treatment Hb ≥ 7 g/dl)
❌ TMS Rule 2 (Treatment) = FAIL (Missing blood transfusion or ferrous sulphate injection)

**Result**: Output FAIL

---

## 6. Implementation Checklist

### Document Classification Module
- [ ] Implement classifier for 8 document types
- [ ] Distinguish between pre-treatment and post-treatment lab reports
- [ ] Track missing documents separately (pre vs. post)
- [ ] Generate CONDITIONAL flag for missing docs

### OCR & NLP Module
- [ ] Extract hemoglobin values (handle various formats: Hb, HGB, hemoglobin)
- [ ] Extract units (g/dl, g/dL, gm/dl)
- [ ] Distinguish pre-treatment vs. post-treatment Hb values
- [ ] Extract treatment keywords (blood transfusion, ferrous sulphate)
- [ ] Extract symptom keywords (pallor, fatigue, dyspnea)
- [ ] Extract severe signs (shock, bleeding, respiratory distress)
- [ ] Extract admission and discharge dates
- [ ] Calculate length of stay

### Computer Vision Module
- [ ] Bypass visual evidence checks for MG064A
- [ ] No object detection required
- [ ] No photo matching required

### Rules Engine Module
- [ ] Implement TMS Rule 1 (Severity Threshold - Hb < 7 g/dl)
- [ ] Implement TMS Rule 2 (Treatment Administered)
- [ ] Implement TMS Rule 3 (Lab Report Verification - Pre vs. Post)
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
    "rule": "TMS_RULE_1_SEVERITY",
    "status": "PASS",
    "source_document": "Pre_treatment_CBC.pdf",
    "page_number": 1,
    "bounding_box": [200, 350, 500, 400],
    "confidence": 0.94,
    "extracted_text": "Hemoglobin: 6.2 g/dl",
    "extracted_values": {
        "hemoglobin": 6.2,
        "unit": "g/dl",
        "phase": "pre_treatment"
    },
    "timestamp": "2026-04-25T10:30:00Z"
}
```

---

## 8. Special Considerations

### Hemoglobin Extraction Challenges

```python
def extract_hemoglobin(text):
    """
    Extract hemoglobin value with various formats
    """
    import re
    
    # Patterns to match
    patterns = [
        r'(?:hemoglobin|haemoglobin|Hb|HGB)[:\s]+(\d+\.?\d*)\s*(?:g/dl|g/dL|gm/dl)?',
        r'Hb[:\s]+(\d+\.?\d*)',
        r'hemoglobin[:\s]+(\d+\.?\d*)',
        r'HGB[:\s]+(\d+\.?\d*)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            hb_value = float(match.group(1))
            
            # Sanity check (normal range 12-18 g/dl, severe anemia < 7 g/dl)
            if 2.0 <= hb_value <= 20.0:
                return {
                    "value": hb_value,
                    "unit": "g/dl",
                    "confidence": 0.9
                }
    
    return None
```

### Treatment Keyword Extraction

```python
def extract_treatment_keywords(treatment_text):
    """
    Extract and verify mandatory treatment keywords
    """
    treatment_lower = treatment_text.lower()
    
    # Blood transfusion variations
    transfusion_found = any([
        "blood transfusion" in treatment_lower,
        "blood transfused" in treatment_lower,
        "packed red blood cells" in treatment_lower,
        "prbc" in treatment_lower,
        "whole blood" in treatment_lower
    ])
    
    # Ferrous sulphate variations
    ferrous_found = any([
        "ferrous sulphate injection" in treatment_lower,
        "ferrous sulfate injection" in treatment_lower,
        "iron injection" in treatment_lower,
        "parenteral iron" in treatment_lower,
        "iv iron" in treatment_lower
    ])
    
    return {
        "blood_transfusion": transfusion_found,
        "ferrous_sulphate_injection": ferrous_found,
        "both_present": transfusion_found and ferrous_found
    }
```

### Pre vs. Post Treatment Lab Distinction

```python
def classify_lab_timing_anemia(document_text, document_metadata):
    """
    Distinguish between pre-treatment and post-treatment Hb reports
    """
    pre_keywords = [
        "pre-treatment",
        "pre treatment",
        "initial",
        "admission",
        "baseline",
        "before treatment",
        "before transfusion"
    ]
    
    post_keywords = [
        "post-treatment",
        "post treatment",
        "follow-up",
        "discharge",
        "after treatment",
        "after transfusion",
        "repeat",
        "post transfusion"
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

### Hemoglobin Improvement Validation (Optional Enhancement)

```python
def validate_hb_improvement(pre_hb, post_hb):
    """
    Optional: Validate that Hb improved after treatment
    """
    if pre_hb and post_hb:
        improvement = post_hb - pre_hb
        
        if improvement > 0:
            return {
                "improved": True,
                "improvement": improvement,
                "reason": f"Hb improved from {pre_hb} to {post_hb} g/dl (+{improvement} g/dl)"
            }
        else:
            return {
                "improved": False,
                "improvement": improvement,
                "reason": f"Hb did not improve (from {pre_hb} to {post_hb} g/dl)"
            }
    
    return None
```

---

## 9. Edge Cases and Handling

### Edge Case 1: Hb exactly at 7.0 g/dl
```python
# Rule: Must be STRICTLY less than 7 g/dl
if hb == 7.0:
    # FAIL - not strictly less than 7
    status = "FAIL"
```

### Edge Case 2: Multiple Hb values in same document
```python
# Take the lowest value for pre-treatment (most conservative)
# Take the highest value for post-treatment (best outcome)
def select_hb_value(hb_values, phase):
    if phase == "PRE_TREATMENT":
        return min(hb_values)  # Lowest value
    elif phase == "POST_TREATMENT":
        return max(hb_values)  # Highest value
```

### Edge Case 3: Oral iron vs. Injectable iron
```python
# Only injectable (ferrous sulphate injection) counts
# Oral iron supplements do NOT satisfy the requirement
oral_keywords = ["oral iron", "iron tablets", "iron pills", "ferrous sulphate tablets"]
if any(kw in treatment_text for kw in oral_keywords) and "injection" not in treatment_text:
    # FAIL - only oral iron given
    status = "FAIL"
```

### Edge Case 4: Units other than g/dl
```python
def convert_hb_units(value, unit):
    """
    Convert various Hb units to g/dl
    """
    if unit in ["g/dl", "g/dL", "gm/dl"]:
        return value
    elif unit in ["g/L", "g/l"]:
        return value / 10  # Convert g/L to g/dl
    elif unit in ["mg/dl", "mg/dL"]:
        return value / 1000  # Convert mg/dl to g/dl
    else:
        return None  # Unknown unit
```

---

**Document**: STG_RULES_SEVERE_ANEMIA.md
**Package Code**: MG064A
**Created**: April 25, 2026
**Status**: Complete Technical Specification
