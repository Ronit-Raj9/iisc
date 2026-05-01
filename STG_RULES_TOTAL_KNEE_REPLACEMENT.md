# STG Rules: Total Knee Replacement (Primary and Revision)

## Package Codes
- **SB039A**: Total Knee Replacement - Primary
- **SB039B**: Total Knee Replacement - Revision

---

## 1. Document Classification Rules (Completeness Checks)

### Pre-authorization Phase Documents

| Document Type | Required | Validation Criteria | Missing Action | Package-Specific |
|---------------|----------|---------------------|----------------|------------------|
| Clinical Notes | ✅ Yes | Must include indications for surgery | Flag CONDITIONAL | Both |
| X-ray of Knee | ✅ Yes | Must be labeled with patient ID, date, and Left/Right side | Flag CONDITIONAL | Both |
| CT of Knee | ⚠️ Alternative | Can substitute X-ray. Must be labeled with patient ID, date, and Left/Right side | Flag CONDITIONAL | Both |
| Pre-op X-ray showing old implant | ✅ Yes (Revision only) | Strictly mandatory for Revision TKR. Must show existing implant | Flag CONDITIONAL | SB039B only |

### Claim Submission Phase Documents

| Document Type | Required | Validation Criteria | Missing Action |
|---------------|----------|---------------------|----------------|
| Indoor Case Papers | ✅ Yes | Detailed records of hospital stay | Flag CONDITIONAL |
| Operative/Procedure Note | ✅ Yes | Detailed operative note | Flag CONDITIONAL |
| Implant Invoice/Barcode | ✅ Yes | Mandatory for tracing specific device used | Flag CONDITIONAL |
| Post-op Clinical Photograph | ✅ Yes | Clinical photograph after surgery | Flag CONDITIONAL |
| Post-op X-ray | ✅ Yes | Must be labeled with patient ID, date, Left/Right side, and show new implant | Flag CONDITIONAL |
| Discharge Summary | ✅ Yes | Detailed discharge summary | Flag CONDITIONAL |

**Total Required Documents**: 
- **Primary TKR (SB039A)**: 8 documents (2 pre-auth + 6 claim)
- **Revision TKR (SB039B)**: 9 documents (3 pre-auth + 6 claim)

**Critical Note**: Revision TKR requires additional pre-op X-ray showing the old implant.

---

## 2. OCR & Text Extraction Targets

### Patient Demographics (Mandatory)

```python
PATIENT_DEMOGRAPHICS = {
    "age": {
        "keywords": ["age", "years old", "y/o", "yrs"],
        "primary_tkr_threshold": 55,  # Must be > 55 for primary TKR
        "revision_tkr_threshold": None  # No age restriction for revision
    }
}
```

### Clinical Keywords to Extract

#### Primary Osteoarthritis Keywords (from Clinical Notes)
```python
PRIMARY_OA_KEYWORDS = [
    "primary osteoarthritis",
    "osteoarthritis",
    "OA",
    "genu varum",
    "bow-legged",
    "narrowed joint space",
    "joint space narrowing",
    "frequent locking episodes",
    "locking episodes",
    "joint instability",
    "knee instability"
]
```

#### Secondary Arthritis Keywords (from Clinical Notes)
```python
SECONDARY_ARTHRITIS_KEYWORDS = [
    "post trauma",
    "post-traumatic",
    "rheumatoid arthritis",
    "RA",
    "osteonecrosis",
    "avascular necrosis",
    "inflammatory arthritis",
    "secondary arthritis"
]
```

#### Severity Justification Keywords (from Clinical Notes)
```python
SEVERITY_KEYWORDS = [
    "pain refractive to conservative management",
    "refractory to conservative management",
    "failed conservative management",
    "interferes with sleep",
    "sleep disturbance",
    "inability to cope with side effects",
    "side effects of pain relief",
    "severe pain",
    "debilitating pain",
    "unable to walk"
]
```

#### Exclusion Keywords (from Clinical Notes)
```python
EXCLUSION_KEYWORDS = [
    "evidence of infection",
    "active infection",
    "infected joint",
    "septic arthritis",
    "wound infection"
]
```

### Imaging Report Extraction

#### X-ray/CT Metadata (Mandatory)
```python
IMAGING_METADATA = {
    "patient_id": ["patient ID", "patient name", "ID number"],
    "date": ["date", "scan date", "X-ray date"],
    "laterality": ["left", "right", "L", "R", "left knee", "right knee"]
}
```

### Date Extraction
- Admission Date
- Discharge Date
- Surgery Date (from Operative Note)
- Pre-op X-ray Date
- Post-op X-ray Date
- Calculate: Length of Stay (LOS) = Discharge Date - Admission Date

---

## 3. Computer Vision Rules (Visual Evidence)

### Critical Visual Checks (All Packages)

| Visual Element | Detection Required | Missing Action | Source Document |
|----------------|-------------------|----------------|-----------------|
| Patient ID on Pre-op X-ray/CT | ✅ Yes | Flag CONDITIONAL | Pre-op X-ray/CT |
| Date on Pre-op X-ray/CT | ✅ Yes | Flag CONDITIONAL | Pre-op X-ray/CT |
| Laterality (L/R) on Pre-op X-ray/CT | ✅ Yes | Flag CONDITIONAL | Pre-op X-ray/CT |
| Patient ID on Post-op X-ray | ✅ Yes | Flag CONDITIONAL | Post-op X-ray |
| Date on Post-op X-ray | ✅ Yes | Flag CONDITIONAL | Post-op X-ray |
| Laterality (L/R) on Post-op X-ray | ✅ Yes | Flag CONDITIONAL | Post-op X-ray |
| Implant in Post-op X-ray | ✅ Yes | Flag FAIL | Post-op X-ray |
| Implant Barcode/Invoice | ✅ Yes | Flag CONDITIONAL | Invoice document |

### Package-Specific Visual Checks

#### For Revision TKR (SB039B) Only
| Visual Element | Detection Required | Missing Action | Source Document |
|----------------|-------------------|----------------|-----------------|
| Old Implant in Pre-op X-ray | ✅ Yes | Flag FAIL | Pre-op X-ray |

```python
def check_visual_evidence_tkr(package_code, pre_op_xray, post_op_xray, invoice):
    """
    Computer vision checks for Total Knee Replacement
    """
    checks = {
        "pre_op_patient_id": detect_patient_id(pre_op_xray),
        "pre_op_date": detect_date(pre_op_xray),
        "pre_op_laterality": detect_laterality(pre_op_xray),
        "post_op_patient_id": detect_patient_id(post_op_xray),
        "post_op_date": detect_date(post_op_xray),
        "post_op_laterality": detect_laterality(post_op_xray),
        "post_op_implant": detect_implant(post_op_xray),
        "implant_barcode": detect_barcode(invoice)
    }
    
    # Additional check for Revision TKR
    if package_code == "SB039B":
        checks["pre_op_old_implant"] = detect_implant(pre_op_xray)
    
    return checks
```

---

## 4. TMS Rules Engine (Pass/Fail Logic Gates)

### Rule 1: Post-Op Visual Verification
**Requirement**: Post-op X-ray must show the implant

```python
def check_post_op_visual_verification(post_op_xray_analysis):
    """
    TMS Rule 1: Post-Op Visual Verification
    """
    implant_detected = post_op_xray_analysis.get("implant_detected", False)
    
    if implant_detected:
        return {
            "status": "PASS",
            "rule": "TMS_RULE_1_POST_OP_VISUAL",
            "reason": "Post-operative X-ray shows evidence of implant"
        }
    else:
        return {
            "status": "FAIL",
            "rule": "TMS_RULE_1_POST_OP_VISUAL",
            "reason": "Mandatory post-operative imaging does not show evidence of implant"
        }
```

**Logic**: `IF Implant_Detected_Post_Op_Xray == TRUE → PASS`

**Failure**: Output FAIL (Reason: Mandatory post-operative imaging does not show evidence of implant)

---

### Rule 2: Pre-Op Revision Verification
**Requirement**: For revision surgery, pre-op X-ray must show old implant

```python
def check_pre_op_revision_verification(package_code, pre_op_xray_analysis):
    """
    TMS Rule 2: Pre-Op Revision Verification
    """
    # Only applies to Revision TKR
    if package_code != "SB039B":
        return {
            "status": "PASS",
            "rule": "TMS_RULE_2_PRE_OP_REVISION",
            "reason": "Not applicable for Primary TKR"
        }
    
    old_implant_detected = pre_op_xray_analysis.get("implant_detected", False)
    
    if old_implant_detected:
        return {
            "status": "PASS",
            "rule": "TMS_RULE_2_PRE_OP_REVISION",
            "reason": "Pre-operative X-ray shows existing implant for revision surgery"
        }
    else:
        return {
            "status": "FAIL",
            "rule": "TMS_RULE_2_PRE_OP_REVISION",
            "reason": "Pre-operative imaging for revision surgery does not show the presence of an existing implant"
        }
```

**Logic**: `IF Package == Revision TKR THEN IF Implant_Detected_Pre_Op_Xray == TRUE → PASS`

**Failure**: Output FAIL (Reason: Pre-operative imaging for revision surgery does not show the presence of an existing implant)

---

### Rule 3: Age Check for Primary TKR
**Requirement**: Patient must be old enough for primary joint replacement

```python
def check_age_primary_tkr(package_code, patient_age, diagnosis, history):
    """
    TMS Rule 3: Age Check for Primary TKR
    """
    # Only applies to Primary TKR
    if package_code != "SB039A":
        return {
            "status": "PASS",
            "rule": "TMS_RULE_3_AGE_CHECK",
            "reason": "Not applicable for Revision TKR"
        }
    
    # Check if diagnosis is primary OA
    is_primary_oa = any(kw in diagnosis.lower() for kw in ["primary osteoarthritis", "osteoarthritis"])
    
    # Check if there's trauma or systemic disease
    has_trauma = any(kw in history.lower() for kw in ["trauma", "post-traumatic", "injury"])
    has_systemic = any(kw in history.lower() for kw in ["rheumatoid", "inflammatory", "systemic"])
    
    # Age threshold
    age_threshold = 55
    
    # If primary OA without trauma/systemic disease, age must be > 55
    if is_primary_oa and not has_trauma and not has_systemic:
        if patient_age > age_threshold:
            return {
                "status": "PASS",
                "rule": "TMS_RULE_3_AGE_CHECK",
                "reason": f"Patient age {patient_age} years meets >55 years criteria for primary TKR"
            }
        else:
            return {
                "status": "FAIL",
                "rule": "TMS_RULE_3_AGE_CHECK",
                "reason": f"Patient age {patient_age} years does not meet the >55 years mandatory criteria for primary TKR"
            }
    else:
        # Trauma or systemic disease present - age restriction waived
        return {
            "status": "PASS",
            "rule": "TMS_RULE_3_AGE_CHECK",
            "reason": f"Age restriction waived due to trauma or systemic joint disease"
        }
```

**Logic**: `IF Diagnosis == Primary OA AND History == No Trauma/Systemic Disease THEN IF Patient_Age > 55 → PASS`

**Failure**: Output FAIL (Reason: Patient age does not meet the >55 years mandatory criteria for primary TKR)

**Examples**:
- ✅ PASS: 60 years old with primary OA
- ✅ PASS: 45 years old with post-traumatic arthritis
- ✅ PASS: 50 years old with rheumatoid arthritis
- ❌ FAIL: 45 years old with primary OA (no trauma/systemic disease)

---

### Rule 4: Infection Exclusion Check
**Requirement**: No evidence of infection

```python
def check_infection_exclusion(clinical_notes, operative_notes):
    """
    TMS Rule 4: Infection Exclusion Check
    """
    combined_text = f"{clinical_notes} {operative_notes}".lower()
    
    exclusion_keywords = [
        "evidence of infection",
        "active infection",
        "infected joint",
        "septic arthritis",
        "wound infection"
    ]
    
    found_infections = [kw for kw in exclusion_keywords if kw in combined_text]
    
    if found_infections:
        return {
            "status": "FAIL",
            "rule": "TMS_RULE_4_INFECTION",
            "reason": f"Evidence of infection found: {', '.join(found_infections)}. STG requires no evidence of infection."
        }
    else:
        return {
            "status": "PASS",
            "rule": "TMS_RULE_4_INFECTION",
            "reason": "No evidence of infection"
        }
```

**Logic**: `IF Clinical_Notes CONTAINS "evidence of infection" == FALSE → PASS`

**Failure**: Output FAIL (Reason: Evidence of infection found. STG requires no evidence of infection)

---

### Rule 5: Chronology / ALOS Validation
**Requirement**: Length of stay must be within approved limits

```python
def check_alos_validation_tkr(admission_date, discharge_date):
    """
    TMS Rule 5: Average Length of Stay (ALOS) Validation for TKR
    """
    from datetime import datetime
    
    # Calculate length of stay
    los = (discharge_date - admission_date).days
    
    # ALOS for TKR: 5-7 days
    min_los = 5
    max_los = 7
    
    if los <= max_los:
        return {
            "status": "PASS",
            "rule": "TMS_RULE_5_ALOS",
            "reason": f"Length of stay ({los} days) within ALOS limit (5-7 days)"
        }
    else:
        return {
            "status": "CONDITIONAL",
            "rule": "TMS_RULE_5_ALOS",
            "reason": f"Stay exceeds ALOS of 5-7 days ({los} days), manual review required"
        }
```

**ALOS Limits**:
- **Total Knee Replacement (SB039A/B)**: 5-7 days

**Logic**: `IF (Discharge_Date - Admission_Date) <= 7 days → PASS`

**Failure**: Output CONDITIONAL (Reason: Stay exceeds ALOS of 5-7 days, manual review required)

---

## 5. Final Decision Logic

### Decision Tree

```python
def make_final_decision_tkr(package_code, document_checks, visual_checks, tms_rules):
    """
    Final decision logic for Total Knee Replacement claims
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
    
    # Check visual evidence
    missing_visuals = [vis for vis, present in visual_checks.items() if not present]
    
    if missing_visuals:
        return {
            "decision": "CONDITIONAL",
            "confidence": 0.75,
            "reasons": [f"Missing visual evidence: {vis}" for vis in missing_visuals],
            "missing_documents": missing_visuals
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
    package_name = "Primary TKR" if package_code == "SB039A" else "Revision TKR"
    return {
        "decision": "PASS",
        "confidence": 0.95,
        "reasons": [
            f"All mandatory documents present for {package_name}",
            "Visual evidence verified (laterality, patient ID, implants)",
            "All TMS rules passed"
        ],
        "rule_flags": []
    }
```

### Pass Criteria

#### Primary TKR (SB039A)
✅ All 8 required documents present
✅ Pre-op X-ray/CT with patient ID, date, laterality
✅ Post-op X-ray with patient ID, date, laterality
✅ TMS Rule 1 (Post-op Visual) = PASS (Implant visible)
✅ TMS Rule 2 (Pre-op Revision) = N/A
✅ TMS Rule 3 (Age Check) = PASS (Age >55 OR trauma/systemic disease)
✅ TMS Rule 4 (Infection) = PASS (No infection)
✅ TMS Rule 5 (ALOS) = PASS (≤7 days)

#### Revision TKR (SB039B)
✅ All 9 required documents present
✅ Pre-op X-ray showing old implant
✅ Post-op X-ray with new implant
✅ TMS Rule 1 (Post-op Visual) = PASS (New implant visible)
✅ TMS Rule 2 (Pre-op Revision) = PASS (Old implant visible)
✅ TMS Rule 3 (Age Check) = N/A
✅ TMS Rule 4 (Infection) = PASS (No infection)
✅ TMS Rule 5 (ALOS) = PASS (≤7 days)

**Result**: Output PASS

### Conditional Criteria
⚠️ Any required document missing
⚠️ Visual evidence missing (patient ID, date, laterality)
⚠️ Implant barcode/invoice missing
⚠️ TMS Rule 5 (ALOS) exceeds 7 days

**Result**: Output CONDITIONAL (Requires manual review)

### Fail Criteria
❌ TMS Rule 1 (Post-op Visual) = FAIL (No implant in post-op X-ray)
❌ TMS Rule 2 (Pre-op Revision) = FAIL (No old implant in pre-op X-ray for revision)
❌ TMS Rule 3 (Age Check) = FAIL (Age ≤55 for primary OA without trauma/systemic disease)
❌ TMS Rule 4 (Infection) = FAIL (Evidence of infection found)

**Result**: Output FAIL

---

## 6. Implementation Checklist

### Document Classification Module
- [ ] Implement classifier for 8-9 document types
- [ ] Distinguish between Primary and Revision TKR packages
- [ ] Track package-specific document requirements
- [ ] Generate CONDITIONAL flag for missing docs

### OCR & NLP Module
- [ ] Extract patient age
- [ ] Extract primary OA keywords
- [ ] Extract secondary arthritis keywords
- [ ] Extract severity justification keywords
- [ ] Scan for infection exclusion keywords
- [ ] Extract admission and discharge dates
- [ ] Extract surgery date
- [ ] Calculate length of stay

### Computer Vision Module
- [ ] Detect patient ID on X-rays/CT scans
- [ ] Detect date on X-rays/CT scans
- [ ] Detect laterality (Left/Right) on X-rays/CT scans
- [ ] Detect implant in post-op X-ray
- [ ] Detect old implant in pre-op X-ray (Revision only)
- [ ] Detect and extract implant barcode/invoice
- [ ] Verify laterality consistency across documents

### Rules Engine Module
- [ ] Implement TMS Rule 1 (Post-Op Visual Verification)
- [ ] Implement TMS Rule 2 (Pre-Op Revision Verification)
- [ ] Implement TMS Rule 3 (Age Check for Primary TKR)
- [ ] Implement TMS Rule 4 (Infection Exclusion Check)
- [ ] Implement TMS Rule 5 (ALOS Validation)

### Decision Module
- [ ] Implement final decision logic
- [ ] Handle Primary vs. Revision TKR differences
- [ ] Generate confidence scores
- [ ] Generate reason strings
- [ ] Track rule flags
- [ ] List missing documents

---

## 7. Evidence Provenance Tracking

For each rule check, track:
- **Source Document**: Which document was analyzed
- **Page Number**: Where the evidence was found
- **Bounding Box**: Coordinates of relevant text/image
- **Confidence Score**: Model confidence in extraction
- **Extracted Text/Visual**: Actual content used for rule

```python
evidence_template = {
    "rule": "TMS_RULE_1_POST_OP_VISUAL",
    "status": "PASS",
    "source_document": "Post_op_Xray.jpg",
    "page_number": 1,
    "bounding_box": [300, 400, 600, 800],
    "confidence": 0.91,
    "visual_detection": {
        "implant_detected": True,
        "patient_id": "12345",
        "date": "2026-04-20",
        "laterality": "Right"
    },
    "timestamp": "2026-04-25T10:30:00Z"
}
```

---

## 8. Special Considerations

### Age Extraction

```python
def extract_patient_age(text):
    """
    Extract patient age with various formats
    """
    import re
    
    patterns = [
        r'age[:\s]+(\d+)',
        r'(\d+)\s*(?:years old|y/o|yrs|years)',
        r'age\s*[:\-]\s*(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            age = int(match.group(1))
            
            # Sanity check (reasonable age range)
            if 18 <= age <= 120:
                return age
    
    return None
```

### Laterality Detection from X-ray Images

```python
def detect_laterality_from_image(xray_image):
    """
    Detect Left/Right laterality from X-ray image using OCR
    """
    import pytesseract
    from PIL import Image
    
    # OCR on the image
    text = pytesseract.image_to_string(xray_image)
    text_lower = text.lower()
    
    # Look for laterality markers
    if any(marker in text_lower for marker in ["left", "l knee", "lt", "lft"]):
        return "LEFT"
    elif any(marker in text_lower for marker in ["right", "r knee", "rt", "rgt"]):
        return "RIGHT"
    else:
        return "UNKNOWN"
```

### Implant Detection from X-ray

```python
def detect_implant_from_xray(xray_image):
    """
    Detect presence of knee implant in X-ray using computer vision
    """
    # This would use a trained object detection model
    # Placeholder for actual implementation
    
    # Look for high-density metallic objects in knee region
    # Typical knee implant characteristics:
    # - High contrast (bright white on X-ray)
    # - Specific shapes (femoral component, tibial component)
    # - Located in knee joint area
    
    implant_detected = detect_metallic_implant(xray_image)
    confidence = calculate_detection_confidence(xray_image)
    
    return {
        "implant_detected": implant_detected,
        "confidence": confidence,
        "bounding_box": get_implant_bbox(xray_image)
    }
```

### Barcode Extraction from Invoice

```python
def extract_barcode_from_invoice(invoice_image):
    """
    Extract implant barcode from invoice document
    """
    from pyzbar import pyzbar
    import cv2
    
    # Decode barcodes
    barcodes = pyzbar.decode(invoice_image)
    
    if barcodes:
        barcode_data = []
        for barcode in barcodes:
            barcode_data.append({
                "data": barcode.data.decode("utf-8"),
                "type": barcode.type,
                "bbox": barcode.rect
            })
        
        return {
            "barcode_found": True,
            "barcodes": barcode_data
        }
    else:
        return {
            "barcode_found": False,
            "barcodes": []
        }
```

### Laterality Consistency Check

```python
def check_laterality_consistency(pre_op_laterality, post_op_laterality, clinical_notes_laterality):
    """
    Verify that laterality is consistent across all documents
    """
    lateralities = [pre_op_laterality, post_op_laterality, clinical_notes_laterality]
    
    # Remove None/Unknown values
    valid_lateralities = [lat for lat in lateralities if lat and lat != "UNKNOWN"]
    
    if not valid_lateralities:
        return {
            "consistent": False,
            "reason": "No laterality information found"
        }
    
    # Check if all are the same
    if len(set(valid_lateralities)) == 1:
        return {
            "consistent": True,
            "laterality": valid_lateralities[0]
        }
    else:
        return {
            "consistent": False,
            "reason": f"Laterality mismatch: {', '.join(valid_lateralities)}"
        }
```

---

## 9. Edge Cases and Handling

### Edge Case 1: Age exactly 55
```python
# Rule: Must be STRICTLY greater than 55
if age == 55:
    # FAIL - not strictly greater than 55
    status = "FAIL"
```

### Edge Case 2: Bilateral TKR (Both Knees)
```python
# If both knees operated in same admission
# Should be two separate claims with different laterality
# Each claim must have its own complete documentation set
```

### Edge Case 3: Missing laterality marker on X-ray
```python
# If laterality cannot be detected from X-ray
# Try to extract from clinical notes or operative notes
# If still missing → Flag CONDITIONAL
```

### Edge Case 4: Primary vs. Revision package mismatch
```python
# If claimed as Primary but old implant visible in pre-op X-ray
# → Flag for manual review (possible fraud or incorrect package code)

# If claimed as Revision but no old implant visible
# → FAIL (does not meet revision criteria)
```

### Edge Case 5: Multiple X-rays submitted
```python
# Take the most recent pre-op X-ray (closest to surgery date)
# Take the most recent post-op X-ray (closest to discharge date)
```

---

## 10. Package-Specific Summary

### Primary TKR (SB039A)

**Key Requirements**:
- Age > 55 (unless trauma or systemic disease)
- No old implant in pre-op X-ray
- New implant visible in post-op X-ray
- No evidence of infection

**Common Failure Reasons**:
- Patient too young (<55) with primary OA
- No implant visible in post-op X-ray
- Evidence of infection

### Revision TKR (SB039B)

**Key Requirements**:
- Old implant visible in pre-op X-ray (mandatory)
- New implant visible in post-op X-ray
- No age restriction
- No evidence of infection

**Common Failure Reasons**:
- No old implant in pre-op X-ray
- No new implant in post-op X-ray
- Evidence of infection

---

**Document**: STG_RULES_TOTAL_KNEE_REPLACEMENT.md
**Package Codes**: SB039A (Primary), SB039B (Revision)
**Created**: April 25, 2026
**Status**: Complete Technical Specification
