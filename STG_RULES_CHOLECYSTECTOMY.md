# STG Rules: Cholecystectomy (SG039A/B/C/D)

## Package Codes
- **SG039A**: Open Cholecystectomy
- **SG039B**: Open Cholecystectomy with Exploration of CBD
- **SG039C**: Laparoscopic Cholecystectomy
- **SG039D**: Laparoscopic Cholecystectomy with Exploration of CBD

---

## 1. Document Classification Rules (Completeness Checks)

### Pre-authorization Phase Documents

| Document Type | Required | Validation Criteria | Missing Action |
|---------------|----------|---------------------|----------------|
| Clinical Notes | ✅ Yes | Must contain detailed history, signs & symptoms, and indications for procedure | Flag CONDITIONAL |
| USG Upper Abdomen | ✅ Yes | Must confirm presence of gall stones. For CBD exploration (SG039B/D): Accept MRCP or USG showing large (>1cm) or multiple CBD stones | Flag CONDITIONAL |
| LFT Report | ✅ Yes | Liver function test report must be present | Flag CONDITIONAL |

### Claim Submission Phase Documents

| Document Type | Required | Validation Criteria | Missing Action |
|---------------|----------|---------------------|----------------|
| Operative Notes | ✅ Yes | Must detail indications and outcomes of procedure | Flag CONDITIONAL |
| Pre-anesthesia Check-up | ✅ Yes | Pre-anesthesia report must be present | Flag CONDITIONAL |
| Discharge Summary | ✅ Yes | Must include follow-up advice at time of discharge | Flag CONDITIONAL |
| Intraoperative Photographs | ✅ Yes | Pictures during surgery | Flag CONDITIONAL |
| Gross Specimen Pictures | ✅ Yes | Pictures of removed gallbladder | Flag CONDITIONAL |
| Histopathology Report | ✅ Yes | Pathology report of removed specimen | Flag CONDITIONAL |

**Total Required Documents**: 9

---

## 2. OCR & Text Extraction Targets

### Clinical Keywords to Extract

#### From Clinical Notes
```python
SYMPTOM_KEYWORDS = [
    "right hypochondrium",
    "epigastrium",
    "biliary colic",
    "jaundice",
    "fever",
    "vomiting"
]
```

#### From USG/MRCP Reports
```python
DIAGNOSTIC_KEYWORDS = [
    "calculi",
    "gall stones",
    "choledocholithiasis"
]
```

#### From LFT Reports
```python
LAB_VALUE_KEYWORDS = [
    "serum bilirubin",
    "transaminase",
    "alkaline phosphate"
]
```

#### From Medical History
```python
HISTORY_KEYWORDS = [
    "prior Cholecystectomy",
    "previous Cholecystectomy",
    "history of Cholecystectomy"
]
```

#### Date Extraction
- Admission Date
- Discharge Date
- Calculate: Length of Stay (LOS) = Discharge Date - Admission Date

---

## 3. Computer Vision Rules (Visual Evidence)

### Standard Visual Checks (All Packages)

| Visual Element | Detection Required | Missing Action |
|----------------|-------------------|----------------|
| Intraoperative Photograph | ✅ Yes | Flag CONDITIONAL |
| Gross Specimen Picture (Removed Gallbladder) | ✅ Yes | Flag CONDITIONAL |

### Package-Specific Visual Checks

#### For CBD Exploration Packages (SG039B, SG039D)
| Visual Element | Detection Required | Missing Action |
|----------------|-------------------|----------------|
| Intraoperative Photo/Video showing stones being removed from CBD | ✅ Yes | Flag CONDITIONAL |

---

## 4. TMS Rules Engine (Pass/Fail Logic Gates)

### Rule 1: Imaging Evidence
**Requirement**: USG must show presence of calculi in gall bladder

```python
def check_imaging_evidence(usg_text):
    """
    TMS Rule 1: Imaging Evidence
    """
    keywords = ["calculi", "gall stones"]
    
    if any(keyword.lower() in usg_text.lower() for keyword in keywords):
        return {
            "status": "PASS",
            "rule": "TMS_RULE_1_IMAGING",
            "reason": "USG confirms presence of calculi"
        }
    else:
        return {
            "status": "FAIL",
            "rule": "TMS_RULE_1_IMAGING",
            "reason": "USG does not show calculi"
        }
```

**Logic**: `IF USG_text CONTAINS ("calculi" OR "gall stones") == TRUE → PASS`

**Failure**: Output FAIL (Reason: USG does not show calculi)

---

### Rule 2: Clinical Manifestation
**Requirement**: Patient must have complained of pain in right hypochondrium or epigastrium

```python
def check_clinical_manifestation(clinical_notes):
    """
    TMS Rule 2: Clinical Manifestation
    """
    keywords = ["right hypochondrium", "epigastrium"]
    
    if any(keyword.lower() in clinical_notes.lower() for keyword in keywords):
        return {
            "status": "PASS",
            "rule": "TMS_RULE_2_CLINICAL",
            "reason": "Mandatory clinical symptoms recorded"
        }
    else:
        return {
            "status": "FAIL",
            "rule": "TMS_RULE_2_CLINICAL",
            "reason": "Mandatory clinical symptoms not recorded"
        }
```

**Logic**: `IF Clinical_Notes CONTAINS ("right hypochondrium" OR "epigastrium") == TRUE → PASS`

**Failure**: Output FAIL (Reason: Mandatory clinical symptoms not recorded)

---

### Rule 3: Fraud/History Check
**Requirement**: Patient must NOT have had a prior Cholecystectomy

```python
def check_fraud_history(patient_history):
    """
    TMS Rule 3: Fraud/History Check
    """
    fraud_keywords = [
        "prior Cholecystectomy",
        "previous Cholecystectomy",
        "history of Cholecystectomy"
    ]
    
    if any(keyword.lower() in patient_history.lower() for keyword in fraud_keywords):
        return {
            "status": "FAIL",
            "rule": "TMS_RULE_3_FRAUD",
            "reason": "Patient already had gallbladder removed previously"
        }
    else:
        return {
            "status": "PASS",
            "rule": "TMS_RULE_3_FRAUD",
            "reason": "No prior Cholecystectomy found"
        }
```

**Logic**: `IF Patient_History CONTAINS "prior Cholecystectomy" == FALSE → PASS`

**Failure**: Output FAIL (Reason: Patient already had gallbladder removed previously)

---

### Rule 4: Clinical Marker Check (LFT)
**Requirement**: LFT report must show elevated markers

```python
def check_lft_markers(lft_report):
    """
    TMS Rule 4: Clinical Marker Check
    """
    markers = ["serum bilirubin", "transaminase", "alkaline phosphate"]
    elevated_keywords = ["elevated", "high", "increased", "raised"]
    
    # Check if any marker is mentioned with elevation indicator
    for marker in markers:
        if marker.lower() in lft_report.lower():
            # Check for elevation indicators near the marker
            if any(elev.lower() in lft_report.lower() for elev in elevated_keywords):
                return {
                    "status": "PASS",
                    "rule": "TMS_RULE_4_LFT",
                    "reason": f"LFT shows elevated {marker}"
                }
    
    return {
        "status": "FAIL",
        "rule": "TMS_RULE_4_LFT",
        "reason": "LFT markers do not indicate gallbladder/biliary issue"
    }
```

**Logic**: `IF LFT_Report INDICATES elevated ("serum bilirubin" OR "transaminase" OR "alkaline phosphate") == TRUE → PASS`

**Failure**: Output FAIL (Reason: LFT markers do not indicate gallbladder/biliary issue)

---

### Rule 5: Chronology / ALOS Validation
**Requirement**: Length of stay must be within approved limits

```python
def check_alos_validation(admission_date, discharge_date, package_code):
    """
    TMS Rule 5: Average Length of Stay (ALOS) Validation
    """
    from datetime import datetime
    
    # Calculate length of stay
    los = (discharge_date - admission_date).days
    
    # Define ALOS limits by package
    alos_limits = {
        "SG039A": 6,  # Open Cholecystectomy
        "SG039B": 6,  # Open with CBD exploration
        "SG039C": 3,  # Laparoscopic
        "SG039D": 3   # Laparoscopic with CBD exploration
    }
    
    max_los = alos_limits.get(package_code, 3)
    
    if los <= max_los:
        return {
            "status": "PASS",
            "rule": "TMS_RULE_5_ALOS",
            "reason": f"Length of stay ({los} days) within ALOS limit ({max_los} days)"
        }
    else:
        return {
            "status": "CONDITIONAL",
            "rule": "TMS_RULE_5_ALOS",
            "reason": f"Stay exceeds ALOS ({los} days > {max_los} days), requires manual review"
        }
```

**ALOS Limits**:
- **Open Cholecystectomy (SG039A/B)**: ≤ 6 days
- **Laparoscopic (SG039C/D)**: ≤ 3 days

**Logic**: `IF (Discharge_Date - Admission_Date) ≤ ALOS_Limit → PASS`

**Failure**: Output CONDITIONAL (Reason: Stay exceeds ALOS, requires manual review)

---

## 5. Final Decision Logic

### Decision Tree

```python
def make_final_decision(document_checks, visual_checks, tms_rules):
    """
    Final decision logic for Cholecystectomy claims
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
    return {
        "decision": "PASS",
        "confidence": 0.95,
        "reasons": ["All document checks satisfied", "Visual evidence found", "All TMS rules passed"],
        "rule_flags": []
    }
```

### Pass Criteria
✅ All 9 required documents present
✅ All visual evidence detected
✅ TMS Rule 1 (Imaging) = PASS
✅ TMS Rule 2 (Clinical) = PASS
✅ TMS Rule 3 (Fraud) = PASS
✅ TMS Rule 4 (LFT) = PASS
✅ TMS Rule 5 (ALOS) = PASS

**Result**: Output PASS

### Conditional Criteria
⚠️ Any required document missing
⚠️ Any visual evidence missing
⚠️ TMS Rule 5 (ALOS) exceeds limit without documented complications

**Result**: Output CONDITIONAL (Requires manual review)

### Fail Criteria
❌ TMS Rule 1 (Imaging) = FAIL (No calculi in USG)
❌ TMS Rule 2 (Clinical) = FAIL (No mandatory symptoms)
❌ TMS Rule 3 (Fraud) = FAIL (Prior Cholecystectomy found)
❌ TMS Rule 4 (LFT) = FAIL (No elevated markers)

**Result**: Output FAIL

---

## 6. Implementation Checklist

### Document Classification Module
- [ ] Implement classifier for 9 document types
- [ ] Track missing documents
- [ ] Generate CONDITIONAL flag for missing docs

### OCR & NLP Module
- [ ] Extract symptom keywords from Clinical Notes
- [ ] Extract diagnostic keywords from USG/MRCP
- [ ] Extract lab values from LFT
- [ ] Extract medical history
- [ ] Extract admission and discharge dates
- [ ] Calculate length of stay

### Computer Vision Module
- [ ] Detect intraoperative photographs
- [ ] Detect gross specimen pictures
- [ ] For CBD packages: Detect CBD stone removal evidence

### Rules Engine Module
- [ ] Implement TMS Rule 1 (Imaging Evidence)
- [ ] Implement TMS Rule 2 (Clinical Manifestation)
- [ ] Implement TMS Rule 3 (Fraud/History Check)
- [ ] Implement TMS Rule 4 (LFT Markers)
- [ ] Implement TMS Rule 5 (ALOS Validation)

### Decision Module
- [ ] Implement final decision logic
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
- **Extracted Text**: Actual text snippet used for rule

```python
evidence_template = {
    "rule": "TMS_RULE_1_IMAGING",
    "status": "PASS",
    "source_document": "USG_Upper_Abdomen.pdf",
    "page_number": 1,
    "bounding_box": [100, 200, 400, 250],
    "confidence": 0.92,
    "extracted_text": "Multiple calculi seen in gall bladder",
    "timestamp": "2026-04-25T10:30:00Z"
}
```

---

**Document**: STG_RULES_CHOLECYSTECTOMY.md
**Package Codes**: SG039A, SG039B, SG039C, SG039D
**Created**: April 25, 2026
**Status**: Complete Technical Specification
