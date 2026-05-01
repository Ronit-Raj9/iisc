"""Run the notebook cells in-process against a mocked NHAclient to verify
the full pipeline works without spending any tokens.

Strategy:
- Replace the `from nha_client import NHAclient` import with a stub that
  returns canned Gemma-shaped JSON for each page based on filename hints.
- Execute every code cell in notebook order.
- After run_batch, validate the per-package JSON files against PACKAGE_SCHEMAS.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "solution_notebook.ipynb"


# ---------- mock NHAclient ----------------------------------------------------

def _mock_completion(model, messages, metadata=None):
    """Return a canned Gemma JSON payload based on filename in the prompt."""
    fname = ""
    package = ""
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                if part.get("type") == "text":
                    txt = part.get("text", "")
                    if "package " in txt:
                        # find the package code
                        import re
                        match = re.search(r"package\s+([A-Z0-9]+)", txt)
                        if match:
                            package = match.group(1)
    # We don't actually have the filename here; classify based on package only.
    # For testing, return mostly-extra-document with one realistic doc per case.
    # The unit test below pre-seeds explicit cache files instead.
    payload = {
        "doc_type": "extra_document",
        "doc_type_confidence": 0.5,
        "ocr_snippet": "",
        "language": "en",
        "is_blurry": 0,
        "visual_elements": {
            "has_stamp": 0, "has_signature": 0, "has_photo_evidence": 0,
            "has_implant_sticker": 0, "has_table": 0, "has_xray": 0,
        },
        "entities": {
            "patient_age": None, "dates_found": [], "doa": None, "dod": None,
            "hb_values": [], "temperature_celsius": None, "fever_duration_days": None,
            "diagnoses": [], "symptoms": [], "treatments": [],
        },
    }
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


class _MockNHAclient:
    def __init__(self, *args, **kwargs):
        pass

    def completion(self, model, messages, metadata=None):
        return _mock_completion(model, messages, metadata)


def _install_mocks():
    sys.modules["nha_client"] = types.SimpleNamespace(NHAclient=_MockNHAclient)
    # Some cells optionally import a databank widget; replace with a no-op.
    class _Widget:
        def display(self):
            pass
    sys.modules["databank_download_widget"] = types.SimpleNamespace(DatabankDownloadWidget=_Widget)
    # The notebook calls display(...) — provide a fallback that just prints.
    import builtins
    if not hasattr(builtins, "display"):
        builtins.display = lambda x: print(x)


def _seed_cache(cache_root: Path, case_id: str, files_with_doc_type: list[tuple[str, str, dict]]):
    """Pre-populate vlm_cache so the pipeline doesn't call any model."""
    case_dir = cache_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    for file_name, doc_type, extras in files_with_doc_type:
        payload = {
            "doc_type": doc_type,
            "doc_type_confidence": 0.95,
            "ocr_snippet": extras.get("ocr_snippet", ""),
            "language": "en",
            "is_blurry": 0,
            "visual_elements": {
                "has_stamp": 0, "has_signature": 0, "has_photo_evidence": 0,
                "has_implant_sticker": 0, "has_table": 0, "has_xray": 0,
            } | extras.get("visual_elements", {}),
            "entities": {
                "patient_age": None, "dates_found": [], "doa": None, "dod": None,
                "hb_values": [], "temperature_celsius": None, "fever_duration_days": None,
                "diagnoses": [], "symptoms": [], "treatments": [],
            } | extras.get("entities", {}),
        }
        path = case_dir / f"{file_name}__p1.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


def _exec_notebook(test_dir: Path):
    with NB_PATH.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    g = {
        "__name__": "__main__",
        "__file__": str(NB_PATH),
    }

    # Each cell should run as if at notebook top level.
    cells_executed = 0
    cells_skipped = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        if isinstance(src, list):
            src = "".join(src)
        if not src.strip():
            cells_skipped += 1
            continue
        # Skip the dataset-download widget cell (no-op).
        if "DatabankDownloadWidget" in src and ".display" in src:
            cells_skipped += 1
            continue
        # Skip the original simple example-Sample.jpg cell (preserved by id),
        # but RUN our self-test cell which also references Sample.jpg.
        if 'with open("Sample.jpg"' in src:
            # Still need to define `nc` so later cells work.
            exec(
                "from nha_client import NHAclient\n"
                "import base64\n"
                "clientId = ''\n"
                "clientSecret = ''\n"
                "nc = NHAclient(clientId, clientSecret)\n",
                g,
            )
            cells_skipped += 1
            continue
        # Skip the example-jsons load (data/ folder doesn't exist locally).
        if "EXAMPLE_JSON_PATHS" in src or "load_example_jsons" in src:
            # Define stubs so subsequent cells importing example_jsons don't crash.
            g["EXAMPLE_JSON_PATHS"] = {}
            g["example_jsons"] = {}
            cells_skipped += 1
            continue
        if "for pkg, rows in example_jsons.items()" in src:
            cells_skipped += 1
            continue
        try:
            exec(src, g)
            cells_executed += 1
        except SystemExit:
            raise
        except Exception as e:
            print(f"  [{i:02d}] EXECUTION ERROR: {type(e).__name__}: {e}")
            first = src.split("\n")[0][:80]
            print(f"        first line: {first}")
            raise

    print(f"Cells executed: {cells_executed}, skipped: {cells_skipped}")
    return g


def _exec_real_data_run():
    """Second run: against the actual Dataset-1, with MAX_VLM_CALLS=0 so we
    walk all 40 cases without spending tokens.  Verifies discover_cases and
    schema validation across every package."""
    import shutil
    rd_g = {"__name__": "__main__"}
    with NB_PATH.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    # Strip our test fixture so we operate on Dataset-1 only
    fixture_dir = ROOT / "Claims" / "MG064A" / "TEST_CASE_001"
    if fixture_dir.exists():
        shutil.rmtree(fixture_dir, ignore_errors=True)

    # Override CONFIG cell to: (a) set MAX_VLM_CALLS = 0 (zero token spend)
    # (b) point DATA_ROOT at the real Dataset-1/Claims/ folder
    for i, cell in enumerate(nb["cells"]):
        if cell.get("id") == "7e47ecbe":
            src = cell["source"]
            if isinstance(src, list):
                src = "".join(src)
            import re as _re
            src = _re.sub(r"MAX_VLM_CALLS = \S+", "MAX_VLM_CALLS = 0", src)
            src = src.replace(
                'DATA_ROOT = Path("./Claims")',
                'DATA_ROOT = Path("./Dataset-1/Claims")',
            )
            cell["source"] = src

    cells_executed = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        if isinstance(src, list):
            src = "".join(src)
        if not src.strip():
            continue
        if "DatabankDownloadWidget" in src and ".display" in src:
            continue
        if "Sample.jpg" in src:
            exec("from nha_client import NHAclient\nclientId = ''\nclientSecret = ''\nnc = NHAclient(clientId, clientSecret)\n", rd_g)
            continue
        if "EXAMPLE_JSON_PATHS" in src or "load_example_jsons" in src:
            rd_g["EXAMPLE_JSON_PATHS"] = {}
            rd_g["example_jsons"] = {}
            continue
        if "for pkg, rows in example_jsons.items()" in src:
            continue
        try:
            exec(src, rd_g)
            cells_executed += 1
        except Exception as e:
            print(f"  [real-data run] cell [{i:02d}] failed: {type(e).__name__}: {e}")
            raise
    print(f"  Real-data cells executed: {cells_executed}")
    return rd_g


def main() -> int:
    _install_mocks()

    # Set CWD to the project root so DATA_ROOT=./Claims resolves to Dataset-1's content.
    import os
    os.chdir(ROOT)

    # Symlink Dataset-1/Claims to ./Claims temporarily for this test.
    claims_link = ROOT / "Claims"
    test_link_created = False
    if not claims_link.exists():
        target = ROOT / "Dataset-1" / "Claims"
        try:
            claims_link.symlink_to(target, target_is_directory=True)
            test_link_created = True
        except OSError:
            # fallback: copy a small subset of one case to ./Claims
            (claims_link / "MG064A" / "TEST_CASE").mkdir(parents=True, exist_ok=True)
            test_link_created = True

    # Pre-seed cache for one synthetic MG064A case so no real Gemma call fires.
    cache_root = ROOT / "vlm_cache"
    case_id = "TEST_CASE_001"

    # Test dir under DATA_ROOT
    test_case_dir = ROOT / "Claims" / "MG064A" / case_id
    test_case_dir.mkdir(parents=True, exist_ok=True)
    # Make a tiny JPEG so extract_pages has something to read
    try:
        from PIL import Image
        img = Image.new("RGB", (10, 10), "white")
        for fn in ("clinical.jpg", "cbc.jpg", "indoor.jpg", "treatment.jpg",
                   "post_hb.jpg", "discharge.jpg", "consent.jpg"):
            img.save(test_case_dir / fn)
    except Exception as e:
        print(f"failed to create test images: {e}")
        return 1

    _seed_cache(cache_root, case_id, [
        ("clinical.jpg", "clinical_notes", {
            "ocr_snippet": "Patient with pallor and fatigue, breathlessness on exertion. Diagnosis: severe anemia.",
        }),
        ("cbc.jpg", "cbc_hb_report", {
            "ocr_snippet": "CBC report\nHb 5.4 g/dL\nWBC 7000",
            "entities": {"hb_values": [5.4]},
        }),
        ("indoor.jpg", "indoor_case", {
            "ocr_snippet": "Patient admitted to medicine ward, IPD bed 12.",
        }),
        ("treatment.jpg", "treatment_details", {
            "ocr_snippet": "Blood transfusion 1 unit. Ferrous sulphate IV.",
            "entities": {"treatments": ["blood transfusion", "ferrous sulphate injection"]},
        }),
        ("post_hb.jpg", "post_hb_report", {
            "ocr_snippet": "Repeat Hb after transfusion: 9.6 g/dL",
            "entities": {"hb_values": [9.6]},
        }),
        ("discharge.jpg", "discharge_summary", {
            "ocr_snippet": "Patient discharged in stable condition.",
        }),
        ("consent.jpg", "extra_document", {
            "ocr_snippet": "Consent form for blood transfusion.",
        }),
    ])

    try:
        g = _exec_notebook(ROOT)
    finally:
        # Clean up the test fixture
        import shutil
        try:
            if test_case_dir.exists():
                shutil.rmtree(test_case_dir)
        except Exception:
            pass

    # Validate the produced output JSON
    out_dir = ROOT / "output"
    print(f"\nFiles in {out_dir}:")
    for p in sorted(out_dir.glob("*.json")):
        with p.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        print(f"  {p.name}: {len(rows)} rows")
        if rows:
            print(f"    first row keys: {list(rows[0].keys())}")
            print(f"    severe_anemia values: {[r.get('severe_anemia') for r in rows]}")

    # Schema check using the global validate_output_rows from the notebook
    if "validate_output_rows" in g:
        for pkg in ["MG064A", "SG039C", "MG006A", "SB039A"]:
            p = out_dir / f"{pkg}.json"
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    rows = json.load(f)
                ok, issues = g["validate_output_rows"](pkg, rows)
                status = "OK" if ok else "FAIL"
                print(f"  validate {pkg}: {status}")
                for i in issues[:2]:
                    print(f"    - {i}")

    # Run #2: real Dataset-1 with MAX_VLM_CALLS=0
    print("\n=== Real Dataset-1 walk (MAX_VLM_CALLS=0) ===")
    rd_g = _exec_real_data_run()
    cases_found = len(rd_g.get("PACKAGE_CODE_LOOKUP", {}))
    print(f"  cases discovered: {cases_found}")
    for pkg in ("MG064A", "SG039C", "MG006A", "SB039A"):
        p = out_dir / f"{pkg}.json"
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                rows = json.load(f)
            ok = True
            issues = []
            if "validate_output_rows" in rd_g:
                ok, issues = rd_g["validate_output_rows"](pkg, rows)
            print(f"  {pkg}.json: {len(rows)} rows, schema_ok={ok}")
            if not ok and issues:
                print(f"    - {issues[0]}")

    # Cleanup
    if test_link_created:
        try:
            claims_link.unlink()
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
