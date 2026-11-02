"""The magnetic-resonance (NMR/ESR) domain config is well-formed and discoverable.

Added on the back of A-11 (single-spin Larmor models now work): the domain ships
the standard meta + supervisor/OCR hints, auto-discovered by directory name.
"""
import json
import pathlib

_DOMAINS = pathlib.Path(__file__).resolve().parent.parent / "domains"
_MR = _DOMAINS / "magnetic_resonance"


def test_meta_is_active_physics():
    meta = json.loads((_MR / "meta.json").read_text())
    assert meta["category"] == "physics"
    assert "磁気共鳴" in meta["label_ja"]
    assert "Magnetic Resonance" in meta["label_en"]
    # active (no status field): the rotating-frame transformation (F-4) is implemented
    assert "status" not in meta


def test_category_key_exists():
    cats = json.loads((_DOMAINS / "categories.json").read_text())
    assert "physics" in {c["key"] for c in cats}


def test_all_three_files_present():
    for f in ("meta.json", "supervisor_hint.txt", "ocr_hint.txt"):
        assert (_MR / f).is_file()


def test_supervisor_hint_covers_core_concepts():
    sup = (_MR / "supervisor_hint.txt").read_text()
    for token in ("Larmor", "ω0 = γ B0", "Zeeman", "commutator", "回転座標系"):
        assert token in sup


def test_ocr_hint_covers_symbols():
    ocr = (_MR / "ocr_hint.txt").read_text()
    assert "gyromagnetic" in ocr and "T_1, T_2" in ocr
