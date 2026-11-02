"""Template-input contract tests (open_items I-1).

The template-input palette is a frontend feature integrated into the
handwriting screen's LaTeX Editor (ui/index.html, ui/main.js): palette buttons
insert LaTeX skeletons into the textarea, and the assembled LaTeX flows into
the SAME pipeline entry as direct typing / handwriting
(BackendHandler.latex_to_canonical).  The LaTeX strings are authored in
``ui/math_templates.json`` (the single source of truth shared by the JS
palette).

These tests guard the contract that *every* LaTeX form the palette can produce
is one the Maxima converter accepts — so a user clicking a template never ends
up with un-parseable input.  This is the Python side of the "representative
template output LaTeX -> canonical round-trip" required by open_items I-1.
"""

import json
from pathlib import Path

import pytest

from notebook.runtime.maxima_latex_converter import MaximaLatexConverter

TEMPLATES_PATH = Path(__file__).resolve().parent.parent / "ui" / "math_templates.json"


@pytest.fixture(scope="module")
def templates():
    with open(TEMPLATES_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def converter():
    return MaximaLatexConverter()


def test_templates_file_well_formed(templates):
    assert templates.get("groups"), "expected non-empty 'groups'"
    assert templates.get("test_examples"), "expected non-empty 'test_examples'"
    for g in templates["groups"]:
        assert g.get("id"), "each group needs an internal 'id'"
        assert g.get("items"), "each group needs items"
        for it in g["items"]:
            assert it.get("label"), f"item missing label: {it}"
            assert it.get("latex"), f"item missing latex: {it}"


def test_example_latex_converts_without_error(templates, converter):
    """Every representative filled template parses to a non-empty canonical."""
    for ex in templates["test_examples"]:
        latex = ex["latex"]
        canonical = converter.to_canonical(latex)
        assert canonical and canonical.strip(), f"empty canonical for {latex!r}"
        if "expect_contains" in ex:
            assert ex["expect_contains"] in canonical, (
                f"{latex!r} -> {canonical!r} missing {ex['expect_contains']!r}"
            )


def test_matrix_uses_project_literal_convention(templates):
    """The matrix template must emit the [a,b;c,d] literal, not \\begin{pmatrix}
    (the Maxima converter does not accept LaTeX matrix environments)."""
    matrix_items = [
        it for g in templates["groups"] for it in g["items"]
        if ";" in it["latex"] and it["latex"].lstrip().startswith("[")
    ]
    assert matrix_items, "expected a matrix template item using [..;..]"
    for it in matrix_items:
        assert "pmatrix" not in it["latex"], (
            "matrix template must not use \\begin{pmatrix} (unsupported by Maxima)"
        )


def test_skeleton_placeholders_are_empty_braces(templates):
    """Structure templates expose placeholders as empty '{}' so the JS caret
    navigation (jump to next empty {}) has something to land on."""
    structure = next(g for g in templates["groups"] if g["id"] == "structure")
    labels_with_gap = {"□⁄□", "√□", "□ⁿ", "□ₙ", "Σ", "∫"}
    for it in structure["items"]:
        if it["label"] in labels_with_gap:
            assert "{}" in it["latex"], (
                f"{it['label']} skeleton should contain an empty '{{}}' placeholder"
            )


def test_no_translatable_strings_in_palette(templates):
    """The handwriting screen has no i18n loader; the palette must therefore
    expose only language-neutral glyphs (no group label text to translate)."""
    for g in templates["groups"]:
        assert "label_key" not in g and "label" not in g, (
            "groups must not carry visible label text (i18n rule); use 'id'"
        )
