"""The developer-manual index links resolve, and it covers every technical doc.

Guards against the index going stale: every Markdown link in developer_manual.md
must point at an existing file, and every technical design doc in docs/ must be
referenced somewhere in the index.
"""
import re
import pathlib

_DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"
_MANUAL = _DOCS / "developer_manual.md"
_LINK = re.compile(r"\]\(([A-Za-z0-9_./-]+\.md)\)")

# docs that are NOT design/technical (covered as cross-references, not required to
# be in the technical TOC) — and the manual itself.
_NON_TECHNICAL = {
    "developer_manual.md", "user_guide.md", "operation_reference.md", "open_items.md",
}


def _links():
    return [m.group(1) for m in _LINK.finditer(_MANUAL.read_text(encoding="utf-8"))]


def test_all_links_resolve():
    for link in _links():
        target = (_MANUAL.parent / link).resolve()
        assert target.is_file(), f"broken link in developer_manual.md: {link}"


def test_every_technical_doc_is_indexed():
    linked = {pathlib.Path(l).name for l in _links()}
    for md in _DOCS.glob("*.md"):
        if md.name.startswith("scenario_") or md.name in _NON_TECHNICAL:
            continue
        assert md.name in linked, f"technical doc not referenced in the manual: {md.name}"
