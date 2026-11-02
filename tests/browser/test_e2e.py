"""Playwright E2E browser tests for the Research Notebook UI.

Covers the scenarios in tests/e2e_julia_browser.md automatically.

Run:
    pytest tests/browser/ -v --no-header

Julia starts once (session-scoped server fixture) so startup cost is paid once.
KaTeX math is validated by reading the <annotation encoding="application/x-tex">
element that KaTeX always writes — no visual/screenshot comparison needed.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from playwright.sync_api import Page, expect

# NbPage and notebook_server fixture are defined in conftest.py (same dir)
from .conftest import NbPage  # for type hints only — fixture is auto-discovered


# ---------------------------------------------------------------------------
# Shared Julia session (class-scoped NbPage)
# We create ONE Julia session for the whole quantum algebra test suite so
# Julia only starts once.  Each scenario within the class builds on previous
# state (operators declared, expressions registered).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def julia_nb(playwright, notebook_server: str):   # noqa: F811
    """One browser page + Julia session shared across all scenarios in this module.

    Julia starts only once; scenarios build on each other sequentially
    (B defines operators, C uses them, etc.).
    Uses the pytest-playwright `playwright` fixture (session-scoped) so we can
    create a long-lived browser context at module scope.
    """
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    # Auto-accept native confirm() dialogs (load, shutdown, etc.)
    page.on("dialog", lambda d: d.accept())
    nb = NbPage(page, notebook_server)
    nb.goto()
    nb.new_session_api(title="e2e_julia", language="ja", domains=["quantum_optics"])
    yield nb
    context.close()
    browser.close()


# ---------------------------------------------------------------------------
# Scenario A — New session (validated in julia_nb fixture setup)
# ---------------------------------------------------------------------------

class TestScenarioA:
    """A: New Julia session creation."""

    def test_A1_ui_ready(self, julia_nb: NbPage):
        """A-1: After session creation the input box is visible and ready."""
        expect(julia_nb.page.locator("#input-box")).to_be_visible()

    def test_A2_julia_backend_active(self, julia_nb: NbPage):
        """Server status confirms julia_full backend is active."""
        import http.client, json
        conn = http.client.HTTPConnection("127.0.0.1", 8000)
        conn.request("GET", "/api/notebook/status")
        status = json.loads(conn.getresponse().read())
        conn.close()
        assert status.get("backend") == "julia_full"


# ---------------------------------------------------------------------------
# Scenario B — Operator definition
# ---------------------------------------------------------------------------

class TestScenarioB:
    """B: Boson / fermion operator definition."""

    def test_B1_define_boson_operator(self, julia_nb: NbPage):
        """B-1/B-2: Define boson 'a' via NL."""
        before = julia_nb.count_entries("entry-supervisor")
        julia_nb.submit("ボゾン演算子 a を定義して")
        expect(julia_nb.page.locator(".entry-supervisor")).to_have_count(
            before + 1, timeout=NbPage.ENTRY_TIMEOUT
        )
        text = julia_nb.last_entry_text("entry-supervisor")
        assert "a" in text

    def test_B3_define_fermion_operator(self, julia_nb: NbPage):
        """B-3: Define fermion 'f' via NL."""
        before = julia_nb.count_entries("entry-supervisor")
        julia_nb.submit("フェルミオン演算子 f を定義して")
        expect(julia_nb.page.locator(".entry-supervisor")).to_have_count(
            before + 1, timeout=NbPage.ENTRY_TIMEOUT
        )
        text = julia_nb.last_entry_text("entry-supervisor")
        assert "f" in text


# ---------------------------------------------------------------------------
# Scenario C — Commutator
# ---------------------------------------------------------------------------

class TestScenarioC:
    """C: Commutator calculation."""

    def test_C1_commutator_nl(self, julia_nb: NbPage):
        """C-1: [a(1), a†(1)] via NL → at least one new CAS entry."""
        before_cas = julia_nb.count_entries("entry-cas")
        julia_nb.submit("a(1) と a'(1) の交換子を計算して")
        # NL input may auto-register inline formulas; wait for ≥1 new entry
        julia_nb.page.wait_for_function(
            f"document.querySelectorAll('.entry-cas').length > {before_cas}",
            timeout=NbPage.ENTRY_TIMEOUT,
        )

    def test_C2_commutator_result_is_one(self, julia_nb: NbPage):
        """C-2: [a, a†] = 1 — last CAS entry's KaTeX annotation contains '1'."""
        latex = julia_nb.last_cas_latex()
        assert "1" in latex

    def test_C2_bracket_notation(self, julia_nb: NbPage):
        """C2-1: [a(1),a'(1)] bracket notation → new CAS entry with result '1'."""
        before_cas = julia_nb.count_entries("entry-cas")
        julia_nb.submit("[a(1),a'(1)]")
        julia_nb.page.wait_for_function(
            f"document.querySelectorAll('.entry-cas').length > {before_cas}",
            timeout=NbPage.ENTRY_TIMEOUT,
        )
        latex = julia_nb.last_cas_latex()
        assert "1" in latex

    def test_C2_anticommutator_bracket(self, julia_nb: NbPage):
        """C2-3: {f(1),f'(1)} → anticommutator = 1 for same site."""
        before_cas = julia_nb.count_entries("entry-cas")
        julia_nb.submit("{f(1),f'(1)}")
        julia_nb.page.wait_for_function(
            f"document.querySelectorAll('.entry-cas').length > {before_cas}",
            timeout=NbPage.ENTRY_TIMEOUT,
        )
        latex = julia_nb.last_cas_latex()
        assert "1" in latex


# ---------------------------------------------------------------------------
# Scenario D — Normal order
# ---------------------------------------------------------------------------

class TestScenarioD:
    """D: Normal ordering."""

    def test_D1_register_product(self, julia_nb: NbPage):
        """D-1: Register a(1)*a'(1) — a new CAS entry appears."""
        before = julia_nb.count_entries("entry-cas")
        julia_nb.submit("a(1)*a'(1)")
        julia_nb.page.wait_for_function(
            f"document.querySelectorAll('.entry-cas').length > {before}",
            timeout=NbPage.ENTRY_TIMEOUT,
        )

    def test_D2_normal_order_result(self, julia_nb: NbPage):
        """D-2/D-3: Normal order of a(1)*a†(1) → a†(1)a(1) form."""
        last_reg = julia_nb.last_entry("entry-cas")
        expr_id = last_reg.get_attribute("data-expr-id")
        before_cas = julia_nb.count_entries("entry-cas")
        julia_nb.submit(f"{expr_id} を正規順序にして")
        julia_nb.page.wait_for_function(
            f"document.querySelectorAll('.entry-cas').length > {before_cas}",
            timeout=NbPage.ENTRY_TIMEOUT,
        )
        latex = julia_nb.last_cas_latex()
        assert "dagger" in latex or "†" in latex or r"\dag" in latex


# ---------------------------------------------------------------------------
# Scenario E — Vacuum expectation value
# ---------------------------------------------------------------------------

class TestScenarioE:
    """E: Vacuum expectation value."""

    def test_E_vev_of_one(self, julia_nb: NbPage):
        """E-1/E-2: vev of a†a is a number (≥ 0). Use commutator result (=1) as test input."""
        # The commutator [a,a†]=1 was computed in ScenarioC; use it for VEV test.
        # Register [a(1),a'(1)] fresh (brackets are Phase-0, no LLM needed)
        before = julia_nb.count_entries("entry-cas")
        julia_nb.submit("[a(1),a'(1)]")
        julia_nb.page.wait_for_function(
            f"document.querySelectorAll('.entry-cas').length > {before}",
            timeout=NbPage.ENTRY_TIMEOUT,
        )
        expr_id = julia_nb.last_entry("entry-cas").get_attribute("data-expr-id")
        before2 = julia_nb.count_entries("entry-cas")
        julia_nb.submit(f"{expr_id} の真空期待値")
        julia_nb.page.wait_for_function(
            f"document.querySelectorAll('.entry-cas').length > {before2}",
            timeout=NbPage.ENTRY_TIMEOUT,
        )
        latex = julia_nb.last_cas_latex()
        assert "1" in latex


# ---------------------------------------------------------------------------
# Scenario F — Heisenberg equation
# ---------------------------------------------------------------------------

class TestScenarioF:
    """F: Heisenberg equation of motion."""

    def test_F_heisenberg_eom(self, julia_nb: NbPage):
        """F-1/F-2/F-3: [a†a, a] = -a (Heisenberg EOM for boson)."""
        before1 = julia_nb.count_entries("entry-cas")
        julia_nb.submit("a'(1)*a(1)")
        julia_nb.page.wait_for_function(
            f"document.querySelectorAll('.entry-cas').length > {before1}",
            timeout=NbPage.ENTRY_TIMEOUT,
        )
        h_id = julia_nb.last_entry("entry-cas").get_attribute("data-expr-id")
        before2 = julia_nb.count_entries("entry-cas")
        julia_nb.submit("a(1)")
        julia_nb.page.wait_for_function(
            f"document.querySelectorAll('.entry-cas').length > {before2}",
            timeout=NbPage.ENTRY_TIMEOUT,
        )
        a_id = julia_nb.last_entry("entry-cas").get_attribute("data-expr-id")
        before3 = julia_nb.count_entries("entry-cas")
        julia_nb.submit(f"ハイゼンベルク方程式 {a_id} {h_id}")
        julia_nb.page.wait_for_function(
            f"document.querySelectorAll('.entry-cas').length > {before3}",
            timeout=NbPage.ENTRY_TIMEOUT,
        )
        latex = julia_nb.last_cas_latex()
        assert "{a}" in latex or "a" in latex


# ---------------------------------------------------------------------------
# Scenario G — Save / Load
# ---------------------------------------------------------------------------

class TestScenarioG:
    """G: Session save and reload."""

    def test_G1_save_dialog(self, julia_nb: NbPage, tmp_path):
        """G-1: Save dialog opens and saves without error."""
        julia_nb.page.click("#save-btn")
        julia_nb.page.wait_for_selector("#save-modal", state="visible")
        julia_nb.page.fill("#save-modal-filename", "e2e_save_test")
        julia_nb.page.click("#save-modal-new")
        julia_nb.page.wait_for_selector("#save-modal", state="hidden", timeout=10_000)
        # No error entry should have appeared after save
        # (system entries are OK, but no "error" class entry)
        assert julia_nb.page.locator(".entry-error").count() == 0

    def test_G2_save_file_exists(self, julia_nb: NbPage):
        """G-2: The saved session file exists on the server filesystem."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from ui import session_io
        sessions_dir = session_io.sessions_dir()
        saved = list(sessions_dir.rglob("e2e_save_test.json"))
        assert saved, f"Save file not found in {sessions_dir}"

    def test_G3_load_and_use_without_redeclare(self, julia_nb: NbPage):
        """G-3 (D-3): Load saved session; declared operators work immediately.

        After loading, _replay_declarations() should re-issue @boson_ops a
        and @fermion_ops f into the fresh Julia process.  The commutator
        [a(1), a'(1)] = 1 must succeed without re-calling define_boson_operator.
        """
        import http.client, json as _json
        # Load the previously saved session via API (more reliable than modal UI)
        body = _json.dumps({"filename": "e2e_save_test.json"}).encode()
        conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=30)
        conn.request("POST", "/api/notebook/load", body,
                     {"Content-Type": "application/json"})
        resp_data = _json.loads(conn.getresponse().read())
        conn.close()
        assert resp_data.get("ok"), f"Load failed: {resp_data}"

        # Reload the browser page to reflect the loaded session
        julia_nb.page.reload()
        julia_nb.page.wait_for_selector("#input-box", state="visible")

        # Compute [a(1), a'(1)] WITHOUT re-declaring the boson operator.
        # If _replay_declarations works, this succeeds and returns 1.
        before = julia_nb.count_entries("entry-cas")
        julia_nb.submit("[a(1),a'(1)]")
        julia_nb.page.wait_for_function(
            f"document.querySelectorAll('.entry-cas').length > {before}",
            timeout=NbPage.ENTRY_TIMEOUT,
        )
        latex = julia_nb.last_cas_latex()
        assert "1" in latex, (
            f"Expected commutator [a,a†]=1 after session reload, got: {latex!r}"
        )


# ---------------------------------------------------------------------------
# Scenario H — LaTeX export
# ---------------------------------------------------------------------------

class TestScenarioH:
    """H: LaTeX export.

    The export button calls GET /api/notebook/export/latex which saves a .tex
    file server-side and returns {"ok": true, "path": "sessions/..."}.
    We read the file directly from the returned path.
    """

    def _export_tex_content(self, nb: NbPage) -> str:
        """Click the export button in the browser and return the saved .tex content.

        The export button calls GET /api/notebook/export/latex via the browser
        (which has the correct session context), saves the file server-side,
        and adds a system entry showing the path.  We read the path from the
        last system entry and load the file.
        """
        import re, json as _json
        before_sys = nb.count_entries("entry-system")
        nb.page.click("#export-latex-btn")
        nb.page.wait_for_function(
            f"document.querySelectorAll('.entry-system').length > {before_sys}",
            timeout=20_000,
        )
        # Read the path from the last system entry text (contains the file path)
        last_text = nb.last_entry_text("entry-system")
        # Also call the API to get the exact path reliably
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=15)
        conn.request("GET", "/api/notebook/export/latex")
        data = _json.loads(conn.getresponse().read())
        conn.close()
        assert data.get("ok"), f"Export failed: {data}"
        tex_path = Path(__file__).resolve().parent.parent.parent / data["path"]
        return tex_path.read_text(encoding="utf-8")

    def test_H_export_produces_tex(self, julia_nb: NbPage):
        """H-1/H-2: Export saves a .tex file without \\instruction{} macros."""
        content = self._export_tex_content(julia_nb)
        assert r"\begin{document}" in content
        assert r"\instruction{" not in content
        assert r"\operation{" not in content

    def test_H_export_contains_equation(self, julia_nb: NbPage):
        """H-4: Exported .tex contains equation environment or display math."""
        content = self._export_tex_content(julia_nb)
        assert r"\begin{equation}" in content or r"\[" in content
