"""Shared fixtures for Playwright browser E2E tests.

These tests are SKIPPED by default (they need a live server + Julia + a browser
and are order-sensitive across the shared module).  Run them explicitly:

    pytest tests/browser/ --run-browser -v
    RUN_BROWSER=1 pytest tests/browser/

Requires:
    pip install playwright pytest-playwright
    playwright install chromium

The server is started once per pytest session so Julia only initialises once.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # project root
_BASE_URL = "http://127.0.0.1:8000"
_BROWSER_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Opt-in gate
#
# These Playwright E2E tests need a live server + Julia + a browser, and they
# share one server/engine across the whole module, so they are order- and
# timing-sensitive when run together (each class passes on its own).  They are
# meant to be run explicitly (see the module docstring), not as part of the
# routine unit-test suite — otherwise the aggregate run intermittently fails
# and masks real regressions.  Skip them by default; enable with the
# `--run-browser` flag or the RUN_BROWSER=1 environment variable.
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--run-browser", action="store_true", default=False,
        help="run the Playwright browser E2E tests under tests/browser/",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "browser: Playwright browser E2E test (opt-in via --run-browser)")


def pytest_collection_modifyitems(config, items):
    import os
    if config.getoption("--run-browser") or os.environ.get("RUN_BROWSER"):
        return
    skip = pytest.mark.skip(
        reason="browser E2E: run with --run-browser or RUN_BROWSER=1")
    for item in items:
        try:
            in_browser_dir = _BROWSER_DIR in Path(str(item.fspath)).resolve().parents \
                or Path(str(item.fspath)).resolve() == _BROWSER_DIR
        except Exception:
            in_browser_dir = "tests/browser" in str(item.fspath).replace("\\", "/")
        if in_browser_dir:
            item.add_marker(pytest.mark.browser)
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# Server lifecycle (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def notebook_server():
    """Start the FastAPI server as a subprocess for the whole test session."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "ui.app:app",
         "--host", "127.0.0.1", "--port", "8000", "--no-access-log"],
        cwd=_PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait until the server responds.
    # Use http.client directly to bypass HTTP_PROXY env var (127.0.0.1 is local).
    import http.client
    for _ in range(30):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=1)
            conn.request("GET", "/api/notebook/status")
            resp = conn.getresponse()
            if resp.status == 200:
                break
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
        time.sleep(1)
    else:
        proc.terminate()
        raise RuntimeError("Server did not start in time")
    yield _BASE_URL
    proc.terminate()
    proc.wait()


# ---------------------------------------------------------------------------
# NbPage: thin helper wrapping Playwright Page for notebook-specific actions
# ---------------------------------------------------------------------------

class NbPage:
    """Thin wrapper around Playwright Page for notebook interactions."""

    ENTRY_TIMEOUT = 20_000   # ms — allow time for Julia CAS to respond

    def __init__(self, page: Page, base_url: str) -> None:
        self.page = page
        self.base_url = base_url

    # --- Navigation --------------------------------------------------------

    def goto(self) -> None:
        self.page.goto(f"{self.base_url}/notebook.html")
        self.page.wait_for_selector("#input-box", state="visible")

    # --- Session creation --------------------------------------------------

    def new_session_api(
        self,
        title: str = "e2e_test",
        language: str = "ja",
        domains: list[str] | None = None,
    ) -> None:
        """Create a new session via the REST API (more reliable than modal UI).

        After the POST, reload the page so the timeline reflects the new session.
        """
        import http.client, json as _json
        body = _json.dumps({
            "title": title,
            "domains": domains or [],
            "language": language,
        }).encode()
        conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=10)
        conn.request("POST", "/api/notebook/session/new", body,
                     {"Content-Type": "application/json"})
        conn.getresponse().read()
        conn.close()
        # Reload the page to pick up the new session state
        self.page.reload()
        self.page.wait_for_selector("#input-box", state="visible")
        # Fetch the initial timeline
        self.page.evaluate("window.loadInitialSession && loadInitialSession()")
        time.sleep(0.5)  # brief wait for async render

    # --- Input / submit ----------------------------------------------------

    def submit(self, text: str) -> None:
        """Type text into the NL input and click submit."""
        box = self.page.locator("#input-box")
        box.click()
        box.fill("")          # clear first (contenteditable)
        box.type(text)
        self.page.click("#submit-btn")

    # --- Timeline helpers --------------------------------------------------

    def count_entries(self, css_class: str) -> int:
        return self.page.locator(f".{css_class}").count()

    def wait_for_new_entry(self, css_class: str, after_count: int) -> None:
        """Wait until at least one more entry of given class appears."""
        expect(self.page.locator(f".{css_class}")).to_have_count(
            after_count + 1, timeout=self.ENTRY_TIMEOUT
        )

    def last_entry(self, css_class: str):
        """Return the last timeline entry of the given class."""
        return self.page.locator(f".{css_class}").last

    def last_cas_latex(self) -> str:
        """Return the TeX source of the last CAS result's KaTeX annotation.

        KaTeX writes the original TeX into an MathML <annotation> element.
        We use text_content() because annotation is a MathML node (not HTMLElement)
        so inner_text() raises "Node is not an HTMLElement".
        """
        entry = self.last_entry("entry-cas")
        ann = entry.locator('annotation[encoding="application/x-tex"]').first
        return ann.text_content(timeout=self.ENTRY_TIMEOUT) or ""

    def last_entry_text(self, css_class: str) -> str:
        return self.last_entry(css_class).inner_text()

    # --- Confirmation dialog -----------------------------------------------

    def wait_for_confirm(self):
        return self.page.wait_for_selector(".confirm-btn", timeout=self.ENTRY_TIMEOUT)

    def click_confirm(self, label_contains: str | None = None) -> None:
        """Click the (first matching) confirmation button."""
        if label_contains:
            btn = self.page.locator(
                f".confirm-btn:has-text('{label_contains}')"
            ).first
        else:
            btn = self.page.locator(".confirm-btn").first
        btn.click()

    # --- Save / Load -------------------------------------------------------

    def save(self, filename: str) -> None:
        self.page.click("#save-btn")
        self.page.wait_for_selector("#save-modal", state="visible")
        self.page.fill("#save-modal-filename", filename)
        self.page.click("#save-modal-new")
        self.page.wait_for_selector("#save-modal", state="hidden", timeout=10_000)

    def load(self, filename: str) -> None:
        self.page.click("#load-btn")
        self.page.wait_for_selector("#load-modal", state="visible")
        file_item = self.page.locator(f"#load-modal-list div:has-text('{filename}')").first
        file_item.click()
        self.page.click("#load-modal-ok")
        self.page.wait_for_selector("#load-modal", state="hidden", timeout=10_000)
        self.page.wait_for_selector(".entry-system", timeout=30_000)

    # --- LaTeX export ------------------------------------------------------

    def export_latex(self) -> str:
        """Click LaTeX export and return the downloaded file content."""
        with self.page.expect_download() as dl_info:
            self.page.click("#export-latex-btn")
        download = dl_info.value
        path = download.path()
        return Path(path).read_text(encoding="utf-8")


@pytest.fixture
def nb(page: Page, notebook_server: str) -> NbPage:
    """Return an NbPage helper for a fresh browser page."""
    nb = NbPage(page, notebook_server)
    nb.goto()
    return nb
