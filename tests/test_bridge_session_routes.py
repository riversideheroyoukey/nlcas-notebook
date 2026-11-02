"""R2 (docs/27): the session / status / fork bridge routes after the split into
ui/routes/session.py.

These routes are driven by the front end and had no route-level tests.  The one
with a new seam is GET /status, which now reads the adviser roster through the
bridge's get_advisers() accessor instead of the _advisers module global; /fork
also exercises _save_fork_snapshot end-to-end.  This pins both.
"""
import asyncio
import os

import ui.notebook_bridge as nb
import ui.session_io as session_io
from ui.routes.session import (
    get_status, list_domains, plans_list, fork_branch, new_session,
    ForkRequest, NewSessionRequest,
)


def test_status_route_reads_adviser_roster_via_accessor():
    async def _run():
        await new_session(NewSessionRequest(title="R2StatusRoute", mode="standard", language="ja"))
        try:
            st = await get_status()
            assert st["title"] == "R2StatusRoute"
            assert st["lang"] == "ja"
            # /status builds its adviser list from get_advisers(); the names must
            # match the accessor's roster exactly (regardless of which is selected).
            assert [a["name"] for a in st["advisers"]] == list(nb.get_advisers().keys())
            # /domains is served from the same module and must return categories.
            assert "categories" in (await list_domains())
        finally:
            nb.stop_engine()

    asyncio.run(_run())


def test_fork_route_writes_snapshot():
    async def _run():
        await new_session(NewSessionRequest(title="R2ForkRoute", mode="standard", language="ja"))
        saved = None
        try:
            ex = nb.get_engine().register_expression("a+b")
            resp = await fork_branch(
                ForkRequest(step_id=ex.created_by_step, snapshot_filename="r2_fork_route_test"))
            assert resp["ok"] is True
            saved = resp["snapshot_file"]
            assert (session_io.sessions_dir() / saved).exists()
            # plans polling works on the live engine even with no plans.
            assert (await plans_list())["ok"] is True
        finally:
            if saved:
                _p = session_io.sessions_dir() / saved
                if _p.exists():
                    os.remove(_p)
            nb.stop_engine()

    asyncio.run(_run())
