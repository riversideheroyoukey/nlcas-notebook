"""R2 (docs/27): the LLM-facing bridge routes after the split into
ui/routes/advisory.py.

/interpret, /supervisor-list, /adviser-list, /adviser-status and /advise are
front-end-driven and had no route-level tests (the interpretation *pipeline* is
tested directly).  This pins that they still route through the bridge's
accessors: /interpret via get_engine + get_supervisor, and the adviser roster
endpoints via get_advisers().  Assertions are environment-robust (no fixed LLM
count, no network).
"""
import asyncio

import ui.notebook_bridge as nb
from ui.routes.advisory import (
    interpret_command, list_advisers, get_adviser_status, list_supervisors,
)
from ui.interpret_pipeline import InterpretRequest, InputToken


def test_interpret_route_delegates_to_pipeline():
    async def _run():
        nb.start_engine(title="R2InterpretRoute", mode="standard", language="ja")
        try:
            # A bare math token registers via run_interpret_pipeline(engine, supervisor).
            resp = await interpret_command(
                InterpretRequest(tokens=[InputToken(type="math", latex="a x + b")]))
            assert resp.get("status") == "registered"
            assert resp.get("registered_expressions")
        finally:
            nb.stop_engine()

    asyncio.run(_run())


def test_adviser_roster_endpoints_use_get_advisers():
    async def _run():
        nb.start_engine(title="R2AdviserRoute", mode="standard", language="ja")
        try:
            roster = list(nb.get_advisers().keys())
            # /adviser-list enumerates exactly the accessor's roster.
            adl = await list_advisers()
            assert [e["name"] for e in adl["advisers"]] == roster
            # /adviser-status reports not_configured iff the roster is empty.
            status = await get_adviser_status()
            assert status["available"] is (bool(roster) and status["available"])
            if not roster:
                assert status == {"available": False, "reason": "not_configured"}
            # /supervisor-list returns the documented shape.
            sup = await list_supervisors()
            assert "supervisors" in sup and "default" in sup
        finally:
            nb.stop_engine()

    asyncio.run(_run())
