"""R2 (docs/27): the session-persistence bridge routes after the split into
ui/routes/persistence.py.

The /load route is the one with a side effect the pre-split code reached for via
module globals: it must REPLACE the engine singleton (and rebuild the supervisor)
for the freshly loaded session.  That reassignment now lives in
notebook_bridge.install_loaded_engine, called from the route.  Structure-only
persistence tests exercise session_io.do_load directly and never drove the route,
so this pins the route-level behavior end-to-end.
"""
import asyncio
import os

import ui.notebook_bridge as nb
import ui.session_io as session_io
from ui.routes.persistence import (
    load_session, save_session, LoadRequest, SaveRequest,
)


def test_load_route_replaces_engine_singleton():
    async def _run():
        nb.start_engine(title="R2LoadRoute", mode="standard", language="ja")
        saved = None
        try:
            nb.get_engine().register_expression("a*x+b")
            saved = (await save_session(
                SaveRequest(filename="r2_load_route_test", overwrite=False)))["filename"]
            engine_before = nb.get_engine()

            resp = await load_session(LoadRequest(filename=saved))

            assert resp["ok"] is True
            assert resp["title"] == "R2LoadRoute"
            assert resp["step_count"] == 1
            # install_loaded_engine swapped the singleton and rebuilt the supervisor.
            assert nb.get_engine() is not engine_before
            assert nb.get_supervisor() is not None
        finally:
            if saved:
                _p = session_io.sessions_dir() / saved
                if _p.exists():
                    os.remove(_p)
            nb.stop_engine()

    asyncio.run(_run())


def test_load_route_missing_file_404():
    async def _run():
        nb.start_engine(title="R2LoadMissing", mode="standard", language="ja")
        try:
            from fastapi import HTTPException
            import pytest
            with pytest.raises(HTTPException) as exc:
                await load_session(LoadRequest(filename="does_not_exist_r2.json"))
            assert exc.value.status_code == 404
            # A failed existence check must NOT have torn down the engine.
            assert nb.get_engine() is not None
        finally:
            nb.stop_engine()

    asyncio.run(_run())
