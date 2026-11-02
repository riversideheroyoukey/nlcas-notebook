"""FastAPI sub-routers for the notebook bridge (R2, docs/27).

Each module here owns one endpoint family and exposes a prefix-less
``router = APIRouter()`` that ``ui/notebook_bridge.py`` mounts under
``/api/notebook`` via ``router.include_router(...)`` — so the public paths
are unchanged.  Handlers reach the engine singleton by lazily importing
``get_engine`` from ``ui.notebook_bridge`` inside the handler, keeping these
modules out of an import cycle with the bridge that mounts them.
"""
