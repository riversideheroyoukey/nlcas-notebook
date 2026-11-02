# ui/app.py
import os
import re
import sys
import uuid
import signal
import socket
import subprocess
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

# Windows' console defaults to the legacy codepage (e.g. cp932 for Japanese
# locales), not UTF-8 - printing a non-ASCII symbol (checkmarks, emoji in
# startup/progress messages) raises UnicodeEncodeError there and crashes the
# lifespan startup (e.g. katex_utils.ensure_katex_assets()'s progress prints).
# macOS/Linux terminals are UTF-8 by default already, so this branch never
# touches them.
if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# --- パスを最初に確立 ---
UI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UI_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

# ui/ と project root の両方を sys.path に追加（どちらから起動しても動くように）
for _p in (str(UI_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- 依存インポート（パス確立後）---
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles as _StaticFiles


class NoCacheStaticFiles(_StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


StaticFiles = NoCacheStaticFiles

from config.config_utils import get_setting, get_local_llm_server_config
from katex_utils import ensure_katex_assets
import vlm_client
import image_utils
import notebook_bridge
import llm_server_launcher

# Handle to a local LLM server the app started (Ollama by default; provider is
# config-driven via config/default_settings.json → local_llm_server).
llm_server_proc = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_server_proc

    # KaTeX 資産のローカル準備
    ensure_katex_assets()

    # Start the configured local LLM server (quietly) if it isn't already up.
    # Switching Ollama → vLLM / LocalAI is a config edit, not a code change.
    llm_server_proc = llm_server_launcher.start(get_local_llm_server_config())

    # NotebookEngine を起動（Maxima subprocess を開始）
    notebook_bridge.start_engine(title="Notebook Session")

    threading.Thread(
        target=lambda: (time.sleep(1.5), webbrowser.open("http://127.0.0.1:8000/notebook.html")),
        daemon=True,
    ).start()

    yield

    notebook_bridge.stop_engine()
    llm_server_launcher.stop(llm_server_proc)


app = FastAPI(lifespan=lifespan)
app.include_router(notebook_bridge.router)
tasks = {}


@app.get("/api/models")
async def list_models():
    def get_dirs(cat):
        p = MODELS_DIR / cat
        if not p.exists():
            return []
        result = []
        for d in p.iterdir():
            if not d.is_dir():
                continue
            cfg = d / "config.json"
            if cfg.exists():
                import json as _json
                status = _json.loads(cfg.read_text(encoding="utf-8")).get("_management", {}).get("status", "")
                if status == "deprecated":
                    continue
            result.append(d.name)
        return result

    def_engine = get_setting("vlm_target", "none")
    def_refiner = get_setting("refiner_target", "none")

    return {
        "engines": get_dirs("engines"),
        "refiners": get_dirs("refiners") + ["none"],
        "default_engine": def_engine,
        "default_refiner": def_refiner,
    }


@app.post("/api/recognize")
async def recognize(req: dict, bt: BackgroundTasks):
    tid = str(uuid.uuid4())
    tasks[tid] = {"status": "processing", "result": None}
    bt.add_task(run_task, tid, req)
    return {"task_id": tid}


async def run_task(tid, req):
    try:
        p_img = image_utils.preprocess_image(req['image'])
        domains = req.get('domains', [])
        res = await vlm_client.run_pipeline(req['engine_id'], req['refiner_id'], p_img, domains=domains)
        tasks[tid].update({"status": "completed", "result": res})
    except Exception as e:
        tasks[tid].update({"status": "failed", "error": str(e)})


@app.get("/api/vlm_result/{tid}")
async def get_res(tid: str):
    return tasks.get(tid, {"status": "not_found"})


@app.post("/api/shutdown")
async def shutdown():
    def delayed_exit():
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=delayed_exit).start()
    return {"status": "ok", "message": "Server is shutting down..."}


@app.get("/notebook.html", response_class=HTMLResponse)
async def serve_notebook_html():
    ts = int(time.time())
    content = (UI_DIR / "notebook.html").read_text(encoding="utf-8")
    content = re.sub(r'(notebook_main\.(?:js|css))(?:\?v=\d+)?', rf'\1?v={ts}', content)
    return HTMLResponse(content=content, headers={"Cache-Control": "no-store, must-revalidate"})


app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
