"""HF Agentic Search - Gradio Space entrypoint."""
from __future__ import annotations

import json
import os
import sys
import threading
import uuid

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gradio as gr
from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.agent import MAX_TASK_LENGTH, weave, weave_events

app = gr.Server()
_DIST = os.path.join(_ROOT, "frontend", "dist")
_sessions: dict[str, dict] = {}
_lock = threading.Lock()


def _session_id(request: Request) -> str:
    supplied = (request.headers.get("X-Session-Id") or "").strip()
    return supplied[:80] if supplied else f"dw-{uuid.uuid4().hex[:16]}"


async def _task_from_request(request: Request) -> tuple[str | None, JSONResponse | None]:
    try:
        body = await request.json()
    except Exception:
        return None, JSONResponse({"error": "Body is not valid JSON."}, status_code=400)
    task = str(body.get("task") or "").strip()
    if not task:
        return None, JSONResponse({"error": "Task description is required."}, status_code=400)
    if len(task) > MAX_TASK_LENGTH:
        return None, JSONResponse(
            {"error": f"Task description must be {MAX_TASK_LENGTH} characters or fewer."},
            status_code=422,
        )
    return task, None


@app.post("/weave")
async def api_weave(request: Request):
    task, error = await _task_from_request(request)
    if error:
        return error
    sid = _session_id(request)
    try:
        result = weave(task or "")
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    with _lock:
        _sessions[sid] = result
    return JSONResponse(result, headers={"X-Session-Id": sid})


@app.post("/weave/stream")
async def api_weave_stream(request: Request):
    task, error = await _task_from_request(request)
    if error:
        return error
    sid = _session_id(request)

    def stream():
        try:
            for event in weave_events(task or ""):
                if event["type"] == "complete":
                    with _lock:
                        _sessions[sid] = event["result"]
                yield json.dumps(event, ensure_ascii=True) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"X-Session-Id": sid, "Cache-Control": "no-store"},
    )


@app.get("/state")
async def get_state(request: Request):
    sid = _session_id(request)
    with _lock:
        result = _sessions.get(sid)
    if result is None:
        result = {
            "datasets": [], "nodes": [], "threads": [], "task": "",
            "top_pick": None, "fallback_used": False,
        }
    return JSONResponse(result, headers={"X-Session-Id": sid})


_DIST_ASSETS = os.path.join(_DIST, "assets")
if os.path.isdir(_DIST_ASSETS):
    app.mount("/assets", StaticFiles(directory=_DIST_ASSETS), name="assets")


@app.get("/")
def index():
    path = os.path.join(_DIST, "index.html")
    if os.path.isfile(path):
        return FileResponse(path)
    return JSONResponse({"error": "Frontend not built."}, status_code=503)


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860"))),
        show_error=True,
    )
