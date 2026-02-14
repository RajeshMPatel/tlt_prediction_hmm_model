"""FastAPI application for presenting daily H5 probabilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DAILY_REPORTS_DIR = PROJECT_ROOT / "reports" / "daily"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

app = FastAPI(title="TLT VIX H5 Probability Dashboard")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/", response_class=HTMLResponse)
def today_card(request: Request) -> HTMLResponse:
    """Render main today-card view."""
    latest_path = DAILY_REPORTS_DIR / "latest_probabilities.json"
    payload = _read_json(latest_path)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "payload": payload,
            "has_data": bool(payload),
        },
    )


@app.get("/status", response_class=HTMLResponse)
def status_page(request: Request) -> HTMLResponse:
    """Render status/debug view for data freshness and diagnostics."""
    status_path = DAILY_REPORTS_DIR / "status.json"
    payload = _read_json(status_path)
    return templates.TemplateResponse(
        "status.html",
        {
            "request": request,
            "payload": payload,
            "has_data": bool(payload),
        },
    )
