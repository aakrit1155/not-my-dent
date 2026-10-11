"""Not my Dent application."""
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.pipeline import runner
from app.routes.claims import router as claims_router

BASE = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the models in the background so the site is reachable while weights load.
    threading.Thread(target=runner.preload, daemon=True).start()
    yield


app = FastAPI(title="Not my Dent | Team Everest", version="0.2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")
app.include_router(claims_router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/healthz")
async def healthz():
    return {"status": "ok", **runner.STATUS}
