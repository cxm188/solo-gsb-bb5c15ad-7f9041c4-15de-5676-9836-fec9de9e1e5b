from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .segmenter import Config, solve

app = FastAPI(title="字幕断句", version="1.1.0")

STATIC_DIR = Path(__file__).parent / "static"


class WordIn(BaseModel):
    text: str
    start: float
    end: float


class BoundaryIn(BaseModel):
    """词间隙边界约束: index = k 表示词 k-1 与词 k 之间的间隙(1 <= k <= len(words)-1)。"""
    index: int
    type: Literal["must_break", "no_break"]


class SegmentRequest(BaseModel):
    words: list[WordIn]
    phrases: list[list[int]] = Field(default_factory=list)
    shots: list[float] = Field(default_factory=list)
    boundaries: list[BoundaryIn] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)


@app.post("/api/segment")
def segment(req: SegmentRequest):
    cfg_fields = {k: v for k, v in req.config.items() if k in Config.__dataclass_fields__}
    cfg = Config(**cfg_fields)
    try:
        return solve([w.model_dump() for w in req.words], req.phrases, req.shots, cfg,
                     boundaries=[b.model_dump() for b in req.boundaries])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
