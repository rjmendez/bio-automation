#!/usr/bin/env python3
"""
Topaz enhancement HTTP server.

POST /sharpen   multipart image file  -> enhanced JPEG (quality=97)
POST /upscale   multipart image file  -> 2x upscaled JPEG
POST /enhance   multipart image file + ?mode=stock|upscale  -> JPEG
GET  /health    -> {"status": "ok", "device": "...", "models": [...]}

Usage:
    python tools/topaz_serve.py [--host 0.0.0.0] [--port 8091] [--device AUTO]

Requires: fastapi uvicorn Pillow openvino

The server loads ModelCache once at startup and keeps models warm.
Designed to run on a dedicated GPU host on your LAN (Linux or Windows) as a persistent enhancement service.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from bio_automation.topaz import ModelCache, sharpen, upscale_2x

app = FastAPI(title="Topaz Enhancement Server")
_cache: ModelCache | None = None
_device: str = "AUTO"


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    img_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    return img_rgb[:, :, ::-1].copy()


def _bgr_to_jpeg_bytes(img_bgr: np.ndarray, quality: int = 97) -> bytes:
    img_rgb = img_bgr[:, :, ::-1]
    pil_out = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_out.save(buf, "JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


async def _read_image(file: UploadFile) -> np.ndarray:
    data = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(data))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}") from exc
    return _pil_to_bgr(pil_img)


@app.get("/health")
def health():
    try:
        import openvino as ov
        devices = ov.Core().available_devices
    except Exception:
        devices = ["unavailable"]

    loaded_models = list(_cache._lru.keys()) if _cache else []
    return {"status": "ok", "device": _device, "available_devices": devices, "models": loaded_models}


@app.post("/sharpen")
async def route_sharpen(file: UploadFile = File(...)):
    assert _cache is not None
    try:
        img_bgr = await _read_image(file)
        result = sharpen(img_bgr, _cache)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(content=_bgr_to_jpeg_bytes(result), media_type="image/jpeg")


@app.post("/upscale")
async def route_upscale(file: UploadFile = File(...)):
    assert _cache is not None
    try:
        img_bgr = await _read_image(file)
        result = upscale_2x(img_bgr, _cache)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(content=_bgr_to_jpeg_bytes(result), media_type="image/jpeg")


@app.post("/enhance")
async def route_enhance(file: UploadFile = File(...), mode: str = "stock"):
    assert _cache is not None
    if mode not in ("stock", "inat", "upscale"):
        raise HTTPException(status_code=400, detail=f"Unknown mode {mode!r}; choose stock|inat|upscale")
    try:
        img_bgr = await _read_image(file)
        if mode in ("stock", "inat"):
            result = sharpen(img_bgr, _cache)
        else:
            sharpened = sharpen(img_bgr, _cache)
            result = upscale_2x(sharpened, _cache)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(content=_bgr_to_jpeg_bytes(result), media_type="image/jpeg")


def main() -> None:
    global _cache, _device

    parser = argparse.ArgumentParser(description="Topaz enhancement HTTP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--device", default="AUTO",
                        help="OpenVINO device: CPU, GPU, AUTO (default: AUTO)")
    args = parser.parse_args()

    _device = args.device
    _cache = ModelCache(max_models=3, device=args.device)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
