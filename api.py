# api.py — chalao:  uvicorn api:app --port 8000   phir browser: http://localhost:8000/docs
import json, numpy as np, xarray as xr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import core

app = FastAPI(title="XtremeCast Alerting API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
ALERTS = json.load(open("outputs/alerts.json"))

@app.get("/health")
def health(): return {"status": "ok", "n_alerts": len(ALERTS)}

@app.get("/alerts")
def get_alerts(date: str = None):
    """date=YYYY-MM-DD optional filter. Pinpoint coordinate + severity + 5km radius."""
    if not date: return ALERTS
    out = [a for a in ALERTS if a["valid_time"].startswith(date)]
    if not out: raise HTTPException(404, "no alerts for this date")
    return out

@app.post("/process/{date}")
def process(date: str):
    """LIVE: ek date ka poora pipeline chala ke fresh alert nikaalta hai (continuous ingestion demo)."""
    ev = xr.open_dataarray("data/event.nc"); cl = xr.open_dataarray("data/clim.nc")
    idx = [i for i, t in enumerate(ev.time.values) if str(t)[:10] == date]
    if not idx: raise HTTPException(404, "date cached event window me nahi hai")
    i = idx[0]
    clim_q, cmean, cstd = core.clim_stats(cl.values)
    ens = core.pseudo_ensemble(ev.values[i:i+1], cstd)
    e = core.efi(ens[:, 0], clim_q)
    anom = float((ens[:, 0].mean(0) - cmean).max())
    pt = core.track([e], ev.latitude.values, ev.longitude.values)[0]
    sev = core.severity(float(e.max()), anom)
    return {"date": date, "centroid": {"lat": round(pt[0], 3), "lon": round(pt[1], 3)},
            "peak_efi": round(float(e.max()), 3), "anomaly_c": round(anom, 1),
            "severity": sev or "none", "radius_km": 5}