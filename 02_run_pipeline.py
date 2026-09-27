# 02_run_pipeline.py — EFI + tracking + alerts generate karo
import os, json, numpy as np, xarray as xr, matplotlib.pyplot as plt
import core

os.makedirs("outputs", exist_ok=True)
ev = xr.open_dataarray("data/event.nc"); cl = xr.open_dataarray("data/clim.nc")
lats, lons = ev.latitude.values, ev.longitude.values
times = ev.time.values

clim_q, cmean, cstd = core.clim_stats(cl.values)
ens = core.pseudo_ensemble(ev.values, cstd, M=50)
efi_series = [core.efi(ens[:, t], clim_q) for t in range(ens.shape[1])]
anom = ens.mean(0) - cmean                       # °C above normal
pts = core.track(efi_series, lats, lons)

alerts = []
for i, (e, p) in enumerate(zip(efi_series, pts)):
    if p is None: continue
    sev = core.severity(float(e.max()), float(anom[i].max()))
    if sev is None: continue
    alerts.append({
        "event": "heatwave",
        "valid_time": str(np.datetime_as_string(times[i], unit="s")),
        "centroid": {"lat": round(p[0], 3), "lon": round(p[1], 3)},
        "peak_efi": round(float(e.max()), 3),
        "anomaly_c": round(float(anom[i].max()), 1),
        "severity": sev, "radius_km": 5})
json.dump(alerts, open("outputs/alerts.json", "w"), indent=2)

np.savez("outputs/pipeline.npz", efi=np.array(efi_series), anom=anom,
         lats=lats, lons=lons,
         times=np.array([str(np.datetime_as_string(t, unit="s"))[:10] for t in times]))
print(f"✅ {len(alerts)} alerts generated")
for a in alerts: print(a["valid_time"][:10], a["severity"], a["centroid"], a["anomaly_c"], "°C")

# ---- static plots ----
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for i, ax in enumerate(axes.flat):
    if i >= len(efi_series): ax.axis("off"); continue
    im = ax.imshow(efi_series[i], vmin=-1, vmax=1, cmap="RdBu_r",
                   extent=[68, 82, 20, 32])
    ax.set_title(str(times[i])[:10], fontsize=9)
plt.colorbar(im, ax=axes, label="EFI"); plt.savefig("outputs/efi_grid.png", dpi=120)
tr = [p for p in pts if p]
plt.figure(figsize=(8, 6))
plt.imshow(efi_series[len(efi_series)//2], vmin=-1, vmax=1, cmap="RdBu_r",
           extent=[68, 82, 20, 32]); plt.colorbar(label="EFI")
plt.plot([p[1] for p in tr], [p[0] for p in tr], "k-o", ms=4, label="AI-tracked path")
plt.legend(); plt.title("Heatwave 2020 — automated anomaly track")
plt.savefig("outputs/track.png", dpi=120)
print("✅ plots saved")