# core.py — EFI, ensemble, tracking, severity (sirf numpy/scipy)
import numpy as np
from scipy import ndimage

PS = np.linspace(0.02, 0.98, 25)   # climatological percentile grid

def clim_stats(clim):
    """clim: (T,lat,lon) -> (quantiles (P,lat,lon), mean, std)"""
    q = np.quantile(clim, PS, axis=0)
    return q, clim.mean(0), clim.std(0)

def pseudo_ensemble(field, std, M=50, seed=0):
    """Production: NEPS-G ke 23 asli members. Demo: ERA5 + calibrated perturbations."""
    rng = np.random.default_rng(seed)
    ens = np.empty((M, *field.shape), dtype=np.float32)
    for m in range(M):
        for t in range(field.shape[0]):
            n = ndimage.gaussian_filter(rng.normal(size=field.shape[1:]), sigma=2.0)
            ens[m, t] = field[t] + 0.6 * std * (n / (n.std() + 1e-9))
    return ens

def efi(ens_t, clim_q):
    """ECMWF Extreme Forecast Index. ens_t:(M,lat,lon), clim_q:(P,lat,lon)"""
    F = (ens_t[:, None] <= clim_q[None]).mean(0)          # (P,lat,lon)
    w = 1.0 / np.sqrt(PS * (1 - PS))                      # tail weighting
    return np.clip((2/np.pi) * ((PS[:, None, None] - F) * w[:, None, None]).mean(0), -1, 1)

def track(efi_series, lats, lons, thr=0.6):
    """EFI-weighted centroid per timestep = anomaly ka trajectory"""
    pts = []
    for e in efi_series:
        w = np.maximum(e - thr, 0)
        if w.sum() < 1e-6:
            pts.append(None); continue
        ys, xs = np.mgrid[0:e.shape[0], 0:e.shape[1]]
        cy, cx = (w*ys).sum()/w.sum(), (w*xs).sum()/w.sum()
        pts.append((float(lats[0] + cy*(lats[1]-lats[0])),
                    float(lons[0] + cx*(lons[1]-lons[0]))))
    return pts

def severity(efi_max, anom_max):
    if efi_max >= 0.85 or anom_max >= 4.0: return "severe"
    if efi_max >= 0.70 or anom_max >= 2.5: return "moderate"
    if efi_max >= 0.55 or anom_max >= 2.0: return "low"
    return None