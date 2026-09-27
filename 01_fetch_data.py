# 01_fetch_data.py — REAL ERA5 data, anonymous GCS se. Ek baar chalao (~10 min).
import os, xarray as xr, pandas as pd

os.makedirs("data", exist_ok=True)
VAR = "2m_temperature"   # 2m temperature — heatwave demo

def open_source():
    urls = [
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
        "gs://weatherbench2/datasets/era5/1959-2023_6h_1440x721.zarr",
    ]
    for u in urls:
        try:
            ds = xr.open_zarr(u, storage_options={"token": "anon"}, chunks={}, consolidated=True)
            if VAR in ds.data_vars:
                print("✅ source:", u); return ds[VAR]
        except Exception as e:
            print("❌ fail:", u, repr(e)[:80])
    raise RuntimeError("koi source nahi mila — internet check karo")

da = open_source()
if "level" in da.dims: da = da.isel(level=0)

# Region: NW/Central India (heatwave zone). Lat descending handle karo:
lat_desc = bool(da.latitude[0] > da.latitude[-1])
da = da.sel(latitude=slice(32, 20) if lat_desc else slice(20, 32),
            longitude=slice(68, 82))

# --- Climatology: 30 saal × (14 May–5 Jun), daily 12Z = 690 samples ---
ts = pd.DatetimeIndex([t for y in range(1991, 2021)
                       for t in pd.date_range(f"{y}-05-14", f"{y}-06-05", freq="D")]) \
     + pd.Timedelta(hours=12)
print("⏳ climatology download ho rahi hai... 5-15 min, chill karo")
cl = da.sel(time=ts).load()

# --- Event: 2020 North India heatwave (Churu 50°C!), daily 12Z ---
ev_t = pd.date_range("2020-05-18", "2020-05-28", freq="D") + pd.Timedelta(hours=12)
ev = da.sel(time=ev_t).load()

cl.to_netcdf("data/clim.nc"); ev.to_netcdf("data/event.nc")
print("✅ saved:", cl.shape, ev.shape)
print("🔥 peak temp in event:", round(float(ev.max()) - 273.15, 1), "°C")  # ~44-46 aana chahiye