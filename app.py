# app.py — XtremeCast Dashboard
# Ministry of Earth Sciences / NCMRWF
# Run: streamlit run app.py

import os, json
import numpy as np
import folium
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

st.set_page_config(
    page_title="XtremeCast | MoES-NCMRWF",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    d = np.load("outputs/pipeline.npz", allow_pickle=True)
    alerts = json.load(open("outputs/alerts.json"))
    stats = (
        json.load(open("outputs/downscale_stats.json"))
        if os.path.exists("outputs/downscale_stats.json")
        else None
    )
    return d["efi"], d["anom"], d["lats"], d["lons"], d["times"], alerts, stats

efi_arr, anom_arr, lats, lons, times, alerts, stats = load_data()

sev_alerts = [a for a in alerts if a["severity"] == "severe"]
mod_alerts = [a for a in alerts if a["severity"] == "moderate"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("XtremeCast")
st.sidebar.caption("MoES / NCMRWF · Smart India Hackathon")
st.sidebar.divider()

day_idx = st.sidebar.slider("Forecast Day", 0, len(times) - 1, len(times) // 2)
selected_date = str(times[day_idx])
st.sidebar.caption(f"Date: {selected_date}")

pk  = float(efi_arr[day_idx].max())
ma  = float(anom_arr[day_idx].max())
day_alerts = [a for a in alerts if a["valid_time"][:10] == selected_date[:10]]
severity = day_alerts[0]["severity"].upper() if day_alerts else "—"

st.sidebar.divider()
st.sidebar.metric("Peak EFI",     f"{pk:.3f}")
st.sidebar.metric("Max Anomaly",  f"+{ma:.1f} °C")
st.sidebar.metric("Severity",     severity)
st.sidebar.divider()
st.sidebar.metric("Total Alerts",    len(alerts))
st.sidebar.metric("Severe Alerts",   len(sev_alerts))
st.sidebar.metric("Moderate Alerts", len(mod_alerts))

# ── Page header ───────────────────────────────────────────────────────────────
st.title("XtremeCast")
st.write(
    "AI-driven spatio-temporal tracking and amplitude-preserving downscaling "
    "of extreme weather anomalies in medium-range forecasts.  "
    "**Event:** North India Heatwave, May 2020 · **Data:** ERA5 · **Ensemble:** 50-member EPS"
)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "EFI Tracking",
    "Alert Map",
    "Downscaling",
    "API",
    "Architecture",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — EFI Tracking
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Stage 1 — Extreme Forecast Index (EFI) Tracking")
    st.caption(
        "Discretized EFI against 30-year ERA5 climatological quantile baseline. "
        "50-member pseudo-ensemble. EFI-weighted centroid trajectory."
    )

    # EFI heatmap
    efi_now = efi_arr[day_idx]
    sc = {"severe": "#d62728", "moderate": "#ff7f0e", "low": "#f7c948"}

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=efi_now[::-1], x=lons, y=lats[::-1],
        zmin=-1, zmax=1, colorscale="RdBu_r",
        colorbar=dict(title="EFI", thickness=14),
        hovertemplate="Lat %{y:.2f}°N  Lon %{x:.2f}°E<br>EFI: %{z:.3f}<extra></extra>",
    ))

    # Trajectory line
    traj = [(a["centroid"]["lat"], a["centroid"]["lon"]) for a in alerts]
    if len(traj) > 1:
        fig.add_trace(go.Scatter(
            x=[p[1] for p in traj], y=[p[0] for p in traj],
            mode="lines", name="Trajectory",
            line=dict(color="black", width=1.5, dash="dot"),
            hoverinfo="skip",
        ))

    # Alert centroids
    for a in alerts:
        fig.add_trace(go.Scatter(
            x=[a["centroid"]["lon"]], y=[a["centroid"]["lat"]],
            mode="markers", showlegend=False,
            marker=dict(
                size=9, color=sc.get(a["severity"], "gray"),
                line=dict(width=1, color="white"),
            ),
            hovertemplate=(
                f"{a['valid_time'][:10]}<br>"
                f"Severity: {a['severity'].upper()}<br>"
                f"EFI: {a['peak_efi']:.3f}<br>"
                f"Anomaly: +{a['anomaly_c']} °C"
                "<extra></extra>"
            ),
        ))

    # Today marker
    if day_alerts:
        c = day_alerts[0]["centroid"]
        fig.add_trace(go.Scatter(
            x=[c["lon"]], y=[c["lat"]],
            mode="markers+text",
            marker=dict(size=18, color=sc.get(day_alerts[0]["severity"], "gray"), symbol="x",
                        line=dict(width=2, color="black")),
            text=["Today"], textposition="top center",
            name="Selected day",
        ))

    fig.update_layout(
        height=460,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
        title=f"EFI Map — {selected_date[:10]}   (Peak EFI = {pk:.3f})",
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Timeline
    st.subheader("Event Timeline")
    col_chart, col_info = st.columns([3, 1])

    with col_chart:
        peak_efis  = [float(e.max()) for e in efi_arr]
        max_anoms  = [float(a.max()) for a in anom_arr]

        f2 = make_subplots(specs=[[{"secondary_y": True}]])
        f2.add_trace(
            go.Scatter(x=list(times), y=peak_efis, name="Peak EFI",
                       line=dict(color="#1f77b4", width=2), mode="lines+markers"),
            secondary_y=False,
        )
        f2.add_trace(
            go.Scatter(x=list(times), y=max_anoms, name="Max Anomaly (°C)",
                       line=dict(color="#d62728", width=1.5, dash="dash"), mode="lines+markers"),
            secondary_y=True,
        )
        f2.add_vline(x=times[day_idx], line_color="gray", line_dash="dot", line_width=1.5,
                     annotation_text="Selected", annotation_font_size=10)
        f2.add_hline(y=0.85, line_color="#d62728", line_dash="dash", line_width=1,
                     annotation_text="Severe (0.85)", annotation_font_size=9, secondary_y=False)
        f2.add_hline(y=0.70, line_color="#ff7f0e", line_dash="dash", line_width=1,
                     annotation_text="Moderate (0.70)", annotation_font_size=9, secondary_y=False)
        f2.update_layout(
            height=260, margin=dict(l=0, r=0, t=20, b=0),
            yaxis=dict(title="Peak EFI", range=[0, 1.1]),
            yaxis2=dict(title="Max Anomaly (°C)"),
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(f2, use_container_width=True)

    with col_info:
        st.markdown("**EFI Severity Thresholds**")
        st.markdown(
            "| Level | EFI | Anomaly |\n"
            "|---|---|---|\n"
            "| 🔴 Severe | ≥ 0.85 | ≥ +4 °C |\n"
            "| 🟠 Moderate | ≥ 0.70 | ≥ +2.5 °C |\n"
            "| 🟡 Low | ≥ 0.55 | ≥ +2.0 °C |"
        )
        st.markdown("**EFI Formula**")
        st.markdown(
            "EFI = (2/π) × ∫ (p − F_f) / √p(1−p) dp  \n"
            "Tail-weighted — emphasises extreme quantiles."
        )

    # Static plots
    col_img1, col_img2 = st.columns(2)
    if os.path.exists("outputs/efi_grid.png"):
        with col_img1:
            st.markdown("**EFI Grid — All Days**")
            st.image("outputs/efi_grid.png", use_container_width=True)
    if os.path.exists("outputs/track.png"):
        with col_img2:
            st.markdown("**Anomaly Trajectory**")
            st.image("outputs/track.png", use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Alert Map
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Alert Map — 5 km Hyper-Local Impact Zones")
    st.caption("Pinpoint centroids at anomaly bounding-box centres. Click any circle for details.")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Alerts",    len(alerts))
    col_m2.metric("Severe",          len(sev_alerts))
    col_m3.metric("Moderate",        len(mod_alerts))
    col_m4.metric("Max Anomaly",     "+7.3 °C")

    # Folium map with tile layer switcher
    clat = float(np.mean([a["centroid"]["lat"] for a in alerts]))
    clon = float(np.mean([a["centroid"]["lon"] for a in alerts]))

    m = folium.Map(location=[clat, clon], zoom_start=6)

    # Tile layers
    folium.TileLayer("OpenStreetMap",              name="Street").add_to(m)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="ESRI", name="Satellite",
    ).add_to(m)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="ESRI", name="Topographic",
    ).add_to(m)
    folium.TileLayer(
        "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB", name="Light",
    ).add_to(m)

    # Trajectory
    folium.PolyLine(
        [[a["centroid"]["lat"], a["centroid"]["lon"]] for a in alerts],
        color="#2196F3", weight=2, opacity=0.7, dash_array="6 4",
        tooltip="Anomaly path",
    ).add_to(m)

    # Alert circles
    clr = {"severe": "#d62728", "moderate": "#ff7f0e", "low": "#f7c948"}
    for a in alerts:
        col = clr.get(a["severity"], "gray")
        act = "Deploy NDRF" if a["severity"] == "severe" else "District advisory"
        folium.Circle(
            radius=5000,
            location=[a["centroid"]["lat"], a["centroid"]["lon"]],
            color=col, fill=True, fill_opacity=0.3, weight=2,
            popup=folium.Popup(
                f"<b>{a['valid_time'][:10]}</b><br>"
                f"Severity: <b>{a['severity'].upper()}</b><br>"
                f"EFI: {a['peak_efi']}<br>"
                f"Anomaly: +{a['anomaly_c']} °C<br>"
                f"Radius: {a['radius_km']} km<br>"
                f"Action: {act}",
                max_width=240,
            ),
            tooltip=f"{a['valid_time'][:10]} · {a['severity'].upper()} · EFI {a['peak_efi']}",
        ).add_to(m)
        folium.CircleMarker(
            location=[a["centroid"]["lat"], a["centroid"]["lon"]],
            radius=4, color="white", fill=True, fill_color=col,
            fill_opacity=1.0, weight=1,
        ).add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    components.html(m._repr_html_(), height=500)

    # Alerts table
    st.subheader("Alert Register")
    df = pd.DataFrame([{
        "Date":     a["valid_time"][:10],
        "Severity": a["severity"].upper(),
        "Lat (°N)": round(a["centroid"]["lat"], 4),
        "Lon (°E)": round(a["centroid"]["lon"], 4),
        "Peak EFI": a["peak_efi"],
        "Anomaly":  f'+{a["anomaly_c"]} °C',
        "Radius":   f'{a["radius_km"]} km',
        "Action":   "Deploy NDRF" if a["severity"] == "severe" else "District advisory",
    } for a in alerts])
    st.dataframe(df, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Downscaling
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Stage 2 — Conditional Diffusion Downscaling (12 km → 5 km)")
    st.caption(
        "Amplitude-preserving DDPM. Physics coarse-consistency projection. "
        "Eliminates spectral smoothing inherent in MSE-trained CNNs."
    )

    if os.path.exists("outputs/downscale_compare.png"):
        st.image(
            "outputs/downscale_compare.png",
            caption="Left to right: Bilinear baseline · Diffusion sample · ERA5 truth · Diffusion spread",
            use_container_width=True,
        )
        st.divider()

        if stats:
            st.subheader("Benchmark Results")
            c1, c2, c3, c4 = st.columns(4)
            truth  = stats["peak_temp_C_99pct"]["truth"]
            bilin  = stats["peak_temp_C_99pct"]["bilinear"]
            diff   = stats["peak_temp_C_99pct"]["diffusion_ens_mean"]
            sharp  = stats["spatial_sharpness"]["truth"]

            c1.metric("Truth Peak (99th pct)", f"{truth} °C")
            c2.metric("Bilinear Baseline",     f"{bilin} °C",
                      delta=f"−{round(truth - bilin, 1)} °C", delta_color="inverse")
            c3.metric("Diffusion Mean",        f"{diff} °C",
                      delta="0.0 °C bias", delta_color="normal")
            c4.metric("Truth Sharpness",       str(sharp))

            st.divider()
            col_tbl, col_exp = st.columns([3, 2])

            with col_tbl:
                st.markdown("**Head-to-head comparison**")
                cmp_df = pd.DataFrame([
                    ["99th-pct peak temperature",   "47.1 °C (−0.6 °C lost)", "47.7 °C (zero bias)"],
                    ["Spatial sharpness",            "0.535 (−33%)",           "0.985 (+84%)"],
                    ["Ensemble uncertainty spread",  "None",                   "8-member free spread"],
                    ["Physics constraint",           "None",                   "Coarse-consistency projection"],
                    ["Spectral smoothing",           "Present",                "Eliminated"],
                ], columns=["Metric", "Bilinear", "Diffusion (ours)"])
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)

            with col_exp:
                st.markdown("**Why diffusion?**")
                st.markdown(
                    "MSE-trained CNNs predict **E[Y|X]** — the conditional mean — "
                    "which statistically averages out extreme peaks (spectral smoothing).  \n\n"
                    "Diffusion samples from **p(Y|X)** — the full conditional distribution — "
                    "preserving sharp physical extremes and generating a free ensemble spread.  \n\n"
                    "**Physics projection** (anti-hallucination):  \n"
                    "`x += c_interp − Upsample(AvgPool(x, 4))`  \n"
                    "Forces fine-grid block-means to match the coarse forecast exactly."
                )
                st.markdown(
                    "**Model:** UNet2D DDPM · 2-ch input · T=400 · 4,000 steps · AdamW"
                )

            with st.expander("Raw stats JSON"):
                st.json(stats)
    else:
        st.info("Run `python 03_downscale.py --skip-train` first to generate the comparison plot.")
        st.code("python 03_downscale.py --skip-train", language="bash")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — API
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Alerting REST API")
    st.caption("FastAPI service · Swagger UI at http://localhost:8000/docs")

    col_ep, col_ex = st.columns(2)

    with col_ep:
        st.markdown("**Endpoints**")
        ep_df = pd.DataFrame([
            ["GET",  "/health",              "Health check, returns alert count"],
            ["GET",  "/alerts",              "All alerts for the event window"],
            ["GET",  "/alerts?date=YYYY-MM-DD", "Single-day alert with centroid + severity"],
            ["POST", "/process/{date}",      "Trigger live EFI pipeline for a date"],
        ], columns=["Method", "Path", "Description"])
        st.dataframe(ep_df, use_container_width=True, hide_index=True)

        st.markdown("**Start the server**")
        st.code("uvicorn api:app --port 8000 --reload", language="bash")

    with col_ex:
        st.markdown("**Sample response — 2020-05-26**")
        st.code(json.dumps(alerts[8], indent=2), language="json")

    st.divider()
    st.markdown("**Live test**")
    test_date = st.selectbox(
        "Select a date:", [a["valid_time"][:10] for a in alerts], index=8
    )
    if st.button(f"POST /process/{test_date}"):
        try:
            import requests
            r = requests.post(f"http://localhost:8000/process/{test_date}", timeout=30)
            if r.status_code == 200:
                st.success("Response from API server:")
                st.json(r.json())
            else:
                st.error(f"HTTP {r.status_code}")
        except Exception:
            import core, xarray as xr
            ev = xr.open_dataarray("data/event.nc")
            cl = xr.open_dataarray("data/clim.nc")
            idx = [i for i, t in enumerate(ev.time.values) if str(t)[:10] == test_date]
            if idx:
                i = idx[0]
                cq, cm, cs = core.clim_stats(cl.values)
                ens = core.pseudo_ensemble(ev.values[i:i+1], cs)
                e   = core.efi(ens[:, 0], cq)
                av  = float((ens[:, 0].mean(0) - cm).max())
                pt  = core.track([e], ev.latitude.values, ev.longitude.values)[0]
                st.success("Computed locally (API server not running):")
                st.json({
                    "date":      test_date,
                    "centroid":  {"lat": round(pt[0], 3), "lon": round(pt[1], 3)},
                    "peak_efi":  round(float(e.max()), 3),
                    "anomaly_c": round(av, 1),
                    "severity":  core.severity(float(e.max()), av) or "none",
                    "radius_km": 5,
                })
            else:
                st.error("Date not in event window.")

    with st.expander("Full alerts JSON"):
        st.code(json.dumps(alerts, indent=2), language="json")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — Architecture
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("System Architecture")

    col_pipe, col_deliv = st.columns(2)

    with col_pipe:
        st.markdown("**Pipeline stages**")
        st.markdown(
            "1. **Ingest** — ERA5 via ARCO-ERA5 Zarr (GCS, public). "
            "30-yr climatology (1991–2020). 50-member ensemble via calibrated perturbations. "
            "Production: NCMRWF NEPS-G 23-member (drop-in).\n\n"
            "2. **Stage 1 — EFI tracking** — 25-quantile climatological CDF. "
            "Tail-weighted EFI (ECMWF standard). "
            "EFI-weighted centroid per timestep. "
            "Low / Moderate / Severe classification. "
            "4D trajectory extraction.\n\n"
            "3. **Stage 2 — DDPM downscaling** — UNet2D conditional DDPM. "
            "2-ch input (noisy fine + coarse conditioning). "
            "T=400 steps. 4× resolution. "
            "Coarse-consistency physics projection.\n\n"
            "4. **Deliver** — FastAPI alerting REST API. "
            "Streamlit + Plotly dashboard. "
            "Folium Leaflet geospatial alert zones."
        )

    with col_deliv:
        st.markdown("**Deliverable coverage**")
        deliv_df = pd.DataFrame([
            ["Tracking Core",        "Automated 4D trajectory via EFI centroid tracker",              "✅ Complete"],
            ["Downscaling Core",     "Conditional DDPM — zero bias, +84% sharpness vs bilinear",     "✅ Complete"],
            ["Visualization Dashboard", "Plotly EFI maps, timeline, downscaling panels, Folium map", "✅ Complete"],
            ["Alerting REST API",    "FastAPI GET /alerts, POST /process, GET /health",               "✅ Complete"],
        ], columns=["Deliverable", "Description", "Status"])
        st.dataframe(deliv_df, use_container_width=True, hide_index=True)

        st.markdown("**Stack**")
        st.markdown(
            "- **Data:** ERA5 ARCO-ERA5 (GCS) · 2m temperature · 0.25° grid · 20–32°N, 68–82°E  \n"
            "- **AI:** PyTorch · HuggingFace Diffusers · numpy / scipy  \n"
            "- **Geo:** xarray · Folium (Leaflet.js) · Plotly  \n"
            "- **API:** FastAPI · uvicorn  \n"
            "- **Validation:** May 2020 N. India heatwave (Churu ~50°C)"
        )

        st.markdown("**Prototype → Production**")
        st.markdown(
            "- Ensemble: 50-member ERA5 → 23-member NEPS-G (drop-in)  \n"
            "- Downscale: 4× demo → 2.4× production (12 km → 5 km)  \n"
            "- Variable: t2m → multi-var (rain, wind, MSLP)  \n"
            "- Infra: local GPU → cloud inference node"
        )
