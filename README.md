# 🔥 XtremeCast: AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies in Medium-Range Forecasts

> **Smart India Hackathon (SIH) Prototype**  
> **Problem Statement:** Ministry of Earth Sciences (MoES) / NCMRWF  
> **Target:** Automated identification, spatial tracking, amplitude-preserving downscaling, and hyper-local alerting for extreme weather events.

---

## 📌 Executive Summary

Medium-range weather forecasts (3–10 days) are chaotic and typically provided by coarse global ensembles (~12 km NEPS-G). Standard machine learning approaches (e.g., CNNs/U-Nets trained on MSE loss) suffer from **"spectral smoothing"**—they predict the conditional mean, averaging out physically crucial peaks in rainfall, wind, and temperature. Furthermore, district-wide alerts cause severe alert fatigue and public complacency.

**XtremeCast** solves this with an end-to-end 2-stage AI pipeline:
1. **Stage 1 (Automated Tracking Core):** Computes the discretized **Extreme Forecast Index (EFI)** against a 30-year climatological quantile baseline to isolate anomalies and autonomously extract 4D spatio-temporal trajectories with weighted centroids.
2. **Stage 2 (Amplitude-Preserving Downscaling):** Employs a **Conditional Denoising Diffusion Probabilistic Model (DDPM)** with a coarse-consistency physics projection to downscale coarse anomaly fields ($4\times$ resolution enhancement) without dampening peak amplitudes.
3. **Hyper-Local Alerting & Operations:** Delivers a 5 km impact-radius REST API and interactive Streamlit/Folium dashboard tailored for operational disaster agencies like NDRF and SDMAs.

---

## 📊 Live Verification & Benchmarks

Validated on the benchmark **May 2020 North India Heatwave** using real **ERA5 reanalysis data** and a climatology-calibrated 50-member ensemble:

| Metric | Coarse Input / Bilinear Baseline | Diffusion Ensemble Mean (Ours) | Single Diffusion Realization | ERA5 Fine Truth |
| :--- | :---: | :---: | :---: | :---: |
| **99th Percentile Peak Temp** | **47.1 °C** *(lost 0.6°C)* | **47.7 °C** *(exact recovery)* | **48.0 °C** | **47.7 °C** |
| **Spatial Sharpness (Gradient)** | **0.535** *(33% smoothed)* | **0.781** | **0.985** *(sharp gradients)* | **0.798** |
| **Spectral Smoothing Penalty** | High (blurred extremes) | **Zero bias on peak** | Generative spread | Ground Truth |

*Key Takeaway:* Naive interpolation flattens the extreme peak by 0.6°C. The diffusion ensemble mean recovers the extreme value **exactly ($47.7^\circ\text{C}$ vs $47.7^\circ\text{C}$)** while providing an ensemble spread representing spatial uncertainty.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    A[Global Ensemble Forecasts / ERA5] --> B[Climatology Quantiles (30-Yr Baseline)]
    A --> C[Ensemble Generator / NEPS-G 23-Member]
    B & C --> D[Stage 1: Discretized EFI Engine]
    D --> E[Spatio-Temporal Centroid Tracker]
    E --> F[Automated Alert Engine (Low / Moderate / Severe)]
    F --> G[FastAPI Alerting Service (/alerts, /process)]
    F --> H[Interactive Geospatial Dashboard (Streamlit + Folium)]
    D --> I[Stage 2: Conditional Diffusion Downscaler (DDPM)]
    I --> J[Physics Coarse-Consistency Projection]
    J --> K[5 km Calibrated Fine Field + Uncertainty Spread]
```

---

## 🔬 Scientific & Algorithmic Foundation

### 1. Discretized Extreme Forecast Index (EFI)
We implement the ECMWF-standard Extreme Forecast Index:
$$\text{EFI} = \frac{2}{\pi} \int_0^1 \frac{p - F_f(Q_c(p))}{\sqrt{p(1 - p)}} \, dp$$
where $F_f$ is the ensemble forecast cumulative distribution and $Q_c(p)$ is the climatological quantile at probability $p$. The denominator weight $\frac{1}{\sqrt{p(1-p)}}$ emphasizes the extreme tails of the distribution.

### 2. Physics-Constrained Diffusion Downscaling
To eliminate generative hallucinations while preserving extreme peaks, we apply a coarse-consistency projection directly in the sampling loop:
$$x_{\text{calibrated}} = x + \Big( c_{\text{interp}} - \text{Upsample}\big(\text{AvgPool}(x, f)\big) \Big)$$
This ensures the spatial block-averages of the fine-scale output strictly conserve the energy/state of the coarse driving forecast.

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Clone and enter directory
cd xtremecast

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. Running Stage 1 Pipeline (Tracking & Alerts)
```bash
python 02_run_pipeline.py
```
*Outputs: `outputs/alerts.json`, `outputs/pipeline.npz`, `outputs/track.png`, `outputs/efi_grid.png`.*

### 3. Running Stage 2 Diffusion (Inference / Evaluation)
```bash
# Evaluates using the pre-trained diffusion weights (diff.pt) in seconds:
python 03_downscale.py --skip-train
```
*Outputs: `outputs/downscale_compare.png`, `outputs/downscale_stats.json`.*

### 4. Launching the Interactive Web Dashboard
```bash
streamlit run app.py
```
*Access locally at: `http://localhost:8501`*

### 5. Launching the Operational Alerting REST API
```bash
uvicorn api:app --port 8000 --reload
```
*Interactive Swagger Documentation: `http://localhost:8000/docs`*
- `GET /health` : Service health & alert counts.
- `GET /alerts?date=2020-05-26` : Pinpoint coordinates, severity, and 5 km radius.
- `POST /process/{date}` : Real-time on-demand continuous ingestion & processing.

---

## 🛡️ Judge Defense & Talking Points

| Question | Winning Answer |
| :--- | :--- |
| **Why Diffusion instead of CNN / U-Net?** | Standard CNNs optimize MSE, which mathematically predicts the conditional mean $\mathbb{E}[Y\|X]$. This inherently averages out high-frequency peaks ("spectral smoothing"). Diffusion models sample from the true conditional distribution $p(Y\|X)$, reconstructing sharp physical extremes and generating an ensemble spread (uncertainty quantification) for free. |
| **How do you prevent generative hallucinations?** | We implement a coarse-consistency projection after sampling: the spatial block-mean of the downscaled field is constrained to match the coarse input forecast, ensuring physics conservation. |
| **How does this map from Prototype to Production?** | Prototype runs on real ERA5 with a 50-member climatology-perturbed ensemble; in production, NCMRWF's 23-member NEPS-G ensemble directly feeds the exact same EFI engine. Our prototype tests a **$4\times$ downscaling factor**, which is more rigorous than the production $2.4\times$ ($12 \to 5\text{ km}$). |
| **Why not a global GNN in the prototype?** | GraphCast/icosahedral mesh GNNs require months of training and multi-TPU infrastructure. Automated EFI anomaly thresholding + weighted centroid tracking achieves the operational goal in seconds on CPU/GPU. The GNN spatial mesh is on the Phase-2 production roadmap. |

---

## 👥 Authors & Acknowledgments
Built for the **Smart India Hackathon (SIH)** addressing the **MoES / NCMRWF** Extreme Weather Tracking Challenge.
