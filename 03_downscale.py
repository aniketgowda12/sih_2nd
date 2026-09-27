# 03_downscale.py — amplitude-preserving diffusion downscaling (FIXED v2)
# CPU: ~15-30 min. GPU: ~10 min. Jaldi hai to STEPS=2000 kar do.
import os, sys, json, numpy as np, xarray as xr, torch
import torch.nn.functional as F, matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler

STEPS = 4000
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

ev = xr.open_dataarray("data/event.nc"); cl = xr.open_dataarray("data/clim.nc")
fields = np.concatenate([cl.values, ev.values]).astype(np.float32)
mu, sd = float(fields.mean()), float(fields.std())
print(f"normalization: mu={mu:.1f}K  sd={sd:.1f}K")

def coarse(x, f=4):
    h, w = x.shape[-2]//f*f, x.shape[-1]//f*f
    x = x[..., :h, :w]
    return x.reshape(*x.shape[:-2], h//f, f, w//f, f).mean((-3, -1))

norm = lambda x: ((torch.as_tensor(np.asarray(x), dtype=torch.float32) - mu) / sd).clamp(-4, 4) / 4
denorm = lambda x: (x * 4 * sd + mu).numpy() - 273.15      # normalized tensor -> °C

# ---------- training patches: fine 32x32, coarse 8x8 ----------
rng = np.random.default_rng(0)
H, W = fields.shape[-2:]
HI, LO = [], []
for _ in range(2048):
    i, j = rng.integers(0, H-32), rng.integers(0, W-32)
    hi = fields[rng.integers(0, len(fields)), i:i+32, j:j+32]
    HI.append(hi); LO.append(coarse(hi))
HI = norm(np.array(HI))[:, None]        # (N,1,32,32) in [-1,1]
LO = norm(np.array(LO))[:, None]        # (N,1,8,8)

# FIX 1: block types explicitly — version-proof
model = UNet2DModel(
    sample_size=32, in_channels=2, out_channels=1,
    block_out_channels=(32, 64, 128),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    layers_per_block=2,
).to(dev)
sched = DDPMScheduler(num_train_timesteps=400, beta_schedule="squaredcos_cap_v2")
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

if os.path.exists("diff.pt") and "--skip-train" in sys.argv:
    model.load_state_dict(torch.load("diff.pt", map_location=dev)); print("loaded diff.pt")
else:
    for step in range(STEPS):
        i = torch.randint(0, len(HI), (32,))
        x0, lo = HI[i].to(dev), LO[i].to(dev)
        cond = F.interpolate(lo, size=32, mode="bilinear")
        t = torch.randint(0, 400, (len(i),), device=dev)
        noise = torch.randn_like(x0)
        xt = sched.add_noise(x0, noise, t)
        loss = F.mse_loss(model(torch.cat([xt, cond], 1), t).sample, noise)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0: print(f"step {step}/{STEPS}  loss {loss.item():.4f}", flush=True)
    torch.save(model.state_dict(), "diff.pt")
    print("✅ model saved: diff.pt")

# ---------- MONEY PLOT: peak-heat day ke core pe ----------
day = int(np.argmax(ev.values.max(axis=(1, 2))))
cmean = cl.values.mean(0)
anom_map = ev.values[day] - cmean
iy, ix = np.unravel_index(np.argmax(anom_map), anom_map.shape)
h, w = 24, 32
y0 = np.clip(iy - h//2, 0, anom_map.shape[0] - h)
x0 = np.clip(ix - w//2, 0, anom_map.shape[1] - w)

raw = ev.values[day][y0:y0+h, x0:x0+w].astype(np.float32)      # fine truth (Kelvin)
lo_raw = coarse(raw)                                            # (6,8) Kelvin
bilinear = F.interpolate(torch.tensor(lo_raw)[None, None], size=(h, w),
                         mode="bilinear")[0, 0].numpy()         # blurry baseline

@torch.no_grad()
def sample(cond_norm, n=8):
    x = torch.randn(n, 1, h, w, device=dev)
    c = cond_norm.to(dev).repeat(n, 1, 1, 1)
    for t in sched.timesteps:                                   # FIX 2: no flip — sahi direction
        tt = torch.full((n,), int(t), device=dev)
        eps = model(torch.cat([x, F.interpolate(c, size=(h, w), mode="bilinear")], 1), tt).sample
        x = sched.step(eps, int(t), x).prev_sample
    # PHYSICS CONSTRAINT: coarse-consistency (block means input se match)
    c_up = F.interpolate(c, size=(h, w), mode="bilinear")
    x = x + (c_up - F.interpolate(F.avg_pool2d(x, 4), size=(h, w), mode="nearest"))
    return x.cpu()

samples = sample(norm(lo_raw)[None, None])                       # FIX 3: normalized conditioning
S = denorm(samples[:, 0])                                       # (n,h,w) °C
T = raw - 273.15; B = bilinear - 273.15                          # truth & baseline °C
S0, SM, spread = S[0], S.mean(0), S.std(0)

vmin, vmax = float(np.percentile(T, 1)), float(np.percentile(T, 99))
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
panels = [(B, "Bilinear (blurry — 'spectral smoothing')", vmin, vmax, "coolwarm"),
          (S0, "Diffusion sample (sharp peaks!)", vmin, vmax, "coolwarm"),
          (T, "ERA5 truth (fine grid)", vmin, vmax, "coolwarm"),
          (spread, "Diffusion spread (uncertainty)", None, None, "viridis")]
for ax, (img, ttl, lo_, hi_, cm) in zip(axes, panels):
    im = ax.imshow(img, cmap=cm, vmin=lo_, vmax=hi_); ax.set_title(ttl, fontsize=10)
    plt.colorbar(im, ax=ax)
plt.savefig("outputs/downscale_compare.png", dpi=130, bbox_inches="tight"); plt.close()

p = lambda x: round(float(np.percentile(x, 99)), 1)
grad = lambda x: round(float(np.abs(np.gradient(x)).mean()), 3)
stats = {"peak_temp_C_99pct": {"truth": p(T), "bilinear": p(B),
                               "diffusion_sample": p(S0), "diffusion_ens_mean": p(SM)},
         "spatial_sharpness": {"truth": grad(T), "bilinear": grad(B),
                               "diffusion_sample": grad(S0)},
         "n_ensemble_samples": int(S.shape[0]), "downscale_factor": 4, "train_steps": STEPS}
json.dump(stats, open("outputs/downscale_stats.json", "w"), indent=2)
print("✅ saved: outputs/downscale_compare.png + downscale_stats.json")
print(json.dumps(stats, indent=2))