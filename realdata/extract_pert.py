"""Scanner-like acquisition shift on REAL images: re-extract deep features after changing the images
(gamma, contrast, noise, blur), for female patients only (subgroup B). Output: features_pert.npz"""
import os, time
import numpy as np, pandas as pd, torch, cv2
import torchxrayvision as xrv
from extract import load, model, COVID, ROOT  # same preprocessing and model

torch.set_num_threads(2)
m = pd.read_csv(os.path.join(ROOT, "meta_used.csv"))
LEVELS = {"strong": dict(gamma=1.8, contrast=0.75, noise=0.04, blur=3)}
rng = np.random.default_rng(20261018)


def perturb(x, gamma, contrast, noise, blur):
    """x: preprocessed image in [-1024, 1024], shape (1, 224, 224)."""
    z = (x[0] + 1024) / 2048.0                     # to [0, 1]
    z = np.clip(z, 0, 1) ** gamma                  # tone curve (different detector / processing)
    z = 0.5 + contrast * (z - 0.5)                 # lower contrast
    if blur:
        z = cv2.GaussianBlur(z.astype(np.float32), (blur, blur), 0)
    z = z + noise * rng.standard_normal(z.shape)   # quantum noise
    return (np.clip(z, 0, 1) * 2048 - 1024)[None].astype(np.float32)


out = {k: np.full((len(m), 1024), np.nan, np.float32) for k in LEVELS}
t0 = time.time()
with torch.no_grad():
    for i, (f, s) in enumerate(zip(m.filename, m.sex)):
        if s != "F":
            continue
        x = load(os.path.join(COVID, "images", f))
        for k, kw in LEVELS.items():
            out[k][i] = model.features2(torch.from_numpy(perturb(x, **kw))[None])[0].numpy()
        if i % 100 == 0:
            print(i, f"{time.time()-t0:.0f}s", flush=True)
np.savez_compressed(os.path.join(ROOT, "features_pert.npz"), **out)
print("done", f"{time.time()-t0:.0f}s")
