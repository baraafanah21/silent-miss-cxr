"""Extract frozen deep features from real chest X-rays with a pretrained chest X-ray DenseNet-121
(torchxrayvision 'densenet121-res224-all', trained on NIH, PadChest, CheXpert, MIMIC-CXR, Google, OpenI, RSNA).

Input : covid/metadata.csv + covid/images/ (ieee8023 COVID-19 image data collection, cloned from GitHub)
Output: features.npz  (1024-d embeddings, model pathology outputs, metadata)
"""
import os, sys, time
import numpy as np, pandas as pd, torch, skimage.io, skimage.color
import torchxrayvision as xrv

ROOT = os.path.dirname(os.path.abspath(__file__))
COVID = os.path.join(ROOT, "covid")
torch.set_num_threads(2)

m = pd.read_csv(os.path.join(COVID, "metadata.csv"))
frontal = ["PA", "AP", "AP Supine", "AP semi erect", "AP Erect"]
m = m[(m.modality == "X-ray") & (m.view.isin(frontal)) & (m.folder == "images")].copy()
m = m[m.filename.map(lambda f: os.path.exists(os.path.join(COVID, "images", f)))].reset_index(drop=True)
print("frontal X-ray images found:", len(m), flush=True)

model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
tf = xrv.datasets.XRayCenterCrop()
rs = xrv.datasets.XRayResizer(224, engine="cv2")


def load(path):
    img = skimage.io.imread(path)
    if img.ndim == 3:
        img = skimage.color.rgb2gray(img[..., :3]) * 255 if img.shape[2] >= 3 else img[..., 0]
    img = img.astype(np.float32)
    maxval = 65535 if img.max() > 255 else 255
    img = xrv.datasets.normalize(img, maxval)
    img = img[None, ...]
    return rs(tf(img))


if __name__ == "__main__":
    feats, outs, keep = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, f in enumerate(m.filename):
            try:
                x = torch.from_numpy(load(os.path.join(COVID, "images", f)))[None]
            except Exception as e:
                print("skip", f, e); continue
            feats.append(model.features2(x)[0].numpy())
            outs.append(model(x)[0].numpy())
            keep.append(i)
            if i % 100 == 0:
                print(i, f"{time.time()-t0:.0f}s", flush=True)
    m = m.iloc[keep].reset_index(drop=True)
    np.savez_compressed(os.path.join(ROOT, "features.npz"), feats=np.array(feats), outs=np.array(outs),
                        pathologies=np.array(model.pathologies))
    m.to_csv(os.path.join(ROOT, "meta_used.csv"), index=False)
    print("done", len(m), f"{time.time()-t0:.0f}s")
