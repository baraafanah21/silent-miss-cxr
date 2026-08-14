"""Real-image experiment: selective COVID-19 classification from frozen deep features of real chest X-rays,
subgroups by patient sex (B = female, A = male). Repeated patient-level splits.

Scenarios
  real          : no intervention (the dataset as it is)
  label bias    : 35% of female COVID-19 positives relabelled negative in TRAINING labels only
  label bias, sex erased : same, after linearly erasing sex information from the embeddings (separability removed)
  view shift    : female AP-supine (portable ICU) images withheld from training/calibration, deployed only
  scanner shift : female deployment images altered (gamma, contrast, noise, blur) and re-encoded by the network
  label bias + scanner shift
Evaluation always uses the original labels. Usage: python3 realexp.py [n_splits]
"""
import sys, os, json, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results_real"); os.makedirs(OUT, exist_ok=True)
NSPLIT = int(sys.argv[1]) if len(sys.argv) > 1 else 300
TARGET_COV, U_B = 0.8, 0.35

F = np.load(os.path.join(ROOT, "features.npz")); P = np.load(os.path.join(ROOT, "features_pert.npz"))
m = pd.read_csv(os.path.join(ROOT, "meta_used.csv"))
keep = (m.finding != "todo").values & m.sex.isin(["M", "F"]).values
X = F["feats"][keep].astype(float); m = m[keep].reset_index(drop=True)
XP = {k: P[k][keep].astype(float) for k in P.files}
y = m.finding.str.contains("COVID-19").values
g = (m.sex == "F").values
grp = m.patientid.astype(str).values
supine = (m.view == "AP Supine").values
print(f"images {len(m)}, patients {len(np.unique(grp))}, female {g.sum()}, COVID {y.sum()}", flush=True)


# ---------- separability: how well do the embeddings predict sex? (patient-grouped CV) ----------
def sex_auc(Z):
    clf = LogisticRegression(C=0.1, max_iter=3000)
    p = cross_val_predict(clf, StandardScaler().fit_transform(Z), g, groups=grp, cv=GroupKFold(5), method="predict_proba")[:, 1]
    return roc_auc_score(g, p)


auc_raw = sex_auc(X)
print(f"sex AUC from embeddings (patient-grouped 5-fold CV): {auc_raw:.3f}", flush=True)


def metrics(pred, conf, tau, yy, gg, out, prefix=""):
    acc = conf >= tau
    for name, mk in (("A", ~gg), ("B", gg)):
        y_, p_, a_ = yy[mk], pred[mk], acc[mk]
        pos = max(y_.sum(), 1); fn = y_ & ~p_
        out[f"{prefix}abst_{name}"] = 1 - a_.mean()
        out[f"{prefix}fnr_{name}"] = fn.sum() / pos
        out[f"{prefix}smr_{name}"] = (fn & a_).sum() / pos
        out[f"{prefix}covpos_{name}"] = a_[y_].mean() if y_.any() else np.nan
        out[f"{prefix}selfnr_{name}"] = (fn & a_).sum() / max((y_ & a_).sum(), 1)
        out[f"{prefix}selrisk_{name}"] = (p_[a_] != y_[a_]).mean() if a_.any() else np.nan
    for k in ("abst", "fnr", "smr", "covpos", "selfnr", "selrisk"):
        out[f"{prefix}d_{k}"] = out[f"{prefix}{k}_B"] - out[f"{prefix}{k}_A"]


def selective(Ztr, ytr, Zcal, Zte, seed):
    sc = StandardScaler().fit(Ztr)
    clf = LogisticRegression(C=0.01, max_iter=5000).fit(sc.transform(Ztr), ytr)
    pc = clf.predict_proba(sc.transform(Zcal))[:, 1]; pt = clf.predict_proba(sc.transform(Zte))[:, 1]
    tau = np.quantile(np.maximum(pc, 1 - pc), 1 - TARGET_COV)
    return pt >= 0.5, np.maximum(pt, 1 - pt), tau, pt


gss1 = GroupShuffleSplit(n_splits=NSPLIT, test_size=0.2, random_state=20261018)
rows = []; t0 = time.time()
for s, (trcal, te) in enumerate(gss1.split(X, y, grp)):
    rng = np.random.default_rng([20261018, s])
    # split train/cal (75/25 of the remaining patients)
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=s)
    tr_i, cal_i = next(inner.split(trcal, y[trcal], grp[trcal])); tr, cal = trcal[tr_i], trcal[cal_i]
    flip = np.zeros(len(y), bool)
    flip[tr] = g[tr] & y[tr] & (rng.random(len(tr)) < U_B)
    ybias = y & ~flip
    row = {"split": s}
    # pseudo-group: random patients, same share as female patients -> not detectable from the images
    upat = np.unique(grp); fem_share = g.mean()
    rpat = set(rng.choice(upat, int(round(fem_share * len(upat))), replace=False))
    gr = np.array([p in rpat for p in grp])
    flip_r = np.zeros(len(y), bool); flip_r[tr] = gr[tr] & y[tr] & (rng.random(len(tr)) < U_B)
    for scen in ("real", "label bias", "real (random group)", "label bias (random group)", "view shift",
                 "scanner shift", "label bias + scanner shift"):
        Z = X
        G = gr if "random group" in scen else g
        ytr = (y & ~flip_r)[tr] if scen == "label bias (random group)" else (ybias[tr] if "label bias" in scen else y[tr])
        trS, calS, teS = tr, cal, te
        if scen == "view shift":                       # female supine images never seen before deployment
            trS = tr[~(g[tr] & supine[tr])]; calS = cal[~(g[cal] & supine[cal])]
            moved = np.concatenate([tr[g[tr] & supine[tr]], cal[g[cal] & supine[cal]]])
            teS = np.concatenate([te[~(g[te] & ~supine[te])], moved])   # B deployed = supine only
            ytr = y[trS]
        Zte = Z[teS].copy()
        if "scanner" in scen:
            Zte[g[teS]] = XP["strong"][teS][g[teS]]
        pred, conf, tau, pt = selective(Z[trS], ytr, Z[calS], Zte, s)
        out = {}
        metrics(pred, conf, tau, y[teS], G[teS], out)
        try:
            out["auc_A"] = roc_auc_score(y[teS][~G[teS]], pt[~G[teS]]); out["auc_B"] = roc_auc_score(y[teS][G[teS]], pt[G[teS]])
        except ValueError:
            out["auc_A"] = out["auc_B"] = np.nan
        out["n_B"] = int(G[teS].sum()); out["npos_B"] = int((G[teS] & y[teS]).sum())
        rows.append(dict(split=s, scenario=scen, **out))
    if s % 50 == 0:
        print(s, f"{time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "real_splits.csv"), index=False)
json.dump(dict(n_images=int(len(m)), n_patients=int(len(np.unique(grp))), n_female=int(g.sum()), n_covid=int(y.sum()),
               sex_auc_raw=auc_raw, n_splits=NSPLIT,
               target_cov=TARGET_COV, u_B=U_B), open(os.path.join(OUT, "real_config.json"), "w"), indent=1)
cols = ["abst_A", "abst_B", "d_abst", "fnr_A", "fnr_B", "d_fnr", "smr_A", "smr_B", "d_smr", "selrisk_A", "selrisk_B", "d_selrisk", "auc_A", "auc_B"]
S = R.groupby("scenario")[cols].mean() * 100
S[["auc_A", "auc_B"]] /= 100
print(S.round(1).to_string())
print(R.groupby("scenario")[["n_B", "npos_B"]].mean().round(1))
