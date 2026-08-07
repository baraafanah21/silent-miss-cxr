"""Extensions requested in review:
 (1) metric decomposition: silent-miss rate (SMR) vs FNR, selective FNR and positive-class coverage
 (2) mitigation: group-specific thresholds; per-group recalibration on a small clean-label audit set
 (3) robustness: non-random ("subtle-case") underdiagnosis, nonlinear features + MLP, other coverage targets

Reuses the data-generating process of sim.py. Usage: python3 sim2.py [reps]
Outputs -> results_sim2/
"""
import sys, os, time, zlib, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")
import sim

OUT = "results_sim2"
os.makedirs(OUT, exist_ok=True)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
D = sim.D


def observe(X, y, g, rng, p):
    """Recorded labels. flip='random' (as in sim.py) or 'subtle': the u_B share of B's positives
    with the weakest disease evidence (lowest x0 + noise) are recorded negative."""
    yo = y.copy()
    idx = np.where(g & y)[0]
    k = int(round(p["uB"] * len(idx)))
    if k > 0:
        if p.get("flip", "random") == "random":
            yo[g & y & (rng.random(len(y)) < p["uB"])] = False
        else:
            score = X[idx, 0] + 0.5 * rng.standard_normal(len(idx))
            yo[idx[np.argsort(score)[:k]]] = False
    flip = rng.random(len(y)) < p["rho"]
    return np.where(flip, ~yo, yo)


def shift_shared(X1, g1, X2, g2, rng, p):
    """Same contrast gain + translation (same random direction) applied to subgroup B in two sets."""
    if p["m"] == 0:
        return X1, X2
    u = rng.standard_normal(D); u /= np.linalg.norm(u)
    out = []
    for X, g in ((X1, g1), (X2, g2)):
        Xs = X.copy()
        Xs[g] = Xs[g] * (1 + p["gain"] * p["m"]) + p["kt"] * p["m"] * u
        out.append(Xs)
    return out


def make_warp(rng):
    W1 = rng.standard_normal((D, D)) / np.sqrt(D)
    return lambda X: np.tanh(X @ W1) * 2.0 + 0.3 * X


def fit(X, y, p, seed):
    if p["model"] == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed).fit(X, y)
    return LogisticRegression(C=1.0, max_iter=2000).fit(X, y)


def group_metrics(prefix, pred, conf, tau, yte, gte, out):
    acc = conf >= tau
    for name, mk in (("A", ~gte), ("B", gte)):
        yy, pr, ac = yte[mk], pred[mk], acc[mk]
        pos = yy.sum()
        fn = yy & ~pr
        out[f"{prefix}abst_{name}"] = 1 - ac.mean()
        out[f"{prefix}fnr_{name}"] = fn.sum() / pos                       # full-coverage FNR
        out[f"{prefix}smr_{name}"] = (fn & ac).sum() / pos                # silent-miss rate
        out[f"{prefix}covpos_{name}"] = ac[yy].mean()                      # P(accepted | y=1)
        out[f"{prefix}selfnr_{name}"] = (fn & ac).sum() / max((yy & ac).sum(), 1)  # P(miss | y=1, accepted)
        out[f"{prefix}selrisk_{name}"] = (pr[ac] != yy[ac]).mean() if ac.any() else np.nan
    for k in ("abst", "fnr", "smr", "covpos", "selfnr", "selrisk"):
        out[f"{prefix}d_{k}"] = out[f"{prefix}{k}_B"] - out[f"{prefix}{k}_A"]


def one_rep(p, ss):
    rng = np.random.default_rng(ss)
    warp = make_warp(rng) if p.get("nonlin") else (lambda X: X)
    Xtr, gtr, ytr = sim.gen(p["n_tr"], rng, p); ytr_o = observe(Xtr, ytr, gtr, rng, p)
    Xva, gva, yva = sim.gen(p["n_val"], rng, p)
    Xte, gte, yte = sim.gen(p["n_te"], rng, p)
    Xau, gau, yau = sim.gen(p["n_audit"], rng, p)                       # small clean-label audit set
    # the audit set comes from the deployment setting, so it receives the SAME shift as the test set
    Xte, Xau = shift_shared(Xte, gte, Xau, gau, rng, p)
    Xtr, Xva, Xte, Xau = warp(Xtr), warp(Xva), warp(Xte), warp(Xau)
    m = fit(Xtr, ytr_o, p, int(ss.generate_state(1)[0] % 2**31))
    q = 1 - p["target_cov"]
    pv = m.predict_proba(Xva)[:, 1]; cv = np.maximum(pv, 1 - pv)
    pt = m.predict_proba(Xte)[:, 1]; ct = np.maximum(pt, 1 - pt)
    out = {}
    # (a) baseline: one global threshold from the calibration set
    group_metrics("", pt >= 0.5, ct, np.quantile(cv, q), yte, gte, out)
    # (b) group-specific thresholds: each group abstains on 20% of its own calibration cases
    tau_g = np.where(gte, np.quantile(cv[gva], q), np.quantile(cv[~gva], q))
    group_metrics("gthr_", pt >= 0.5, ct, tau_g, yte, gte, out)
    # (c) per-group Platt recalibration on the clean audit set, then a global 20% threshold on audit confidence
    pa = m.predict_proba(Xau)[:, 1]
    lo = lambda z: np.log(np.clip(z, 1e-6, 1 - 1e-6) / (1 - np.clip(z, 1e-6, 1 - 1e-6)))
    pt2 = np.empty_like(pt); pa2 = np.empty_like(pa)
    for gval in (False, True):
        ma, mt = gau == gval, gte == gval
        if len(np.unique(yau[ma])) < 2:
            pt2[mt], pa2[ma] = pt[mt], pa[ma]; continue
        cal = LogisticRegression(C=1e3).fit(lo(pa[ma]).reshape(-1, 1), yau[ma])
        pt2[mt] = cal.predict_proba(lo(pt[mt]).reshape(-1, 1))[:, 1]
        pa2[ma] = cal.predict_proba(lo(pa[ma]).reshape(-1, 1))[:, 1]
    ca2 = np.maximum(pa2, 1 - pa2); ct2 = np.maximum(pt2, 1 - pt2)
    group_metrics("recal_", pt2 >= 0.5, ct2, np.quantile(ca2, q), yte, gte, out)
    return out


def run(p, reps, tag):
    rows = [one_rep(p, ss) for ss in np.random.SeedSequence([20261018, zlib.crc32(tag.encode())]).spawn(reps)]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    BASE = dict(sim.BASE, n_audit=600, flip="random", nonlin=False)
    ARCH = {"baseline": {}, "signal gap": dict(dB=1.8), "annotation bias": dict(sep=2.0, uB=0.35),
            "domain shift": dict(m=1.5), "annot+shift": dict(sep=2.0, uB=0.35, m=1.5),
            "all four": dict(sep=2.0, uB=0.35, m=1.5, dB=1.8)}
    VARIANTS = {"main (LR, random flips)": {},
                "subtle-case flips": dict(flip="subtle"),
                "nonlinear features + MLP": dict(nonlin=True, model="mlp"),
                "coverage 90%": dict(target_cov=0.9),
                "coverage 70%": dict(target_cov=0.7)}
    rows = []
    for vn, vkw in VARIANTS.items():
        reps = REPS if vkw.get("model") != "mlp" else max(REPS // 2, 20)
        for an, akw in ARCH.items():
            df = run(dict(BASE, **akw, **vkw), reps, f"{vn}|{an}")
            df["variant"], df["scenario"] = vn, an
            rows.append(df)
        print(f"{vn} done {time.time()-t0:.0f}s", flush=True)
    R = pd.concat(rows); R.to_csv(f"{OUT}/sim2.csv", index=False)
    json.dump(dict(base=BASE, arch=ARCH, variants=VARIANTS, reps=REPS), open(f"{OUT}/config.json", "w"), indent=1, default=str)
    print(f"all done {time.time()-t0:.0f}s")
