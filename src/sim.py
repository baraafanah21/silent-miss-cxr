"""Controlled simulation: what drives subgroup disparities under selective classification?

Factors (all manipulated independently):
  annot : annotation bias   -> true positives of subgroup B are recorded as negative with prob u_B (underdiagnosis)
  shift : domain shift      -> deployment features of subgroup B are translated/rescaled
  sep   : subgroup separability -> how linearly detectable group membership is from the features
  gap   : disease-signal gap -> weaker disease signal in subgroup B (full-coverage accuracy gap)

Usage: python3 sim.py [reps_factorial]
All outputs -> results_sim/
"""
import sys, os, time, itertools, json, zlib
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

OUT = "results_sim"
os.makedirs(OUT, exist_ok=True)
D = 20
BASE = dict(frac_B=0.2, prev=0.3, dA=2.5, dB=2.5, sep=0.0, uB=0.0, rho=0.03, m=0.0, mode="B", gain=0.15, kt=1.0,
            n_tr=4000, n_val=2000, n_te=4000, target_cov=0.8, model="lr")
REPS_F = int(sys.argv[1]) if len(sys.argv) > 1 else 200
REPS_D = max(REPS_F // 2, 20)
ROOT = np.random.SeedSequence(20261018)


def gen(n, rng, p):
    g = rng.random(n) < p["frac_B"]
    y = rng.random(n) < p["prev"]
    delta = np.where(g, p["dB"], p["dA"])
    X = rng.standard_normal((n, D))
    X[:, 0] += (y - 0.5) * delta          # disease axis
    X[:, 1] += g * p["sep"]               # subgroup axis (orthogonal to disease axis)
    return X, g, y


def observe(y, g, rng, p):
    yo = y.copy()
    yo[g & y & (rng.random(len(y)) < p["uB"])] = False      # underdiagnosis in B
    flip = rng.random(len(y)) < p["rho"]                      # uniform label noise
    return np.where(flip, ~yo, yo)


def shift(X, g, rng, p):
    if p["m"] == 0:
        return X
    u = rng.standard_normal(D); u /= np.linalg.norm(u)
    mask = g if p["mode"] == "B" else np.ones(len(g), bool)
    Xs = X.copy()
    Xs[mask] = Xs[mask] * (1 + p["gain"] * p["m"]) + p["kt"] * p["m"] * u   # contrast gain + translation
    return Xs


def fit(Xtr, ytr, p, seed):
    if p["model"] == "mlp":
        return MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=seed).fit(Xtr, ytr)
    return LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)


def one_rep(p, ss, curve=False):
    rng = np.random.default_rng(ss)
    Xtr, gtr, ytr = gen(p["n_tr"], rng, p); ytr_o = observe(ytr, gtr, rng, p)
    Xva, gva, yva = gen(p["n_val"], rng, p)
    Xte, gte, yte = gen(p["n_te"], rng, p); yte_o = observe(yte, gte, rng, p)
    Xte = shift(Xte, gte, rng, p)
    m = fit(Xtr, ytr_o, p, int(ss.generate_state(1)[0] % 2**31))
    conf_va = np.maximum(pv := m.predict_proba(Xva)[:, 1], 1 - pv)
    tau = np.quantile(conf_va, 1 - p["target_cov"])           # threshold fixed in the *training* domain
    pt = m.predict_proba(Xte)[:, 1]; conf = np.maximum(pt, 1 - pt); pred = pt >= 0.5
    acc = conf >= tau
    r = {}
    for name, mk in (("A", ~gte), ("B", gte)):
        yy, yo, pr, ac = yte[mk], yte_o[mk], pred[mk], acc[mk]
        pos = yy.sum()
        r[f"abst_{name}"] = 1 - ac.mean()
        r[f"acc_full_{name}"] = (pr == yy).mean()
        r[f"sens_full_{name}"] = (pr & yy).sum() / pos
        r[f"risk_sel_{name}"] = (pr[ac] != yy[ac]).mean() if ac.any() else np.nan
        r[f"risk_sel_obs_{name}"] = (pr[ac] != yo[ac]).mean() if ac.any() else np.nan   # what an audit with biased labels sees
        r[f"miss_full_{name}"] = (yy & ~pr).sum() / pos                              # missed cases, no abstention
        r[f"miss_silent_{name}"] = (yy & ~pr & ac).sum() / pos                       # missed AND accepted by the system
    for k in ("abst", "acc_full", "sens_full", "risk_sel", "risk_sel_obs", "miss_full", "miss_silent"):
        r[f"d_{k}"] = r[f"{k}_B"] - r[f"{k}_A"]
    r["risk_gap_full"] = (1 - r["acc_full_B"]) - (1 - r["acc_full_A"])
    r["magnification"] = r["d_risk_sel"] - r["risk_gap_full"]
    if curve:   # risk-coverage curves per group (oracle coverage on the deployment sample)
        covs = np.linspace(0.5, 1.0, 11)
        for c in covs:
            t = np.quantile(conf, 1 - c) if c < 1 else -1
            a = conf >= t
            for name, mk in (("A", ~gte), ("B", gte)):
                aa = a & mk
                r[f"rc_{name}_{c:.2f}"] = (pred[aa] != yte[aa]).mean() if aa.any() else np.nan
                r[f"rcsm_{name}_{c:.2f}"] = (yte[aa] & ~pred[aa]).sum() / yte[mk].sum()
    return r


def run_cell(p, reps, tag, curve=False):
    rows = []
    for ss in np.random.SeedSequence([20261018, zlib.crc32(tag.encode())]).spawn(reps):
        rows.append(one_rep(p, ss, curve))
    df = pd.DataFrame(rows); df["rep"] = range(reps)
    return df


if __name__ == "__main__":
    t0 = time.time()
    # ---- 1. 2^4 factorial ----
    LEVELS = dict(annot=dict(uB=[0.0, 0.35]), shift=dict(m=[0.0, 1.5]), sep=dict(sep=[0.0, 2.0]), gap=dict(dB=[2.5, 1.8]))
    fac = list(LEVELS)
    out = []
    for code in itertools.product([0, 1], repeat=4):
        p = dict(BASE)
        for f, c in zip(fac, code):
            for k, v in LEVELS[f].items():
                p[k] = v[c]
        tag = "F" + "".join(map(str, code))
        df = run_cell(p, REPS_F, tag)
        for f, c in zip(fac, code):
            df[f] = c
        out.append(df)
    pd.concat(out).to_csv(f"{OUT}/factorial.csv", index=False)
    print(f"factorial done {time.time()-t0:.0f}s", flush=True)

    # ---- 2. dose-response curves ----
    rows = []
    for sep in (0.0, 2.0):
        for u in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            p = dict(BASE, sep=sep, uB=u)
            df = run_cell(p, REPS_D, f"DA{sep}{u}"); df["sep"] = sep; df["uB"] = u; df["curve"] = "annotation"; rows.append(df)
    for mode, stype, kw in (("B", "translation+gain", {}), ("all", "translation+gain", {}),
                            ("B", "translation only", dict(gain=0.0)), ("B", "gain only", dict(kt=0.0))):
        for m in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
            p = dict(BASE, m=m, mode=mode, **kw)
            df = run_cell(p, REPS_D, f"DS{mode}{stype}{m}"); df["m"] = m; df["mode"] = mode
            df["stype"] = stype; df["curve"] = "shift"; rows.append(df)
    for fb in (0.05, 0.1, 0.2, 0.3, 0.5):
        for scen, kw in (("annotation", dict(sep=2.0, uB=0.35)), ("shift", dict(m=1.5))):
            p = dict(BASE, frac_B=fb, **kw)
            df = run_cell(p, REPS_D, f"DF{fb}{scen}"); df["frac_B"] = fb; df["scenario"] = scen; df["curve"] = "minority"; rows.append(df)
    pd.concat(rows).to_csv(f"{OUT}/dose.csv", index=False)
    print(f"dose done {time.time()-t0:.0f}s", flush=True)

    # ---- 3. risk-coverage archetypes (LR) and MLP robustness ----
    ARCH = {"baseline": {}, "annotation bias": dict(sep=2.0, uB=0.35), "domain shift": dict(m=1.5),
            "both": dict(sep=2.0, uB=0.35, m=1.5)}
    rows = []
    for nm, kw in ARCH.items():
        df = run_cell(dict(BASE, **kw), REPS_D, f"RC{nm}", curve=True); df["arch"] = nm; df["model"] = "lr"; rows.append(df)
    pd.concat(rows).to_csv(f"{OUT}/archetypes_lr.csv", index=False)
    rows = []
    for nm, kw in ARCH.items():
        df = run_cell(dict(BASE, model="mlp", n_tr=2000, n_val=1000, n_te=3000, **kw), 30, f"MLP{nm}")
        df["arch"] = nm; df["model"] = "mlp"; rows.append(df)
    pd.concat(rows).to_csv(f"{OUT}/archetypes_mlp.csv", index=False)
    json.dump(dict(base=BASE, levels=LEVELS, reps_factorial=REPS_F, reps_dose=REPS_D, d=D,
                   numpy=np.__version__), open(f"{OUT}/config.json", "w"), indent=1)
    print(f"all done {time.time()-t0:.0f}s")
