"""Statistics on simulation outputs: factorial effects (bootstrap CI), Shapley attribution, summaries."""
import itertools, json
import numpy as np, pandas as pd

S = "results_sim"
rng = np.random.default_rng(7)
fac = ["annot", "shift", "sep", "gap"]
F = pd.read_csv(f"{S}/factorial.csv")
METRICS = ["d_abst", "d_risk_sel", "d_miss_silent", "d_miss_full", "risk_gap_full", "magnification", "d_risk_sel_obs"]

# ---------- cell means ----------
cell = F.groupby(fac)[METRICS + ["abst_A", "abst_B", "risk_sel_A", "risk_sel_B", "miss_silent_B", "miss_full_B"]].mean()
cell.to_csv(f"{S}/cell_means.csv")

# ---------- main + two-way effects via OLS on coded factors ----------
def design(df):
    Z = 2 * df[fac].values - 1
    cols, names = [np.ones(len(df))], ["intercept"]
    for i, f in enumerate(fac):
        cols.append(Z[:, i]); names.append(f)
    for (i, a), (j, b) in itertools.combinations(enumerate(fac), 2):
        cols.append(Z[:, i] * Z[:, j]); names.append(f"{a}x{b}")
    return np.column_stack(cols), names

X, names = design(F)
eff_rows = []
idx_by_cell = {k: v.index.values for k, v in F.groupby(fac)}
B = 400
for met in METRICS:
    y = F[met].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    boots = []
    for _ in range(B):
        samp = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by_cell.values()])
        boots.append(np.linalg.lstsq(X[samp], y[samp], rcond=None)[0])
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    for k, nm in enumerate(names):
        if nm == "intercept":
            continue
        # effect = change in the metric when moving the factor (or pair) from low to high
        eff_rows.append(dict(metric=met, term=nm, effect=2 * beta[k], lo=2 * lo[k], hi=2 * hi[k]))
pd.DataFrame(eff_rows).to_csv(f"{S}/effects.csv", index=False)

# ---------- Shapley attribution of (all-on) minus (all-off) ----------
def shapley(means):
    n = len(fac); phi = np.zeros(n)
    perms = list(itertools.permutations(range(n)))
    for perm in perms:
        on = [0] * n
        prev = means[tuple(on)]
        for f in perm:
            on[f] = 1
            cur = means[tuple(on)]
            phi[f] += cur - prev; prev = cur
    return phi / len(perms)

sh_rows = []
for met in METRICS:
    m = {k: v for k, v in F.groupby(fac)[met].mean().items()}
    phi = shapley(m)
    bs = []
    for _ in range(B):
        mb = {}
        for k, ix in idx_by_cell.items():
            mb[k] = F.loc[rng.choice(ix, len(ix), replace=True), met].mean()
        bs.append(shapley(mb))
    bs = np.array(bs); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    total = m[(1, 1, 1, 1)] - m[(0, 0, 0, 0)]
    for i, f in enumerate(fac):
        sh_rows.append(dict(metric=met, factor=f, shapley=phi[i], lo=lo[i], hi=hi[i], total=total))
pd.DataFrame(sh_rows).to_csv(f"{S}/shapley.csv", index=False)

# ---------- dose-response summaries ----------
D = pd.read_csv(f"{S}/dose.csv")
keys = {"annotation": ["sep", "uB"], "shift": ["mode", "stype", "m"], "minority": ["scenario", "frac_B"]}
mets = ["abst_A", "abst_B", "d_abst", "risk_sel_A", "risk_sel_B", "d_risk_sel", "d_miss_silent", "miss_silent_B",
        "miss_full_B", "d_risk_sel_obs", "risk_gap_full", "magnification"]
for c, k in keys.items():
    g = D[D.curve == c].groupby(k)[mets]
    out = g.mean().join(g.std().add_suffix("_sd")).reset_index()
    out.to_csv(f"{S}/dose_{c}.csv", index=False)

# ---------- archetypes ----------
A = pd.concat([pd.read_csv(f"{S}/archetypes_lr.csv"), pd.read_csv(f"{S}/archetypes_mlp.csv")])
am = A.groupby(["model", "arch"])[["abst_A", "abst_B", "d_abst", "risk_sel_A", "risk_sel_B", "d_risk_sel",
                                   "miss_full_B", "miss_silent_B", "d_miss_silent", "risk_sel_obs_B", "risk_sel_B",
                                   "d_risk_sel_obs", "risk_gap_full", "magnification"]].mean()
am.to_csv(f"{S}/archetype_means.csv")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
print("== cell means (key) ==")
print(cell[["abst_A", "abst_B", "d_abst", "d_risk_sel", "miss_silent_B", "d_miss_silent", "risk_gap_full", "magnification"]].round(3))
print("== effects: abstention disparity / silent-miss gap / selective-risk gap ==")
E = pd.DataFrame(eff_rows)
for met in ["d_abst", "d_miss_silent", "d_risk_sel", "magnification"]:
    print(met); print(E[E.metric == met].round(4).to_string(index=False))
print("== shapley ==")
print(pd.DataFrame(sh_rows).round(4).to_string(index=False))
print("== archetypes ==")
print(am.round(3))
for c in keys:
    print("== dose", c); print(pd.read_csv(f"{S}/dose_{c}.csv")[keys[c] + ["abst_A", "abst_B", "d_abst", "risk_sel_A", "risk_sel_B", "d_miss_silent", "d_risk_sel_obs", "d_risk_sel"]].round(3).to_string(index=False))
