"""Headline numbers with 95% CIs (t-based over replicates) for the paper."""
import numpy as np, pandas as pd
from scipy import stats

S = "results_sim"
F = pd.read_csv(f"{S}/factorial.csv")
AL = pd.concat([pd.read_csv(f"{S}/archetypes_lr.csv"), pd.read_csv(f"{S}/archetypes_mlp.csv")])
cols = ["abst_A", "abst_B", "d_abst", "risk_sel_A", "risk_sel_B", "d_risk_sel", "d_risk_sel_obs", "risk_sel_obs_B",
        "miss_full_B", "miss_silent_B", "d_miss_silent", "risk_gap_full", "magnification"]


def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - h, v.mean() + h, len(v)


rows = []
for (model, arch), g in AL.groupby(["model", "arch"]):
    for c in cols:
        m, lo, hi, n = ci(g[c]); rows.append(dict(set="archetype", model=model, scenario=arch, metric=c, mean=m, lo=lo, hi=hi, n=n))
names = {(0, 0, 0, 0): "all off", (1, 1, 1, 1): "all on", (0, 0, 0, 1): "gap only", (1, 1, 1, 0): "annot+shift+sep (no gap)"}
for code, nm in names.items():
    g = F[(F.annot == code[0]) & (F["shift"] == code[1]) & (F.sep == code[2]) & (F.gap == code[3])]
    for c in cols:
        m, lo, hi, n = ci(g[c]); rows.append(dict(set="factorial", model="lr", scenario=nm, metric=c, mean=m, lo=lo, hi=hi, n=n))
H = pd.DataFrame(rows); H.to_csv(f"{S}/headline.csv", index=False)

# catch rate: share of group-B missed positives that abstention intercepted (ratio of means)
cr = []
for (model, arch), g in AL.groupby(["model", "arch"]):
    cr.append(dict(model=model, scenario=arch, catch_rate=1 - g.miss_silent_B.mean() / g.miss_full_B.mean(),
                   miss_full_B=g.miss_full_B.mean(), miss_silent_B=g.miss_silent_B.mean()))
for code, nm in names.items():
    g = F[(F.annot == code[0]) & (F["shift"] == code[1]) & (F.sep == code[2]) & (F.gap == code[3])]
    cr.append(dict(model="lr-factorial", scenario=nm, catch_rate=1 - g.miss_silent_B.mean() / g.miss_full_B.mean(),
                   miss_full_B=g.miss_full_B.mean(), miss_silent_B=g.miss_silent_B.mean()))
pd.DataFrame(cr).to_csv(f"{S}/catch_rate.csv", index=False)

pd.set_option("display.width", 200)
P = H.pivot_table(index=["set", "model", "scenario"], columns="metric", values="mean")
print((P[["abst_A", "abst_B", "d_abst", "risk_sel_A", "risk_sel_B", "d_miss_silent", "miss_silent_B", "d_risk_sel_obs", "d_risk_sel"]] * 100).round(1))
print(pd.DataFrame(cr).round(3))
print(H[H.metric.isin(["d_abst", "d_miss_silent", "d_risk_sel"]) & (H.set == "factorial")].round(4).to_string(index=False))
print(H[H.metric.isin(["d_abst", "d_miss_silent", "d_risk_sel", "d_risk_sel_obs", "risk_sel_B", "risk_sel_obs_B"]) & (H.set == "archetype")].round(4).to_string(index=False))
