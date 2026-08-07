import numpy as np, pandas as pd
from scipy import stats
S = "results_sim"
F = pd.read_csv(f"{S}/factorial.csv")
AL = pd.concat([pd.read_csv(f"{S}/archetypes_lr.csv").assign(model="LR"), pd.read_csv(f"{S}/archetypes_mlp.csv").assign(model="MLP")])
D = pd.read_csv(f"{S}/dose.csv")


def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean() * 100, (v.mean() - h) * 100, (v.mean() + h) * 100


def fmt(v):
    m, lo, hi = ci(v); return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]"


rows = {"Baseline (all off)": (0, 0, 0, 0), "Signal gap only": (0, 0, 0, 1), "Annotation bias + separability": (1, 0, 1, 0),
        "Domain shift only": (0, 1, 0, 0), "Annotation + separability + shift": (1, 1, 1, 0), "All four drivers": (1, 1, 1, 1)}
out = ["| Scenario | Abstain A (%) | Abstain B (%) | Abstention gap, pp [95% CI] | Selective risk A (%) | Selective risk B (%) | Silent misses in B (% of its cases) | Silent-miss gap, pp [95% CI] | Missed cases caught by abstention in B (%) |",
       "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for nm, c in rows.items():
    g = F[(F.annot == c[0]) & (F["shift"] == c[1]) & (F.sep == c[2]) & (F.gap == c[3])]
    catch = (1 - g.miss_silent_B.mean() / g.miss_full_B.mean()) * 100
    out.append(f"| {nm} | {g.abst_A.mean()*100:.1f} | {g.abst_B.mean()*100:.1f} | {fmt(g.d_abst)} | {g.risk_sel_A.mean()*100:.1f} | "
               f"{g.risk_sel_B.mean()*100:.1f} | {g.miss_silent_B.mean()*100:.1f} | {fmt(g.d_miss_silent)} | {catch:.0f} |")
open(f"{S}/table_scenarios.md", "w").write("\n".join(out))

out = ["| Scenario | Model (n) | Abstention gap | Silent-miss gap | Selective-risk gap, true labels | Selective-risk gap, biased-label audit |",
       "| --- | --- | --- | --- | --- | --- |"]
for arch in ["baseline", "annotation bias", "domain shift", "both"]:
    for mdl in ["LR", "MLP"]:
        g = AL[(AL.arch == arch) & (AL.model == mdl)]
        out.append(f"| {arch} | {mdl} ({len(g)}) | {fmt(g.d_abst)} | {fmt(g.d_miss_silent)} | {fmt(g.d_risk_sel)} | {fmt(g.d_risk_sel_obs)} |")
open(f"{S}/table_archetypes.md", "w").write("\n".join(out))

# minority share
out = ["| Share of subgroup B | Silent-miss gap, annotation bias (pp) | Silent-miss gap, domain shift (pp) |", "| --- | --- | --- |"]
M = D[D.curve == "minority"]
for fb in sorted(M.frac_B.unique()):
    a = M[(M.frac_B == fb) & (M.scenario == "annotation")].d_miss_silent.mean() * 100
    s = M[(M.frac_B == fb) & (M.scenario == "shift")].d_miss_silent.mean() * 100
    out.append(f"| {int(fb*100)}% | {a:+.1f} | {s:+.1f} |")
open(f"{S}/table_minority.md", "w").write("\n".join(out))

# dose: key values
da = D[D.curve == "annotation"].groupby(["sep", "uB"])[["d_miss_silent", "d_abst"]].mean() * 100
print(da.round(2))
g = F[(F.annot == 0) & (F["shift"] == 0) & (F.sep == 0) & (F.gap == 1)]
print("gap-only: full-cov err A %.1f B %.1f (gap %.1f, ratio %.2f) | selective risk A %.1f B %.1f (gap %.1f, ratio %.2f)" % (
    (1 - g.acc_full_A.mean()) * 100, (1 - g.acc_full_B.mean()) * 100, g.risk_gap_full.mean() * 100,
    (1 - g.acc_full_B.mean()) / (1 - g.acc_full_A.mean()), g.risk_sel_A.mean() * 100, g.risk_sel_B.mean() * 100,
    g.d_risk_sel.mean() * 100, g.risk_sel_B.mean() / g.risk_sel_A.mean()))
print("sens full A/B gap-only", g.sens_full_A.mean(), g.sens_full_B.mean())
b = F[(F.annot == 0) & (F["shift"] == 0) & (F.sep == 0) & (F.gap == 0)]
print("baseline sens_full A/B", b.sens_full_A.mean(), b.sens_full_B.mean(), "acc", b.acc_full_A.mean())
for f in ["table_scenarios", "table_archetypes", "table_minority"]:
    print("\n" + open(f"{S}/{f}.md").read())
ds = pd.read_csv(f"{S}/dose_shift.csv")
print(ds[(ds["mode"] == "all") & (ds.m.isin([0, 3.0]))][["m", "abst_A", "abst_B", "risk_sel_A", "risk_sel_B"]].round(4))
print(ds[(ds["mode"] == "B") & (ds.m.isin([0, 1.5, 3.0]))][["stype", "m", "abst_A", "abst_B", "risk_sel_A", "risk_sel_B"]].round(4))
