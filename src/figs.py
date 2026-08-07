import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S, O = "results_sim", "figs"
import os; os.makedirs(O, exist_ok=True)
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots 1-4, fixed order
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False})


def tidy(ax, gridaxis="y"):
    ax.grid(axis=gridaxis, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)


# ---------------- Fig 1: Shapley attribution ----------------
sh = pd.read_csv(f"{S}/shapley.csv")
FACT = [("gap", "Disease-signal gap", C[0]), ("annot", "Annotation bias", C[1]),
        ("shift", "Domain shift", C[2]), ("sep", "Subgroup separability", C[3])]
PAN = [("d_abst", "Abstention-rate gap\n(B minus A)"), ("d_miss_silent", "Silent-miss gap\n(B minus A)"),
       ("d_risk_sel", "Selective-risk gap\n(B minus A)")]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
for ax, (met, ttl) in zip(axs, PAN):
    d = sh[sh.metric == met].set_index("factor")
    for i, (f, lab, col) in enumerate(FACT):
        v, lo, hi = d.loc[f, ["shapley", "lo", "hi"]] * 100
        ax.barh(i, v, color=col, height=0.55, zorder=3)
        ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], color=INK2, lw=1, capsize=2, zorder=4)
        ax.text(v + (1.2 if v >= 0 else -1.2) + (hi - v if v >= 0 else lo - v), i, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8, zorder=2)
    ax.set_title(ttl, loc="left"); tidy(ax, "x")
    ax.set_yticks(range(4)); ax.set_yticklabels([x[1] for x in FACT], color=INK2)
    ax.invert_yaxis(); ax.set_xlabel("percentage points")
    lim = max(abs(d.lo.min()), abs(d.hi.max())) * 100
    ax.set_xlim(-lim * 1.6 if d.shapley.min() < 0 else -lim * 0.15, lim * 1.5)
plt.tight_layout(); plt.savefig(f"{O}/fig1_shapley.png", dpi=220); plt.savefig(f"{O}/fig1_shapley.pdf"); plt.close()

# ---------------- Fig 2: dose-response ----------------
da = pd.read_csv(f"{S}/dose_annotation.csv"); ds = pd.read_csv(f"{S}/dose_shift.csv")
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0))
ax = axs[0, 0]
for sep, col, mk, lab in ((2.0, C[1], "s", "separable (high)"), (0.0, C[0], "o", "not separable")):
    d = da[da.sep == sep]; ax.plot(d.uB, d.d_miss_silent * 100, color=col, marker=mk, ms=4.5, lw=1.8)
    ax.text(0.52, d.d_miss_silent.iloc[-1] * 100, lab, color=INK2, va="center", fontsize=7.5)
ax.set_title("Silent-miss gap vs. underdiagnosis rate", loc="left"); ax.set_ylabel("B minus A (pp)")
ax.set_xlabel("underdiagnosis rate in subgroup B"); ax.set_xlim(0, 0.68); tidy(ax)
ax = axs[0, 1]
for sep, col, mk in ((2.0, C[1], "s"), (0.0, C[0], "o")):
    d = da[da.sep == sep]; ax.plot(d.uB, d.d_abst * 100, color=col, marker=mk, ms=4.5, lw=1.8)
ax.axhline(0, color=INK2, lw=0.8)
ax.text(0.5, 0.62, "not separable", color=INK2, fontsize=7.5, ha="right"); ax.text(0.12, -1.8, "separable", color=INK2, fontsize=7.5, ha="left")
ax.set_ylim(-2.5, 1.0)
ax.set_title("Abstention gap vs. underdiagnosis rate", loc="left"); ax.set_ylabel("B minus A (pp)")
ax.set_xlabel("underdiagnosis rate in subgroup B"); tidy(ax)
ST = [("translation only", C[0], "o"), ("gain only", C[2], "^"), ("translation+gain", C[1], "s")]
ax = axs[1, 0]
for st, col, mk in ST:
    d = ds[(ds["mode"] == "B") & (ds.stype == st)]; ax.plot(d.m, d.risk_sel_B * 100, color=col, marker=mk, ms=4.5, lw=1.8)
    ax.text(3.08, d.risk_sel_B.iloc[-1] * 100, st.replace("+", " + "), color=INK2, va="center", fontsize=7.5)
a0 = ds[(ds["mode"] == "B") & (ds.stype == "gain only") & (ds.m == 0)].risk_sel_A.iloc[0] * 100
ax.axhline(a0, color=MUTED, lw=1, ls="--"); ax.text(3.08, a0 + 0.12, "group A (unshifted)", color=INK2, fontsize=7.5, va="bottom")
ax.set_title("Accepted-case error in B vs. shift", loc="left"); ax.set_ylabel("selective risk, true labels (%)")
ax.set_xlabel("shift magnitude in subgroup B"); ax.set_xlim(0, 5.0); tidy(ax)
ax = axs[1, 1]
YPOS = {"translation only": None, "gain only": 13.4, "translation+gain": 14.5}
for st, col, mk in ST:
    d = ds[(ds["mode"] == "B") & (ds.stype == st)]; ax.plot(d.m, d.abst_B * 100, color=col, marker=mk, ms=4.5, lw=1.8)
    ax.text(3.08, YPOS[st] or d.abst_B.iloc[-1] * 100, st.replace("+", " + "), color=INK2, va="center", fontsize=7.5)
d = ds[(ds["mode"] == "all") & (ds.stype == "translation+gain")]
ax.plot(d.m, d.abst_A * 100, color=INK2, lw=1.4, ls=":", marker="D", ms=3.5)
ax.text(3.08, 15.6, "both groups shifted", color=INK2, va="center", fontsize=7.5)
ax.axhline(20, color=MUTED, lw=1, ls="--"); ax.text(3.08, 19.6, "design target 20%", color=INK2, va="center", fontsize=7.5)
ax.set_title("Abstention rate in B vs. shift", loc="left"); ax.set_ylabel("abstention rate (%)")
ax.set_xlabel("shift magnitude"); ax.set_xlim(0, 5.0); ax.set_ylim(12, 23); tidy(ax)
plt.tight_layout(); plt.savefig(f"{O}/fig2_dose.png", dpi=220); plt.savefig(f"{O}/fig2_dose.pdf"); plt.close()

# ---------------- Fig 3: risk-coverage curves ----------------
A = pd.read_csv(f"{S}/archetypes_lr.csv")
covs = np.linspace(0.5, 1.0, 11)
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)
for ax, arch in zip(axs, ["baseline", "annotation bias", "domain shift", "both"]):
    g = A[A.arch == arch]
    for nm, col, mk, ls in (("A", C[0], "o", "-"), ("B", C[1], "s", "--")):
        y = [g[f"rc_{nm}_{c:.2f}"].mean() * 100 for c in covs]
        ax.plot(covs, y, color=col, marker=mk, ms=3.5, lw=1.6, ls=ls)
        if arch != "baseline":
            ax.text(0.495, y[0], nm, color=col, ha="right", va="center", fontsize=8, fontweight="bold")
    if arch == "baseline":
        ax.text(0.5, 3.4, "A and B\noverlap", color=INK2, fontsize=7.5, ha="left")
    ax.set_title(arch, loc="left", fontsize=8.5); tidy(ax); ax.set_xlim(0.42, 1.02)
    ax.set_xticks([0.5, 0.75, 1.0]); ax.set_xticklabels(["50", "75", "100"])
axs[0].set_ylabel("selective risk (%)"); fig.supxlabel("coverage (%)", fontsize=8.5, color=INK2, y=0.02)
plt.tight_layout(); plt.savefig(f"{O}/fig3_riskcov.png", dpi=220); plt.savefig(f"{O}/fig3_riskcov.pdf"); plt.close()
print("ok")
