"""LaTeX tables for the review extensions (results_sim2/sim2.csv)."""
import numpy as np, pandas as pd
from scipy import stats

R = pd.read_csv("results_sim2/sim2.csv")
SC = ["baseline", "signal gap", "annotation bias", "domain shift", "annot+shift", "all four"]
NAME = {"baseline": "Baseline", "signal gap": "Signal gap", "annotation bias": "Annotation bias",
        "domain shift": "Domain shift", "annot+shift": "Annotation + shift", "all four": "All four"}


def ci(v, short=False):
    v = np.asarray(v, float) * 100
    h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    m = v.mean()
    if short:
        return f"${m:+.1f}$"
    return f"${m:+.1f}$ {{\\scriptsize[{m-h:+.1f}, {m+h:+.1f}]}}"


main = R[R.variant == "main (LR, random flips)"]
# Table A: decomposition
L = []
for s in SC:
    g = main[main.scenario == s]
    L.append(f"{NAME[s]} & {ci(g.d_abst)} & {ci(g.d_fnr)} & {ci(g.d_selfnr)} & {ci(g.d_covpos)} & {ci(g.d_smr)} \\\\")
open("paper/tab_decomp.tex", "w").write("\n".join(L).replace("+", "+").replace("$-", "$-"))

# Table B: mitigation
L = []
for s in SC[1:]:
    g = main[main.scenario == s]
    L.append(f"{NAME[s]} & {ci(g.d_abst, True)} & {ci(g.d_smr, True)} & {ci(g.gthr_d_abst, True)} & {ci(g.gthr_d_smr, True)} & "
             f"{ci(g.recal_d_abst, True)} & {ci(g.recal_d_smr, True)} & {g.recal_abst_B.mean()*100:.0f} / {g.recal_abst_A.mean()*100:.0f} \\\\")
open("paper/tab_mitig.tex", "w").write("\n".join(L))

# Table C: robustness (abstention gap / SMR gap)
VAR = [("main (LR, random flips)", "Main (LR, random flips)"), ("subtle-case flips", "Subtle cases mislabelled"),
       ("nonlinear features + MLP", "Nonlinear features + MLP"), ("coverage 90\\%", None), ("coverage 70\\%", None)]
VAR = [("main (LR, random flips)", "Main setting"), ("subtle-case flips", "Subtle cases mislabelled"),
       ("nonlinear features + MLP", "Nonlinear + MLP"), ("coverage 90%", "Coverage 90\\%"), ("coverage 70%", "Coverage 70\\%")]
L = []
for v, vn in VAR:
    g0 = R[R.variant == v]
    cells = []
    for s in ["annotation bias", "domain shift", "all four"]:
        g = g0[g0.scenario == s]
        cells += [ci(g.d_abst, True), ci(g.d_smr, True)]
    L.append(f"{vn} & " + " & ".join(cells) + " \\\\")
open("paper/tab_robust.tex", "w").write("\n".join(L))
for f in ["tab_decomp", "tab_mitig", "tab_robust"]:
    print(open(f"paper/{f}.tex").read(), "\n")
