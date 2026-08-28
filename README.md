# Drivers of Subgroup Abstention in Selective Chest X-Ray AI

Code for the paper **"Drivers of Subgroup Abstention in Selective Chest X-Ray AI: The Roles of Annotation Bias and Domain Shift"**
by Bara' Afanah, Ahmad Shanti, Adel Ibdah and Basil Ibdah (An-Najah National University).

The paper uses a controlled simulation to ask why a selective classifier, which can abstain on uncertain cases, abstains more or less often on one patient subgroup than another. Four drivers are switched on independently in a 2^4 factorial design:

| Driver | What it represents |
| --- | --- |
| Disease-signal gap | Disease is intrinsically harder to see in subgroup B |
| Annotation bias | A share of B's true positives is recorded as negative (underdiagnosis) |
| Subgroup separability | Group membership is detectable from the features |
| Domain shift | B's deployment features are rescaled and/or translated |

Main result: the drivers move abstention in opposite directions, so equal abstention rates can hide a large gap in **silent misses** (diseased cases the model misses and accepts without deferral).

## Repository layout

```
src/
  sim.py        factorial simulation: data generation, selective classifier, experiments
  analyze.py    factorial effects (OLS + bootstrap), Shapley attribution, summaries
  headline.py   scenario means with t-based 95% confidence intervals
  tables.py     tables for the factorial results
  figs.py       Figures 1-3 (PNG and PDF)
  sim2.py       metric decomposition (FNR, selective FNR, coverage of diseased),
                robustness checks, and two mitigations (group thresholds, audit recalibration)
  tables2.py    tables for sim2
realdata/
  get_data.sh   downloads the public COVID-19 image data collection (GitHub)
  extract.py    1024-d embeddings from a pretrained TorchXRayVision DenseNet-121
  extract_pert.py  scanner-like changes to female deployment images, re-encoded
  realexp.py    300 patient-level splits: label bias by sex vs random group, view and scanner shifts
results/        factorial simulation outputs used in the paper
results_ext/    sim2 outputs used in the paper
results_real/   real-image outputs used in the paper
paper/          LaTeX source and PDF (IEEEtran conference format)
run_all.sh      reproduces everything from scratch
```

## Reproduce

Requires Python 3.13 (tested with 3.13.16).

```bash
pip install -r requirements.txt
./run_all.sh
```

The simulations take about 20 minutes on 2 CPU cores. The real-image part downloads about 600 MB of images and runs the network on CPU (roughly 10 to 25 minutes). Random seeds are fixed, so reruns with the listed versions reproduce the numbers in the results folders.

## Build the paper

```bash
cd paper
latexmk -pdf main.tex
```

Needs a TeX distribution with the `IEEEtran` class (TeX Live package `texlive-publishers`).

## Metrics

All metrics are per subgroup and use the true labels unless stated otherwise.

- **Abstention rate**: share of cases deferred.
- **Selective risk**: error rate among accepted cases.
- **Silent-miss rate**: diseased cases predicted negative and not deferred, divided by all diseased cases in the subgroup.
- **Catch rate**: share of a subgroup's missed diseased cases that abstention intercepted.
- **Audit selective risk**: selective risk measured with the recorded, possibly biased labels.

## Limitations

The simulation describes mechanisms under stated assumptions. The real-image check uses a small public collection with injected (not observed) label bias and frozen embeddings, and each split has about 48 female test patients, so only averages over splits should be interpreted. See the Limitations section of the paper.

## Data

Real images: COVID-19 image data collection, J. P. Cohen et al., https://github.com/ieee8023/covid-chestxray-dataset (images carry their own licences; they are not redistributed here). Pretrained model: TorchXRayVision.

## Citation

```bibtex
@misc{afanah2026drivers,
  author = {Afanah, Bara' and Shanti, Ahmad and Ibdah, Adel and Ibdah, Basil},
  title  = {Drivers of Subgroup Abstention in Selective Chest X-Ray AI: The Roles of Annotation Bias and Domain Shift},
  year   = {2026},
  note   = {An-Najah National University}
}
```
