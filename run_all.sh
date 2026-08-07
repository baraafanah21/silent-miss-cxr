#!/usr/bin/env bash
# Reproduce every number, table and figure in the paper.
set -euo pipefail
cd "$(dirname "$0")"
# 1. Factorial simulation (Sections IV-A to IV-D)
python3 src/sim.py 200
python3 src/analyze.py
python3 src/headline.py
python3 src/tables.py
python3 src/figs.py
# 2. Metric decomposition, robustness and mitigation (Sections IV-E to IV-G)
python3 src/sim2.py 60
python3 src/tables2.py
# 3. Real radiographs (Section IV-H); needs torch + torchxrayvision and ~600 MB of images
(cd realdata && ./get_data.sh && python3 extract.py && python3 extract_pert.py && python3 realexp.py 300)
