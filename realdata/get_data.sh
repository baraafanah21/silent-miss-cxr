#!/usr/bin/env bash
# Download the public COVID-19 image data collection (Cohen et al.) used for the real-image experiment.
set -euo pipefail
cd "$(dirname "$0")"
[ -d covid ] || git clone --depth 1 https://github.com/ieee8023/covid-chestxray-dataset covid
