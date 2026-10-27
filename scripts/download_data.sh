#!/usr/bin/env bash
# Download PKLot (UFPR, CC BY 4.0) and CNRPark-EXT (CNR, ODbL 1.0) from their
# official sources into data/downloads/, then extract them into data/raw/.
# Re-running resumes partial downloads and skips finished extractions.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
# python.org macOS builds ship without CA certificates
export SSL_CERT_FILE="${SSL_CERT_FILE:-$("$PY" -c 'import certifi; print(certifi.where())')}"

CNR=https://github.com/fabiocarrara/deep-parking/releases/download/archive
PKLOT=https://www.inf.ufpr.br/vri/databases/PKLot.tar.gz

mkdir -p data/downloads/cnr data/downloads/pklot data/raw

fetch() { curl -fL --retry 5 -C - -o "$1" "$2"; }

fetch data/downloads/cnr/CNRPark+EXT.csv               "$CNR/CNRPark+EXT.csv"
fetch data/downloads/cnr/CNR-EXT-Patches-150x150.zip   "$CNR/CNR-EXT-Patches-150x150.zip"
fetch data/downloads/cnr/CNR-EXT_FULL_IMAGE_1000x750.tar "$CNR/CNR-EXT_FULL_IMAGE_1000x750.tar"
fetch data/downloads/pklot/PKLot.tar.gz                "$PKLOT"

if [ ! -d data/raw/cnr/PATCHES ]; then
  mkdir -p data/raw/cnr
  unzip -q data/downloads/cnr/CNR-EXT-Patches-150x150.zip -d data/raw/cnr
fi
if [ ! -d data/raw/cnr/FULL_IMAGE_1000x750 ]; then
  tar -xf data/downloads/cnr/CNR-EXT_FULL_IMAGE_1000x750.tar -C data/raw/cnr
fi
cp -n data/downloads/cnr/CNRPark+EXT.csv data/raw/cnr/ || true
if [ ! -d data/raw/pklot/PKLot ]; then
  mkdir -p data/raw/pklot
  # full frames + XML only; the archive's PKLotSegmented/ (695k pre-cut
  # patches) is not used because patches are cut by scripts/prepare_data.py
  tar -xzf data/downloads/pklot/PKLot.tar.gz -C data/raw/pklot PKLot/PKLot
fi
echo "ALL_DONE"
