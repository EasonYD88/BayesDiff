#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Download PDBbind v2020 Refined Set
#
# PDBbind requires registration at http://www.pdbbind.org.cn/
# After registration, download the refined set and run this script.
#
# Method 1 — Manual download (recommended):
#   1. Register at http://www.pdbbind.org.cn/
#   2. Download "PDBbind_v2020_refined.tar.gz"
#   3. Place it in data/ and run this script:
#      bash scripts/utils/download_pdbbind.sh data/PDBbind_v2020_refined.tar.gz
#
# Method 2 — Direct URL (if you have one):
#   PDBBIND_URL="https://..." bash scripts/utils/download_pdbbind.sh
#
# Method 3 — Copy from another cluster user:
#   cp -r /path/to/existing/pdbbind data/pdbbind
#
# Expected output structure:
#   data/pdbbind/
#   ├── refined-set/
#   │   ├── 1a1e/
#   │   │   ├── 1a1e_protein.pdb
#   │   │   ├── 1a1e_ligand.sdf
#   │   │   └── 1a1e_pocket.pdb (optional)
#   │   ├── 1a28/
#   │   │   └── ...
#   │   └── index/
#   │       └── INDEX_refined_data.2020
#   └── INDEX_refined_data.2020
# ─────────────────────────────────────────────────────────────

set -euo pipefail

TARGET_DIR="${TARGET_DIR:-data/pdbbind}"
TARBALL="${1:-}"
PDBBIND_URL="${PDBBIND_URL:-}"

mkdir -p "${TARGET_DIR}"

# ── Method 1: From local tarball ─────────────────────────────
if [[ -n "${TARBALL}" && -f "${TARBALL}" ]]; then
    echo "Extracting from tarball: ${TARBALL}"
    tar -xzf "${TARBALL}" -C "${TARGET_DIR}" --strip-components=1
    echo "Done. Data in: ${TARGET_DIR}"

# ── Method 2: From URL ───────────────────────────────────────
elif [[ -n "${PDBBIND_URL}" ]]; then
    echo "Downloading from: ${PDBBIND_URL}"
    TMPFILE=$(mktemp /tmp/pdbbind_XXXXXXXX.tar.gz)
    wget -q --show-progress -O "${TMPFILE}" "${PDBBIND_URL}"
    echo "Extracting..."
    tar -xzf "${TMPFILE}" -C "${TARGET_DIR}" --strip-components=1
    rm -f "${TMPFILE}"
    echo "Done. Data in: ${TARGET_DIR}"

# ── No input ─────────────────────────────────────────────────
else
    echo "ERROR: No data source specified."
    echo ""
    echo "Usage:"
    echo "  # From local tarball:"
    echo "  bash $0 path/to/PDBbind_v2020_refined.tar.gz"
    echo ""
    echo "  # From URL:"
    echo "  PDBBIND_URL='https://...' bash $0"
    echo ""
    echo "  # Or copy from another user:"
    echo "  cp -r /path/to/pdbbind ${TARGET_DIR}"
    echo ""
    echo "Register at http://www.pdbbind.org.cn/ to get the download."
    exit 1
fi

# ── Verify structure ─────────────────────────────────────────
echo ""
echo "Verifying data structure..."

# Find INDEX file
INDEX=""
for candidate in \
    "${TARGET_DIR}/INDEX_refined_data.2020" \
    "${TARGET_DIR}/index/INDEX_refined_data.2020" \
    "${TARGET_DIR}/refined-set/index/INDEX_refined_data.2020"; do
    if [[ -f "${candidate}" ]]; then
        INDEX="${candidate}"
        break
    fi
done

if [[ -z "${INDEX}" ]]; then
    echo "WARNING: INDEX_refined_data.2020 not found!"
    echo "Searched:"
    echo "  ${TARGET_DIR}/INDEX_refined_data.2020"
    echo "  ${TARGET_DIR}/index/INDEX_refined_data.2020"
    echo "  ${TARGET_DIR}/refined-set/index/INDEX_refined_data.2020"
else
    N_ENTRIES=$(grep -c "^[0-9]" "${INDEX}" 2>/dev/null || echo "?")
    echo "INDEX file: ${INDEX} (${N_ENTRIES} entries)"
fi

# Check refined-set
if [[ -d "${TARGET_DIR}/refined-set" ]]; then
    N_DIRS=$(find "${TARGET_DIR}/refined-set" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "refined-set: ${N_DIRS} complex directories"
    # Sample check
    SAMPLE_DIR=$(find "${TARGET_DIR}/refined-set" -mindepth 1 -maxdepth 1 -type d | head -1)
    if [[ -n "${SAMPLE_DIR}" ]]; then
        CODE=$(basename "${SAMPLE_DIR}")
        echo "Sample complex: ${CODE}"
        ls -la "${SAMPLE_DIR}/" | head -5
    fi
else
    echo "WARNING: refined-set/ directory not found!"
fi

echo ""
echo "=== Setup complete ==="
echo "Next: run the preparation pipeline:"
echo "  bash slurm/pipeline/s00_launch_all.sh"
