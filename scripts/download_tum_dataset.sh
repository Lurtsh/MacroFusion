#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [sequence_name] [dataset_group]"
    echo "  sequence_name: e.g. rgbd_dataset_freiburg1_xyz (default)"
    echo "  dataset_group: e.g. freiburg1 (default)"
    echo ""
    echo "Downloads a TUM RGB-D sequence into data/tum/<sequence_name>/"
    exit 1
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
fi

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
DEST="${ROOT}/data/tum"
SEQ="${1:-rgbd_dataset_freiburg1_xyz}"
GROUP="${2:-freiburg1}"
URL="https://vision.in.tum.de/rgbd/dataset/${GROUP}/${SEQ}.tgz"
ARCHIVE="${DEST}/${SEQ}.tgz"

mkdir -p "${DEST}"

if [ -d "${DEST}/${SEQ}" ]; then
    echo "Dataset already exists: ${DEST}/${SEQ}"
    exit 0
fi

echo "Downloading ${URL}"
curl -fL "${URL}" -o "${ARCHIVE}"

echo "Extracting to ${DEST}"
tar -xzf "${ARCHIVE}" -C "${DEST}"
rm "${ARCHIVE}"

echo "Done: ${DEST}/${SEQ}"
