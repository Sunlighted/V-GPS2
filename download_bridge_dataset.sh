#!/bin/bash

# Script to download Bridge dataset from Berkeley RAIL
# URL: https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds

# Set download directory
DOWNLOAD_DIR="$SCRATCH/bridge_dataset"
mkdir -p "$DOWNLOAD_DIR"

# Download the dataset using wget
# -r: recursive download
# -np: no parent directories
# -nH: no host directories
# -R "index.html*": reject index files
# --cut-dirs=4: remove the first 4 directory levels from the path
# -P: prefix/directory to save files

wget -r -np -nH --cut-dirs=4 \
     -R "index.html*" \
     -P "$DOWNLOAD_DIR" \
     https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/

echo "Download complete! Files saved to: $DOWNLOAD_DIR"
