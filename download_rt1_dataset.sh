#!/bin/bash

# Script to download RT-1 dataset from Open X-Embodiment
# Using wget to download from Google Cloud Storage HTTP endpoint

# Set download directory
DOWNLOAD_DIR="/home/yruan/V-GPS/data"
mkdir -p "$DOWNLOAD_DIR"

# RT-1 dataset name
DATASET_NAME="fractal20220817_data"

echo "Downloading RT-1 dataset: $DATASET_NAME"
echo "Destination: $DOWNLOAD_DIR"
echo "This may take a while depending on your connection speed..."
echo ""

# Check if gsutil is available
if command -v gsutil &> /dev/null; then
    echo "Using gsutil for download..."
    gsutil -m cp -r gs://gresearch/robotics/$DATASET_NAME "$DOWNLOAD_DIR/"
else
    echo "gsutil not found. Using wget with HTTP endpoint..."
    # Download using wget from GCS HTTP endpoint
    # -r: recursive
    # -np: no parent directories
    # -nH: no host directories
    # -R "index.html*": reject index files
    # --cut-dirs=2: remove directory levels
    
    BASE_URL="https://storage.googleapis.com/gresearch/robotics/$DATASET_NAME"
    
    # Create dataset directory
    mkdir -p "$DOWNLOAD_DIR/$DATASET_NAME"
    cd "$DOWNLOAD_DIR/$DATASET_NAME"
    
    wget -r -np -nH --cut-dirs=3 \
         -R "index.html*" \
         -e robots=off \
         --no-check-certificate \
         "$BASE_URL/"
fi

echo ""
echo "Download complete! Dataset saved to: $DOWNLOAD_DIR/$DATASET_NAME"
echo ""
echo "You can now load the dataset using:"
echo "import tensorflow_datasets as tfds"
echo "dataset = tfds.load('$DATASET_NAME')"
