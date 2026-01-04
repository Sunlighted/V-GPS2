#!/bin/bash

# Check what's downloaded and get missing files
DATASET_DIR="$SCRATCH/tensorflow_datasets/fractal20220817_data/0.1.0"

echo "Checking downloaded files..."
cd "$DATASET_DIR"

# Count what we have
TOTAL_FILES=$(ls -1 | wc -l)
echo "Currently have $TOTAL_FILES files"

# List files in the bucket without downloading
echo ""
echo "Listing all files in the bucket..."
gsutil ls gs://gresearch/robotics/fractal20220817_data/0.1.0/ > /tmp/all_files.txt

# Count total files
TOTAL_EXPECTED=$(cat /tmp/all_files.txt | wc -l)
echo "Expected $TOTAL_EXPECTED files total"

# Check for missing files
echo ""
echo "Checking for missing files..."
while IFS= read -r file; do
    filename=$(basename "$file")
    if [ ! -f "$filename" ]; then
        echo "Missing: $filename"
        echo "$file" >> /tmp/missing_files.txt
    fi
done < /tmp/all_files.txt

if [ -f /tmp/missing_files.txt ]; then
    echo ""
    echo "Downloading missing files one by one..."
    while IFS= read -r file; do
        echo "Downloading: $file"
        gsutil cp "$file" .
    done < /tmp/missing_files.txt
    rm /tmp/missing_files.txt
else
    echo "All files downloaded!"
fi

rm /tmp/all_files.txt

echo ""
echo "Download verification complete!"
echo "Total files: $(ls -1 | wc -l)"
