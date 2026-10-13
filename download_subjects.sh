#!/bin/bash

# Base URL for Zenodo record
BASE_URL="https://zenodo.org/records/19021436/files"
TARGET_DIR="/Users/ramanisrinivasan/Documents/Prosthethics_final/datasets"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

for i in {2..25}
do
    echo "=================================================="
    echo "       DOWNLOADING SUBJECT $i...                  "
    echo "=================================================="
    
    ZIP_NAME="sub${i}.zip"
    SUB_DIR="sub${i}"
    
    # Check if folder already exists
    if [ -d "$SUB_DIR" ]; then
        echo "Directory $SUB_DIR already exists. Skipping download."
        continue
    fi
    
    # Download with curl (robust, auto-resume, retry)
    # -C - : Resume if broken
    # -L   : Follow redirects
    curl -L -C - --retry 20 --retry-delay 5 -O "$BASE_URL/$ZIP_NAME"
    
    # Unzip directly into a folder named sub{i}
    echo "Unzipping $ZIP_NAME..."
    unzip -q -o "$ZIP_NAME"
    
    # Clean up the zip file to save space
    rm "$ZIP_NAME"
    
    echo "Subject $i successfully downloaded and extracted."
done

echo "=================================================="
echo "       ALL SUBJECTS SUCCESSFULLY DOWNLOADED!      "
echo "=================================================="
