#!/bin/bash

# Install Google Cloud SDK (gsutil) in user directory without root access
# This is safe for cluster environments

echo "Installing Google Cloud SDK in your home directory..."

# Download and install
cd ~
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz

# Extract
tar -xf google-cloud-cli-linux-x86_64.tar.gz

# Install (no root needed)
./google-cloud-sdk/install.sh --quiet --usage-reporting=false --path-update=true

# Clean up
rm google-cloud-cli-linux-x86_64.tar.gz

echo ""
echo "Installation complete!"
echo ""
echo "Now run this command to update your current shell:"
echo "  source ~/google-cloud-sdk/path.bash.inc"
echo ""
echo "Or restart your shell session."
echo ""
echo "Then you can use gsutil to download datasets."
