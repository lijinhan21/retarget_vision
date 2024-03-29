#!/bin/bash

# Define the URLs of the zip files
zip_url_1="https://utexas.box.com/shared/static/m5skxdc2l8ku49zv0o9oqmaafspz6juh.zip"
zip_url_2="https://utexas.box.com/shared/static/sveqkl2qrtrw2fpex1q84hfyn6d1le62.zip"

# Define the names for the downloaded files
zip_file_1="datasets.zip"
zip_file_2="annotations.zip"

# Use curl or wget to download the zip files
echo "Downloading zip file 1..."
curl -o "$zip_file_1" "$zip_url_1" || wget -O "$zip_file_1" "$zip_url_1"

echo "Downloading zip file 2..."
curl -o "$zip_file_2" "$zip_url_2" || wget -O "$zip_file_2" "$zip_url_2"

# Unzip the downloaded files
echo "Unzipping file 1..."
unzip -o "$zip_file_1"

echo "Unzipping file 2..."
unzip -o "$zip_file_2"

echo "Download and extraction complete."
