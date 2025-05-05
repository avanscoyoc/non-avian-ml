#!/bin/bash

# Install required dependencies
apt-get update && apt-get install -y curl apt-transport-https ca-certificates gnupg

# Add the Google Cloud SDK distribution URI as a package source
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Import the Google Cloud public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Update and install the SDK
apt-get update && apt-get install -y google-cloud-sdk

# Print instructions for the user
echo "Google Cloud SDK installed successfully!"
echo "To authenticate, please run: gcloud auth application-default login"