#!/bin/bash

# Create credentials directory
mkdir -p /workspaces/non-avian-ml-toy/credentials

# Write credentials from environment variable
if [ ! -z "$GOOGLE_APPLICATION_CREDENTIALS_JSON" ]; then
    echo "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > /workspaces/non-avian-ml-toy/credentials/google-credentials.json
    export GOOGLE_APPLICATION_CREDENTIALS="/workspaces/non-avian-ml-toy/credentials/google-credentials.json"
    echo "Google Cloud credentials configured successfully!"
else
    echo "Error: GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable not found"
    echo "Please add your service account key JSON as a Codespace secret"
fi