#!/bin/bash
# Run RAGDocs with FastMCP

# Change to the directory of this script
cd "$(dirname "$0")"

# Run the server with FastMCP
python -m ragdocs.server "$@"
