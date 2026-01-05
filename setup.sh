#!/bin/bash

# SmolLM2 WASM Verification Demo - Full Setup Script
# This script builds the WASM module and starts the server

set -e

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║   🚀 SmolLM2 WASM Verification Demo - Setup                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check Go version
GO_VERSION=$(go version 2>/dev/null | grep -oP 'go\d+\.\d+' | head -1)
if [ -z "$GO_VERSION" ]; then
    echo "❌ Go is not installed. Please install Go 1.21+ first."
    exit 1
fi
echo "✅ Found $GO_VERSION"

# Navigate to project root
cd "$(dirname "$0")"
PROJECT_ROOT=$(pwd)

# Step 1: Build WASM module
echo ""
echo "📦 Step 1: Building WASM module..."
cd wasm
./build_wasm.sh

# Step 2: Copy WASM files to web folder
echo ""
echo "📋 Step 2: Copying WASM files to web folder..."
cp main.wasm ../web/
cp wasm_exec.js ../web/
echo "✅ Copied main.wasm and wasm_exec.js to web/"

# Step 3: Return to project root and run server
cd "$PROJECT_ROOT"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║   🌐 Starting server...                                                  ║"
echo "║   Note: Model (~270MB) will auto-download on first run                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

go run server.go
