#!/bin/bash

# Build WASM module for SmolLM2 verification

echo "🔨 Building SmolLM2 WASM module..."

cd "$(dirname "$0")"

export GOOS=js
export GOARCH=wasm

go build -o main.wasm main.go

if [ $? -eq 0 ]; then
    echo "✅ Build successful: main.wasm created ($(du -h main.wasm | cut -f1))"
    
    # Copy wasm_exec.js if needed
    if [ ! -f "wasm_exec.js" ]; then
        echo "📋 Copying wasm_exec.js..."
        cp "$(go env GOROOT)/misc/wasm/wasm_exec.js" .
    fi
    
    echo ""
    echo "🚀 Ready! Copy main.wasm and wasm_exec.js to web/ folder"
else
    echo "❌ Build failed"
    exit 1
fi
