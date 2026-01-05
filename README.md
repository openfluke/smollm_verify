# SmolLM2 WASM Verification Demo

This example demonstrates **Loom's polyglot isomorphic capabilities** by running the same SmolLM2-135M-Instruct model in both native Go and WebAssembly, proving token-for-token identical output.

## What This Proves

| Capability | Demonstration |
|------------|---------------|
| **Cross-Platform Parity** | Native Go and WASM produce identical tokens |
| **Browser Inference** | Full LLM inference runs in the browser |
| **Browser Training** | Training with backward pass works in WASM |
| **Deterministic AI** | Same model → Same weights → Same output, everywhere |

## Prerequisites

- **Go 1.21+** — Required for WASM compilation and server
- **Modern browser** — Chrome, Firefox, Safari, or Edge with WebAssembly support
- **~300MB disk space** — For model files (auto-downloaded on first run)
- **~1GB RAM** — For browser-side model inference

## Quick Start

### Option 1: One-Command Setup (Recommended)

```bash
./setup.sh
```

This builds WASM, copies files, and starts the server in one step.

### Option 2: Manual Setup

```bash
# 1. Build WASM module
cd wasm && ./build_wasm.sh && cp main.wasm wasm_exec.js ../web/ && cd ..

# 2. Start server (auto-downloads ~270MB model on first run)
go run server.go

# 3. Open browser → http://localhost:8080/web/
```

> **Note**: The server automatically downloads [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) (~270MB) from HuggingFace on first run. This may take a few minutes depending on your connection.

## Demo Flow

### Phase 1: Verify Identical Output
1. Model loads in both backend (native Go) and browser (WASM)
2. Click **Generate** → Both generate 25 tokens from "Once upon a time"
3. See **"✅ 25/25 tokens match"** - proving identical behavior

### Phase 2: Demonstrate Training Divergence
1. Enter different training text for Backend vs WASM
2. Click **🎓 Train Both** → Each model updates weights independently
3. Click **🔄 Regenerate** → See outputs now **DIVERGE**

This proves:
- Same starting weights → Same output ✅
- Different training → Different output ✅
- Both training AND inference work in WASM ✅

## Architecture

```
smollm_verify/
├── server.go         # HTTP server: static files + /api/generate + /api/train
├── backend.go        # Standalone backend test (optional)
├── wasm/
│   ├── main.go       # WASM module: tokenizer, model, generate, train
│   └── build_wasm.sh # Compiles Go → WASM
├── web/
│   ├── index.html    # Web UI with side-by-side comparison
│   ├── main.wasm     # Compiled WASM module
│   └── wasm_exec.js  # Go WASM runtime
└── models/           # Auto-downloaded SmolLM2-135M-Instruct
```

## Key Technical Points

### Isomorphic Code
The same Loom `nn` package is used in both:
- `server.go` → Compiled to native binary
- `wasm/main.go` → Compiled to WebAssembly

### Training in Browser
```go
// WASM trainStep function
network.ForwardCPU(input)
network.BackwardCPU(gradOutput)
network.UpdateWeights(learningRate)
```

This is real gradient-based training, not approximation.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Check if model is ready |
| `/api/generate?prompt=...&max_tokens=25` | GET | SSE token streaming |
| `/api/train` | POST | SSE training with loss streaming |

## Why This Matters

> "Loom is a **polyglot, isomorphic, deterministic AI framework** with cross-platform training and inference."

Most frameworks claim "runs in browser" but use different codebases. Loom uses **one codebase** compiled to multiple targets, ensuring:

1. **Verifiable AI** - Outputs can be cryptographically proven consistent
2. **Privacy-First** - Data never leaves the browser
3. **Edge Training** - Fine-tune models on-device
4. **Federated Learning** - Identical gradient calculations everywhere

## Model Info

- **Model**: [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- **Parameters**: 135M
- **Hidden Size**: 576
- **Layers**: 30 transformer blocks (120 Loom layers)
- **Vocab**: 49,152 tokens
