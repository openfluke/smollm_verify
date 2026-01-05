//go:build js && wasm
// +build js,wasm

package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"syscall/js"

	"github.com/openfluke/loom/nn"
	"github.com/openfluke/loom/tokenizer"
)

// Global state for loaded model
var (
	network    *nn.Network
	tk         *tokenizer.Tokenizer
	embeddings []float32
	finalNorm  []float32
	hiddenSize int
	vocabSize  int
	modelReady bool
)

func main() {
	fmt.Println("🧠 SmolLM2 WASM Module Initializing...")

	// Register JavaScript functions
	js.Global().Set("initTokenizer", js.FuncOf(initTokenizer))
	js.Global().Set("initModel", js.FuncOf(initModel))
	js.Global().Set("generate", js.FuncOf(generate))
	js.Global().Set("encode", js.FuncOf(encode))
	js.Global().Set("decode", js.FuncOf(decode))
	js.Global().Set("isModelReady", js.FuncOf(isModelReady))
	
	// Streaming generation
	js.Global().Set("initGeneration", js.FuncOf(initGeneration))
	js.Global().Set("generateStep", js.FuncOf(generateStep))
	js.Global().Set("decodeToken", js.FuncOf(decodeToken))
	
	// Training
	js.Global().Set("trainStep", js.FuncOf(trainStep))

	fmt.Println("✅ WASM functions registered:")
	fmt.Println("   - initTokenizer(tokenizerJSON)")
	fmt.Println("   - initModel(configJSON, weightsArrayBuffer)")
	fmt.Println("   - generate(prompt, maxTokens)")
	fmt.Println("   - initGeneration(prompt, maxTokens) + generateStep()")
	fmt.Println("   - trainStep(text, learningRate)")
	fmt.Println("   - encode(text)")
	fmt.Println("   - decode(tokenIds)")
	fmt.Println("   - isModelReady()")

	// Keep WASM running
	select {}
}

// initTokenizer loads tokenizer from JSON string
func initTokenizer(this js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return `{"error": "Expected tokenizer JSON string"}`
	}

	tokenizerJSON := args[0].String()
	var err error
	tk, err = tokenizer.LoadFromBytes([]byte(tokenizerJSON))
	if err != nil {
		return fmt.Sprintf(`{"error": "Failed to load tokenizer: %v"}`, err)
	}

	result := map[string]interface{}{
		"success":   true,
		"vocabSize": tk.VocabSize(),
	}
	b, _ := json.Marshal(result)
	return string(b)
}

// initModel loads transformer model from config and safetensors weights
func initModel(this js.Value, args []js.Value) interface{} {
	if len(args) < 2 {
		return `{"error": "Expected configJSON and weightsArrayBuffer"}`
	}

	configJSON := args[0].String()
	weightsJS := args[1]

	// Parse config to get hiddenSize and vocabSize for generation
	var config struct {
		HiddenSize int `json:"hidden_size"`
		VocabSize  int `json:"vocab_size"`
	}
	if err := json.Unmarshal([]byte(configJSON), &config); err != nil {
		return fmt.Sprintf(`{"error": "Failed to parse config: %v"}`, err)
	}

	hiddenSize = config.HiddenSize
	vocabSize = config.VocabSize

	// Copy weights from JS ArrayBuffer
	weightsLen := weightsJS.Get("byteLength").Int()
	weightsBytes := make([]byte, weightsLen)
	js.CopyBytesToGo(weightsBytes, js.Global().Get("Uint8Array").New(weightsJS))

	fmt.Printf("📦 Received %d bytes of weights\n", weightsLen)

	// Use LoadTransformerFromBytes which properly handles network construction
	var err error
	network, err = nn.LoadTransformerFromBytes([]byte(configJSON), weightsBytes)
	if err != nil {
		return fmt.Sprintf(`{"error": "Failed to load transformer: %v"}`, err)
	}

	// Parse tensors to extract embeddings and final norm (not included in network)
	tensors, err := parseSafetensorsBytes(weightsBytes)
	if err != nil {
		return fmt.Sprintf(`{"error": "Failed to parse tensors for embeddings: %v"}`, err)
	}

	// Load embeddings
	embeddings = tryLoadTensor(tensors, []string{
		"model.embed_tokens.weight",
		"transformer.wte.weight",
		"embeddings.weight",
		"embed_tokens.weight",
	})
	if embeddings == nil {
		return `{"error": "Could not find embeddings tensor"}`
	}

	// Load final norm
	finalNorm = tryLoadTensor(tensors, []string{
		"model.norm.weight",
		"transformer.ln_f.weight",
		"ln_f.weight",
		"norm.weight",
	})

	modelReady = true

	result := map[string]interface{}{
		"success":    true,
		"hiddenSize": hiddenSize,
		"vocabSize":  vocabSize,
		"layers":     len(network.Layers),
	}
	b, _ := json.Marshal(result)
	return string(b)
}

// generate produces tokens from a prompt
func generate(this js.Value, args []js.Value) interface{} {
	if !modelReady {
		return `{"error": "Model not loaded"}`
	}
	if tk == nil {
		return `{"error": "Tokenizer not loaded"}`
	}
	if len(args) < 2 {
		return `{"error": "Expected prompt and maxTokens"}`
	}

	prompt := args[0].String()
	maxTokens := args[1].Int()

	// Tokenize prompt
	inputIDs := tk.Encode(prompt, false)
	tokens := make([]int, len(inputIDs))
	for i, id := range inputIDs {
		tokens[i] = int(id)
	}

	generatedTokens := []int{}

	// Generate tokens
	for step := 0; step < maxTokens; step++ {
		nextToken, err := generateNextToken(tokens)
		if err != nil {
			return fmt.Sprintf(`{"error": "Generation error: %v"}`, err)
		}

		generatedTokens = append(generatedTokens, nextToken)
		tokens = append(tokens, nextToken)
	}

	// Decode generated text
	generatedText := tk.Decode(toUint32(generatedTokens), false)

	result := map[string]interface{}{
		"prompt":          prompt,
		"inputTokens":     inputIDs,
		"generatedTokens": generatedTokens,
		"fullSequence":    tokens,
		"generatedText":   generatedText,
	}
	b, _ := json.Marshal(result)
	return string(b)
}

// Streaming generation state
var (
	streamTokens []int
	streamStep   int
	streamMax    int
)

// initGeneration initializes streaming generation
func initGeneration(this js.Value, args []js.Value) interface{} {
	if !modelReady {
		return `{"error": "Model not loaded"}`
	}
	if tk == nil {
		return `{"error": "Tokenizer not loaded"}`
	}
	if len(args) < 2 {
		return `{"error": "Expected prompt and maxTokens"}`
	}

	prompt := args[0].String()
	maxTokens := args[1].Int()

	// Tokenize prompt
	inputIDs := tk.Encode(prompt, false)
	streamTokens = make([]int, len(inputIDs))
	for i, id := range inputIDs {
		streamTokens[i] = int(id)
	}

	streamStep = 0
	streamMax = maxTokens

	result := map[string]interface{}{
		"success":     true,
		"inputTokens": inputIDs,
		"maxTokens":   maxTokens,
	}
	b, _ := json.Marshal(result)
	return string(b)
}

// generateStep generates a single token (for streaming)
func generateStep(this js.Value, args []js.Value) interface{} {
	if streamStep >= streamMax {
		return `{"done": true}`
	}

	nextToken, err := generateNextToken(streamTokens)
	if err != nil {
		return fmt.Sprintf(`{"error": "Generation error: %v"}`, err)
	}

	streamTokens = append(streamTokens, nextToken)
	streamStep++

	// Decode just this token
	tokenText := tk.Decode([]uint32{uint32(nextToken)}, false)

	result := map[string]interface{}{
		"done":      streamStep >= streamMax,
		"step":      streamStep,
		"tokenId":   nextToken,
		"tokenText": tokenText,
	}
	b, _ := json.Marshal(result)
	return string(b)
}

// decodeToken decodes a single token ID
func decodeToken(this js.Value, args []js.Value) interface{} {
	if tk == nil {
		return ""
	}
	if len(args) < 1 {
		return ""
	}
	tokenID := args[0].Int()
	return tk.Decode([]uint32{uint32(tokenID)}, false)
}
// trainStep performs one training step on the provided text using TweenStep
func trainStep(this js.Value, args []js.Value) interface{} {
	if !modelReady {
		return `{"error": "Model not loaded"}`
	}
	if tk == nil {
		return `{"error": "Tokenizer not loaded"}`
	}
	if len(args) < 2 {
		return `{"error": "Expected text and learningRate"}`
	}

	text := args[0].String()
	learningRate := float32(args[1].Float())

	// Tokenize training text
	inputIDs := tk.Encode(text, false)
	tokens := make([]int, len(inputIDs))
	for i, id := range inputIDs {
		tokens[i] = int(id)
	}

	if len(tokens) < 2 {
		return `{"error": "Text too short for training"}`
	}

	// Create TweenState for training (pattern from quick_finetune.go)
	ts := nn.NewTweenState(network, nil)
	ts.Config.UseChainRule = true
	totalLayers := network.TotalLayers()

	totalLoss := float32(0)
	
	// Train on each consecutive token pair
	for pos := 0; pos < len(tokens)-1; pos++ {
		contextTokens := tokens[:pos+1]
		targetToken := tokens[pos+1]

		// Embed context tokens
		inputData := make([]float32, len(contextTokens)*hiddenSize)
		for t, tokenID := range contextTokens {
			for d := 0; d < hiddenSize; d++ {
				inputData[t*hiddenSize+d] = embeddings[tokenID*hiddenSize+d]
			}
		}

		// Forward pass through tween state
		network.BatchSize = 1
		output := ts.ForwardPass(network, inputData)

		if len(output) == 0 {
			continue
		}

		// Create target: we want the last hidden state to project to target token
		// Since network output is hiddenSize, we create a gradient that pushes
		// the hidden state toward the embedding of the target token
		target := make([]float32, len(output))
		lastIdx := (len(contextTokens) - 1) * hiddenSize
		if lastIdx+hiddenSize <= len(output) && targetToken < vocabSize {
			// Set target to be the embedding of the target token
			for d := 0; d < hiddenSize && lastIdx+d < len(target); d++ {
				target[lastIdx+d] = embeddings[targetToken*hiddenSize+d]
			}
		}

		// Compute output gradient (target - output)
		outputGrad := make([]float32, len(output))
		loss := float32(0)
		for i := range outputGrad {
			outputGrad[i] = target[i] - output[i]
			loss += outputGrad[i] * outputGrad[i]
		}
		totalLoss += loss

		// Set gradients for chain rule
		ts.ChainGradients[totalLayers] = outputGrad
		ts.BackwardTargets[totalLayers] = target

		// Update weights using TweenChain
		ts.TweenWeightsChainRule(network, learningRate)
	}

	avgLoss := totalLoss / float32(len(tokens)-1)

	result := map[string]interface{}{
		"success":  true,
		"tokens":   len(tokens),
		"avgLoss":  avgLoss,
	}
	b, _ := json.Marshal(result)
	return string(b)
}

// encode tokenizes text
func encode(this js.Value, args []js.Value) interface{} {
	if tk == nil {
		return `{"error": "Tokenizer not loaded"}`
	}
	if len(args) < 1 {
		return `{"error": "Expected text"}`
	}

	text := args[0].String()
	tokens := tk.Encode(text, false)

	b, _ := json.Marshal(tokens)
	return string(b)
}

// decode converts tokens to text
func decode(this js.Value, args []js.Value) interface{} {
	if tk == nil {
		return `{"error": "Tokenizer not loaded"}`
	}
	if len(args) < 1 {
		return `{"error": "Expected token IDs JSON array"}`
	}

	var tokenIDs []uint32
	if err := json.Unmarshal([]byte(args[0].String()), &tokenIDs); err != nil {
		return fmt.Sprintf(`{"error": "Invalid token IDs: %v"}`, err)
	}

	text := tk.Decode(tokenIDs, false)
	return text
}

// isModelReady checks if model is loaded
func isModelReady(this js.Value, args []js.Value) interface{} {
	return modelReady
}

// generateNextToken generates the next token given context
func generateNextToken(tokens []int) (int, error) {
	// Embed tokens
	input := make([]float32, len(tokens)*hiddenSize)
	for t, tokenID := range tokens {
		if tokenID >= vocabSize || tokenID < 0 {
			return 0, fmt.Errorf("invalid token ID: %d", tokenID)
		}
		for d := 0; d < hiddenSize; d++ {
			input[t*hiddenSize+d] = embeddings[tokenID*hiddenSize+d]
		}
	}

	network.BatchSize = 1
	output, _ := network.ForwardCPU(input)

	// Apply final norm if available
	var normalized []float32
	if finalNorm != nil {
		finalNormConfig := &nn.LayerConfig{
			Type:     nn.LayerRMSNorm,
			NormSize: hiddenSize,
			Gamma:    finalNorm,
			Epsilon:  1e-6,
		}
		normalized = nn.RmsNormForwardCPU(output, nil, finalNormConfig, len(tokens))
	} else {
		normalized = output
	}

	// Extract last token
	lastIdx := (len(tokens) - 1) * hiddenSize
	lastTokenNormalized := normalized[lastIdx : lastIdx+hiddenSize]

	// LM head projection (tied weights)
	logits := make([]float32, vocabSize)
	for v := 0; v < vocabSize; v++ {
		var sum float32
		for d := 0; d < hiddenSize; d++ {
			sum += lastTokenNormalized[d] * embeddings[v*hiddenSize+d]
		}
		logits[v] = sum
	}

	// Greedy sampling (argmax)
	maxIdx := 0
	maxVal := logits[0]
	for j := 1; j < vocabSize; j++ {
		if logits[j] > maxVal {
			maxVal = logits[j]
			maxIdx = j
		}
	}

	return maxIdx, nil
}

// parseSafetensorsBytes parses safetensors format from bytes
func parseSafetensorsBytes(data []byte) (map[string][]float32, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("file too small")
	}

	// Read header size (first 8 bytes, little-endian uint64)
	headerSize := binary.LittleEndian.Uint64(data[:8])
	if headerSize > uint64(len(data)-8) {
		return nil, fmt.Errorf("invalid header size")
	}

	// Parse header JSON
	headerJSON := data[8 : 8+headerSize]
	var header map[string]interface{}
	if err := json.Unmarshal(headerJSON, &header); err != nil {
		return nil, fmt.Errorf("failed to parse header: %v", err)
	}

	tensors := make(map[string][]float32)
	dataStart := 8 + headerSize

	for name, meta := range header {
		if name == "__metadata__" {
			continue
		}

		metaMap, ok := meta.(map[string]interface{})
		if !ok {
			continue
		}

		dtype, _ := metaMap["dtype"].(string)
		offsets, _ := metaMap["data_offsets"].([]interface{})
		if len(offsets) != 2 {
			continue
		}

		start := uint64(offsets[0].(float64))
		end := uint64(offsets[1].(float64))

		tensorData := data[dataStart+start : dataStart+end]

		switch dtype {
		case "F32":
			count := len(tensorData) / 4
			floats := make([]float32, count)
			for i := 0; i < count; i++ {
				bits := binary.LittleEndian.Uint32(tensorData[i*4:])
				floats[i] = math.Float32frombits(bits)
			}
			tensors[name] = floats
		case "F16", "BF16":
			count := len(tensorData) / 2
			floats := make([]float32, count)
			for i := 0; i < count; i++ {
				bits := binary.LittleEndian.Uint16(tensorData[i*2:])
				if dtype == "BF16" {
					floats[i] = math.Float32frombits(uint32(bits) << 16)
				} else {
					floats[i] = float16ToFloat32(bits)
				}
			}
			tensors[name] = floats
		}
	}

	return tensors, nil
}

func float16ToFloat32(h uint16) float32 {
	sign := uint32((h >> 15) & 1)
	exp := uint32((h >> 10) & 0x1F)
	mant := uint32(h & 0x3FF)

	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
	} else if exp == 31 {
		if mant == 0 {
			return math.Float32frombits((sign << 31) | 0x7F800000)
		}
		return math.Float32frombits((sign << 31) | 0x7FC00000)
	}

	exp = exp + 127 - 15
	mant = mant << 13
	return math.Float32frombits((sign << 31) | (exp << 23) | mant)
}

func tryLoadTensor(tensors map[string][]float32, keys []string) []float32 {
	for _, key := range keys {
		if tensor, exists := tensors[key]; exists {
			return tensor
		}
	}
	return nil
}

func toUint32(ints []int) []uint32 {
	result := make([]uint32, len(ints))
	for i, v := range ints {
		result[i] = uint32(v)
	}
	return result
}
