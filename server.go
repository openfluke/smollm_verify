package main

import (
	"encoding/json"
	"fmt"
	"io"
	//"math"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/openfluke/loom/nn"
	"github.com/openfluke/loom/tokenizer"
)

const (
	ModelRepo  = "HuggingFaceTB/SmolLM2-135M-Instruct"
	ModelDir   = "models/SmolLM2-135M-Instruct"
	MaxGenLen  = 25
	ServerPort = ":8080"  // Single server for everything
)

// Global state
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
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🧪 SmolLM2-135M-Instruct Backend Server                                ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════╝")

	// Load model (blocking - wait for it to be ready)
	fmt.Println("⏳ Loading model... please wait")
	if err := loadModel(); err != nil {
		fmt.Printf("❌ Failed to load model: %v\n", err)
		return
	}
	modelReady = true
	fmt.Println("✅ Model ready for inference!")

	// Setup HTTP handlers
	http.HandleFunc("/api/generate", handleGenerate)
	http.HandleFunc("/api/status", handleStatus)
	http.HandleFunc("/api/train", handleTrain)
	
	// Serve static files from web/ folder
	http.Handle("/web/", http.StripPrefix("/web/", http.FileServer(http.Dir("web"))))
	
	// Serve model files
	http.Handle("/models/", http.StripPrefix("/models/", http.FileServer(http.Dir("models"))))
	
	// Redirect root to web
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			http.Redirect(w, r, "/web/", http.StatusFound)
			return
		}
		http.NotFound(w, r)
	})

	fmt.Printf("\n🌐 Server ready at http://localhost%s\n", ServerPort)
	fmt.Println("   Open: http://localhost:8080/web/")
	fmt.Println("\n   Endpoints:")
	fmt.Println("   - GET /api/status - Check if model is ready")
	fmt.Println("   - GET /api/generate?prompt=...&max_tokens=25 - Stream tokens via SSE")
	fmt.Println("   - POST /api/train - Train on provided text")

	if err := http.ListenAndServe(ServerPort, nil); err != nil {
		fmt.Printf("❌ Server error: %v\n", err)
	}
}

func loadModel() error {
	// Ensure model exists
	if err := ensureModel(); err != nil {
		return err
	}

	// Load tokenizer
	tokenPath := filepath.Join(ModelDir, "tokenizer.json")
	var err error
	tk, err = tokenizer.LoadFromFile(tokenPath)
	if err != nil {
		return fmt.Errorf("failed to load tokenizer: %v", err)
	}
	fmt.Printf("📦 Tokenizer loaded (vocab: %d)\n", tk.VocabSize())

	// Load transformer
	fmt.Printf("⚖️  Loading model from %s...\n", ModelDir)
	network, err = nn.LoadTransformerFromSafetensors(ModelDir)
	if err != nil {
		return fmt.Errorf("failed to load transformer: %v", err)
	}
	fmt.Printf("✅ Transformer loaded! Layers: %d\n", len(network.Layers))

	// Load embeddings & final norm
	weightsPath := filepath.Join(ModelDir, "model.safetensors")
	tensors, err := nn.LoadSafetensors(weightsPath)
	if err != nil {
		return fmt.Errorf("failed to load tensors: %v", err)
	}

	embeddings = tryLoadTensor(tensors, []string{
		"model.embed_tokens.weight",
		"transformer.wte.weight",
		"embeddings.weight",
		"embed_tokens.weight",
	})

	finalNorm = tryLoadTensor(tensors, []string{
		"model.norm.weight",
		"transformer.ln_f.weight",
		"ln_f.weight",
		"norm.weight",
	})

	if embeddings == nil {
		return fmt.Errorf("missing embeddings")
	}

	hiddenSize = network.InputSize
	vocabSize = len(embeddings) / hiddenSize

	fmt.Printf("   Hidden: %d, Vocab: %d\n", hiddenSize, vocabSize)
	return nil
}

func handleStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Content-Type", "application/json")
	
	json.NewEncoder(w).Encode(map[string]interface{}{
		"ready":      modelReady,
		"hiddenSize": hiddenSize,
		"vocabSize":  vocabSize,
		"layers":     len(network.Layers),
	})
}
// handleTrain performs training on the provided text using TweenStep
func handleTrain(w http.ResponseWriter, r *http.Request) {
	// CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		return
	}

	if !modelReady {
		http.Error(w, `{"error": "Model not ready"}`, http.StatusServiceUnavailable)
		return
	}

	// Parse request
	var req struct {
		Text         string  `json:"text"`
		Steps        int     `json:"steps"`
		LearningRate float32 `json:"learningRate"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error": "Invalid request body"}`, http.StatusBadRequest)
		return
	}

	if req.Text == "" {
		req.Text = "The AI is learning and improving."
	}
	if req.Steps == 0 {
		req.Steps = 5
	}
	if req.LearningRate == 0 {
		req.LearningRate = 0.001
	}

	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "SSE not supported", http.StatusInternalServerError)
		return
	}

	// Tokenize training text
	inputIDs := tk.Encode(req.Text, false)
	tokens := make([]int, len(inputIDs))
	for i, id := range inputIDs {
		tokens[i] = int(id)
	}

	sendSSE(w, flusher, "init", map[string]interface{}{
		"text":         req.Text,
		"tokens":       len(tokens),
		"steps":        req.Steps,
		"learningRate": req.LearningRate,
	})

	// Create TweenState for training (pattern from quick_finetune.go)
	ts := nn.NewTweenState(network, nil)
	ts.Config.UseChainRule = true
	totalLayers := network.TotalLayers()

	startTime := time.Now()

	for step := 0; step < req.Steps; step++ {
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
			ts.TweenWeightsChainRule(network, req.LearningRate)
		}

		avgLoss := totalLoss / float32(len(tokens)-1)
		
		sendSSE(w, flusher, "step", map[string]interface{}{
			"step": step + 1,
			"loss": avgLoss,
		})
	}

	elapsed := time.Since(startTime)
	sendSSE(w, flusher, "done", map[string]interface{}{
		"totalSteps": req.Steps,
		"elapsedMs":  elapsed.Milliseconds(),
	})

	fmt.Printf("🎓 Backend trained for %d steps in %v\n", req.Steps, elapsed)
}

func handleGenerate(w http.ResponseWriter, r *http.Request) {
	// CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		return
	}

	if !modelReady {
		http.Error(w, `{"error": "Model not ready"}`, http.StatusServiceUnavailable)
		return
	}

	prompt := r.URL.Query().Get("prompt")
	if prompt == "" {
		prompt = "Once upon a time"
	}

	maxTokens := 25
	if mt := r.URL.Query().Get("max_tokens"); mt != "" {
		fmt.Sscanf(mt, "%d", &maxTokens)
	}

	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "SSE not supported", http.StatusInternalServerError)
		return
	}

	// Tokenize prompt
	inputIDs := tk.Encode(prompt, false)
	tokens := make([]int, len(inputIDs))
	for i, id := range inputIDs {
		tokens[i] = int(id)
	}

	// Send initial event
	sendSSE(w, flusher, "init", map[string]interface{}{
		"prompt":      prompt,
		"inputTokens": inputIDs,
		"maxTokens":   maxTokens,
	})

	// Generate tokens one at a time
	startTime := time.Now()
	for step := 0; step < maxTokens; step++ {
		nextToken, err := generateNextToken(tokens)
		if err != nil {
			sendSSE(w, flusher, "error", map[string]interface{}{
				"error": err.Error(),
			})
			return
		}

		tokens = append(tokens, nextToken)
		tokenText := tk.Decode([]uint32{uint32(nextToken)}, false)

		sendSSE(w, flusher, "token", map[string]interface{}{
			"step":      step + 1,
			"tokenId":   nextToken,
			"tokenText": tokenText,
			"done":      step+1 >= maxTokens,
		})
	}

	// Send completion
	elapsed := time.Since(startTime)
	sendSSE(w, flusher, "done", map[string]interface{}{
		"totalTokens": maxTokens,
		"elapsedMs":   elapsed.Milliseconds(),
		"tokensPerSec": float64(maxTokens) / elapsed.Seconds(),
	})
}

func sendSSE(w http.ResponseWriter, flusher http.Flusher, event string, data interface{}) {
	jsonData, _ := json.Marshal(data)
	fmt.Fprintf(w, "event: %s\n", event)
	fmt.Fprintf(w, "data: %s\n\n", jsonData)
	flusher.Flush()
}

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

	// Apply final norm
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

	// LM head projection
	logits := make([]float32, vocabSize)
	for v := 0; v < vocabSize; v++ {
		var sum float32
		for d := 0; d < hiddenSize; d++ {
			sum += lastTokenNormalized[d] * embeddings[v*hiddenSize+d]
		}
		logits[v] = sum
	}

	// Greedy argmax
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

func ensureModel() error {
	os.MkdirAll(ModelDir, 0755)

	files := []string{"config.json", "tokenizer.json", "model.safetensors"}
	baseURL := "https://huggingface.co/" + ModelRepo + "/resolve/main/"

	for _, file := range files {
		path := filepath.Join(ModelDir, file)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			fmt.Printf("⬇️  Downloading %s...\n", file)
			url := baseURL + file
			if err := downloadFile(path, url); err != nil {
				return err
			}
		}
	}
	return nil
}

func downloadFile(filepath string, url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

func tryLoadTensor(tensors map[string][]float32, keys []string) []float32 {
	for _, key := range keys {
		if tensor, exists := tensors[key]; exists {
			return tensor
		}
	}
	return nil
}
