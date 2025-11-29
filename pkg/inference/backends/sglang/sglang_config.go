package sglang

import (
	"fmt"
	"path/filepath"
	"strconv"

	"github.com/docker/model-runner/pkg/distribution/types"
	"github.com/docker/model-runner/pkg/inference"
)

// Config is the configuration for the SGLang backend.
type Config struct {
	// Args are the base arguments that are always included.
	Args []string
}

// NewDefaultSGLangConfig creates a new SGLangConfig with default values.
func NewDefaultSGLangConfig() *Config {
	return &Config{
		Args: []string{},
	}
}

// GetArgs implements BackendConfig.GetArgs.
func (c *Config) GetArgs(bundle types.ModelBundle, socket string, mode inference.BackendMode, config *inference.BackendConfiguration) ([]string, error) {
	// Start with the arguments from SGLangConfig
	args := append([]string{}, c.Args...)

	// SGLang uses Python module: python -m sglang.launch_server
	args = append(args, "-m", "sglang.launch_server")

	// Add model path (SGLang works with safetensors format)
	safetensorsPath := bundle.SafetensorsPath()
	if safetensorsPath == "" {
		return nil, fmt.Errorf("safetensors path required by SGLang backend")
	}
	modelPath := filepath.Dir(safetensorsPath)

	// Add model path argument
	args = append(args, "--model-path", modelPath)

	// Add socket arguments
	args = append(args, "--host", socket)

	// Add mode-specific arguments
	switch mode {
	case inference.BackendModeCompletion:
		// Default mode for SGLang
	case inference.BackendModeEmbedding:
		// SGLang supports embedding models via --is-embedding flag
		args = append(args, "--is-embedding")
	case inference.BackendModeReranking:
		// SGLang may not support reranking mode yet
		return nil, fmt.Errorf("reranking mode not supported by SGLang backend")
	default:
		return nil, fmt.Errorf("unsupported backend mode %q", mode)
	}

	// Add context-length if specified in model config or backend config
	if contextLen := GetContextLength(bundle.RuntimeConfig(), config); contextLen != nil {
		args = append(args, "--context-length", strconv.FormatUint(*contextLen, 10))
	}

	// Add arguments from backend config
	if config != nil {
		args = append(args, config.RuntimeFlags...)
	}

	return args, nil
}

// GetContextLength returns the context length (context size) from model config or backend config.
// Model config takes precedence over backend config.
// Returns nil if neither is specified (SGLang will auto-derive from model).
func GetContextLength(modelCfg types.Config, backendCfg *inference.BackendConfiguration) *uint64 {
	// Model config takes precedence
	if modelCfg.ContextSize != nil {
		return modelCfg.ContextSize
	}
	// else use backend config
	if backendCfg != nil && backendCfg.ContextSize > 0 {
		val := uint64(backendCfg.ContextSize)
		return &val
	}
	// Return nil to let SGLang auto-derive from model config
	return nil
}
