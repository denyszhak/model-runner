package sglang

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/docker/model-runner/pkg/diskusage"
	"github.com/docker/model-runner/pkg/inference"
	"github.com/docker/model-runner/pkg/inference/backends"
	"github.com/docker/model-runner/pkg/inference/models"
	"github.com/docker/model-runner/pkg/inference/platform"
	"github.com/docker/model-runner/pkg/logging"
)

const (
	// Name is the backend name.
	Name      = "sglang"
	sglangDir = "/opt/sglang-env/bin"
)

var ErrorNotFound = errors.New("SGLang binary not found")

// sglang is the SGLang-based backend implementation.
type sglang struct {
	// log is the associated logger.
	log logging.Logger
	// modelManager is the shared model manager.
	modelManager *models.Manager
	// serverLog is the logger to use for the SGLang server process.
	serverLog logging.Logger
	// config is the configuration for the SGLang backend.
	config *Config
	// status is the state in which the SGLang backend is in.
	status string
	// pythonPath is the path to the python3 binary.
	pythonPath string
}

// New creates a new SGLang-based backend.
func New(log logging.Logger, modelManager *models.Manager, serverLog logging.Logger, conf *Config) (inference.Backend, error) {
	// If no config is provided, use the default configuration
	if conf == nil {
		conf = NewDefaultSGLangConfig()
	}

	return &sglang{
		log:          log,
		modelManager: modelManager,
		serverLog:    serverLog,
		config:       conf,
		status:       "not installed",
	}, nil
}

// Name implements inference.Backend.Name.
func (s *sglang) Name() string {
	return Name
}

func (s *sglang) UsesExternalModelManagement() bool {
	return false
}

func (s *sglang) Install(_ context.Context, _ *http.Client) error {
	if !platform.SupportsSGLang() {
		return errors.New("not implemented")
	}

	// Check if we're in Docker environment with pre-installed SGLang
	sglangBinaryPath := s.binaryPath()
	if _, statErr := os.Stat(sglangBinaryPath); statErr == nil {
		// Docker environment - check version from file
		versionPath := filepath.Join(filepath.Dir(sglangDir), "version")
		versionBytes, err := os.ReadFile(versionPath)
		if err != nil {
			s.log.Warnf("could not get sglang version: %v", err)
			s.status = "running sglang version: unknown"
		} else {
			s.status = fmt.Sprintf("running sglang version: %s", strings.TrimSpace(string(versionBytes)))
		}
		return nil
	} else if !errors.Is(statErr, fs.ErrNotExist) {
		// Host environment - some other error checking binary
		return fmt.Errorf("failed to check SGLang binary: %w", statErr)
	}

	// Host environment - check for Python and sglang package

	// Look for python3
	pythonPath, err := exec.LookPath("python3")
	if err != nil {
		s.status = ErrorNotFound.Error()
		return ErrorNotFound
	}

	s.pythonPath = pythonPath

	// Check if sglang package is installed
	cmd := exec.Command(pythonPath, "-c", "import sglang")
	if err := cmd.Run(); err != nil {
		s.status = "sglang package not installed"
		s.log.Warnf("sglang package not found. Install with: pip install sglang[all]")
		return fmt.Errorf("sglang package not installed: %w", err)
	}

	// Get SGLang version
	cmd = exec.Command(pythonPath, "-c", "import sglang; print(sglang.__version__)")
	output, err := cmd.Output()
	if err != nil {
		s.log.Warnf("could not get sglang version: %v", err)
		s.status = "running sglang version: unknown"
	} else {
		s.status = fmt.Sprintf("running sglang version: %s", strings.TrimSpace(string(output)))
	}

	return nil
}

func (s *sglang) Run(ctx context.Context, socket, model string, modelRef string, mode inference.BackendMode, backendConfig *inference.BackendConfiguration) error {
	if !platform.SupportsSGLang() {
		s.log.Warn("SGLang backend is not yet supported")
		return errors.New("not implemented")
	}

	bundle, err := s.modelManager.GetBundle(model)
	if err != nil {
		return fmt.Errorf("failed to get model: %w", err)
	}

	args, err := s.config.GetArgs(bundle, socket, mode, backendConfig)
	if err != nil {
		return fmt.Errorf("failed to get SGLang arguments: %w", err)
	}

	// Add served model name
	args = append(args, "--served-model-name", model, modelRef)

	// Determine binary path - use Docker installation if available, otherwise use Python
	binaryPath := s.binaryPath()
	sandboxPath := sglangDir
	if _, err := os.Stat(binaryPath); errors.Is(err, fs.ErrNotExist) {
		// Use Python installation
		binaryPath = s.pythonPath
		sandboxPath = ""
	}

	return backends.RunBackend(ctx, backends.RunnerConfig{
		BackendName:     "SGLang",
		Socket:          socket,
		BinaryPath:      binaryPath,
		SandboxPath:     sandboxPath,
		SandboxConfig:   "",
		Args:            args,
		Logger:          s.log,
		ServerLogWriter: s.serverLog.Writer(),
	})
}

func (s *sglang) Status() string {
	return s.status
}

func (s *sglang) GetDiskUsage() (int64, error) {
	// Check if Docker installation exists
	if _, err := os.Stat(sglangDir); err == nil {
		size, err := diskusage.Size(sglangDir)
		if err != nil {
			return 0, fmt.Errorf("error while getting store size: %w", err)
		}
		return size, nil
	}
	// Python installation doesn't have a dedicated directory
	return 0, nil
}

func (s *sglang) GetRequiredMemoryForModel(_ context.Context, _ string, _ *inference.BackendConfiguration) (inference.RequiredMemory, error) {
	if !platform.SupportsSGLang() {
		return inference.RequiredMemory{}, errors.New("not implemented")
	}

	// SGLang has similar memory requirements to vLLM
	// TODO: Implement accurate memory estimation based on model size
	return inference.RequiredMemory{
		RAM:  1,
		VRAM: 1,
	}, nil
}

func (s *sglang) binaryPath() string {
	return filepath.Join(sglangDir, "sglang")
}
