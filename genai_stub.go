//go:build !cgo

package ortgenai

import (
	"context"
	"errors"
)

var ErrNotInitialized = errors.New("ortgenai requires CGO - this is a stub build")
var ErrCGORequired = errors.New("ortgenai requires CGO to be enabled")

func IsInitialized() bool {
	return false
}

func InitializeEnvironment() error {
	return ErrCGORequired
}

func DestroyEnvironment() error {
	return ErrCGORequired
}

func SetSharedLibraryPath(path string) {}

type GenerationOptions struct {
	MaxLength int
	BatchSize int
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type SequenceDelta struct {
	Sequence int
	Tokens   string
}

type Statistics struct {
	AvgPrefillSeconds float64
	TokensPerSecond   float64
}

type Session struct{}

func (s *Session) Generate(ctx context.Context, messages [][]Message, generationOptions *GenerationOptions) (<-chan SequenceDelta, <-chan error, error) {
	return nil, nil, ErrCGORequired
}

func (s *Session) GetStatistics() Statistics {
	return Statistics{}
}

func (s *Session) Destroy() {}

func CreateGenerativeSession(modelPath string) (*Session, error) {
	return nil, ErrCGORequired
}

func CreateGenerativeSessionAdvanced(configDirectoryPath string, providers []string, providerOptions map[string]map[string]string) (*Session, error) {
	return nil, ErrCGORequired
}
