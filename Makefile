# Makefile for dasheng-audiogen
# Wraps CMake commands for convenience

BUILD_DIR_CPU   := build-cpu
BUILD_DIR_METAL := build-metal
BUILD_DIR_CUDA  := build-cuda

CMAKE_FLAGS := -DCMAKE_BUILD_TYPE=Release

# Default target
.PHONY: all
all: metal

# CPU-only build
.PHONY: cpu
cpu:
	@mkdir -p $(BUILD_DIR_CPU)
	cmake -B $(BUILD_DIR_CPU) -S . $(CMAKE_FLAGS) \
		-DDASHENG_AUDIOGEN_METAL=OFF \
		-DDASHENG_AUDIOGEN_CUDA=OFF
	cmake --build $(BUILD_DIR_CPU) --parallel
	@echo "CPU build complete: $(BUILD_DIR_CPU)/dasheng-audiogen"

# Metal build (macOS)
.PHONY: metal
metal:
	@mkdir -p $(BUILD_DIR_METAL)
	cmake -B $(BUILD_DIR_METAL) -S . $(CMAKE_FLAGS) \
		-DDASHENG_AUDIOGEN_METAL=ON \
		-DDASHENG_AUDIOGEN_CUDA=OFF
	cmake --build $(BUILD_DIR_METAL) --parallel
	@echo "Metal build complete: $(BUILD_DIR_METAL)/dasheng-audiogen"

# CUDA build
.PHONY: cuda
cuda:
	@mkdir -p $(BUILD_DIR_CUDA)
	cmake -B $(BUILD_DIR_CUDA) -S . $(CMAKE_FLAGS) \
		-DDASHENG_AUDIOGEN_METAL=OFF \
		-DDASHENG_AUDIOGEN_CUDA=ON
	cmake --build $(BUILD_DIR_CUDA) --parallel
	@echo "CUDA build complete: $(BUILD_DIR_CUDA)/dasheng-audiogen"

# Build all targets
.PHONY: build-all
build-all: cpu metal

# Clean all build directories
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR_CPU) $(BUILD_DIR_METAL) $(BUILD_DIR_CUDA)

# Clean and rebuild
.PHONY: rebuild
rebuild: clean all

# Run smoke test (CPU)
.PHONY: test-smoke
test-smoke: cpu
	$(BUILD_DIR_CPU)/dasheng-audiogen --smoke-test

# Run smoke test (Metal)
.PHONY: test-smoke-metal
test-smoke-metal: metal
	$(BUILD_DIR_METAL)/dasheng-audiogen --smoke-test

# Download pre-converted models from HuggingFace
.PHONY: download-models
download-models:
	@mkdir -p models
	@echo "Downloading models from audiohacking/dasheng-audiogen-gguf..."
	@if command -v hf >/dev/null 2>&1; then \
		hf download audiohacking/dasheng-audiogen-gguf --local-dir models/ --include "*.gguf" --include "*.model"; \
	elif command -v huggingface-cli >/dev/null 2>&1; then \
		huggingface-cli download audiohacking/dasheng-audiogen-gguf --local-dir models/ --include "*.gguf" --include "*.model"; \
	else \
		echo "Error: hf or huggingface-cli not found. Install with: pip install huggingface_hub[cli]"; \
		exit 1; \
	fi
	@echo "Models downloaded to models/"
	@ls -lh models/

# Convert models to GGUF (alternative to downloading)
.PHONY: convert
convert:
	@mkdir -p models
	@echo "Converting T5 encoder (F32 to avoid precision issues)..."
	python convert/convert_t5_encoder.py google/flan-t5-large -o models/t5_encoder.gguf --dtype f32
	@echo "Converting DiT..."
	python convert/convert_dit.py mispeech/Dasheng-AudioGen -o models/dit.gguf
	@echo "Converting Vocoder..."
	python convert/convert_vocoder.py mispeech/dashengtokenizer -o models/vocoder.gguf
	@echo "Copying sentencepiece model..."
	@if [ -f ~/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/*/spiece.model ]; then \
		cp ~/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/*/spiece.model models/; \
	else \
		echo "Note: Run 'huggingface-cli download google/flan-t5-large' to get spiece.model"; \
	fi
	@echo "Models saved to models/"
	@ls -lh models/

# Quick test run
.PHONY: test-e2e
test-e2e: cpu
	$(BUILD_DIR_CPU)/dasheng-audiogen models/t5_encoder.gguf models/dit.gguf models/vocoder.gguf models/spiece.model \
		--caption "A dog barking" --steps 10 --output output/test.wav

# Show help
.PHONY: help
help:
	@echo "dasheng-audiogen build targets:"
	@echo ""
	@echo "Build:"
	@echo "  make cpu              Build CPU-only version"
	@echo "  make metal            Build with Metal backend (default, macOS)"
	@echo "  make cuda             Build with CUDA backend"
	@echo "  make build-all        Build both CPU and Metal versions"
	@echo ""
	@echo "Models:"
	@echo "  make download-models  Download pre-converted GGUF models from HuggingFace"
	@echo "  make convert          Convert models yourself from original weights"
	@echo ""
	@echo "Test:"
	@echo "  make test-smoke       Run smoke test (CPU)"
	@echo "  make test-smoke-metal Run smoke test (Metal)"
	@echo "  make test-e2e         Run end-to-end test (CPU)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove all build directories"
	@echo "  make rebuild          Clean and rebuild default target"
	@echo ""
	@echo "Build outputs:"
	@echo "  $(BUILD_DIR_CPU)/dasheng-audiogen"
	@echo "  $(BUILD_DIR_METAL)/dasheng-audiogen"
	@echo "  $(BUILD_DIR_CUDA)/dasheng-audiogen"
