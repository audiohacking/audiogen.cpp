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
# Default: F16 models (~5.7GB total)
.PHONY: download-models
download-models:
	@mkdir -p models
	@echo "Downloading F16 models (~5.7GB)..."
	@$(call hf_download,t5_encoder.gguf dit.gguf vocoder.gguf spiece.model)
	@echo "Models downloaded to models/"
	@ls -lh models/

# Q8 quantized models (~3.8GB total) - good quality, smaller size
.PHONY: download-models-q8
download-models-q8:
	@mkdir -p models
	@echo "Downloading Q8 models (~3.8GB)..."
	@$(call hf_download,t5_encoder.gguf dit_Q8_0.gguf vocoder.gguf spiece.model)
	@mv models/dit_Q8_0.gguf models/dit.gguf 2>/dev/null || true
	@echo "Models downloaded to models/"
	@ls -lh models/

# Q4 quantized models (~2.8GB total) - smallest size, slight quality reduction
.PHONY: download-models-q4
download-models-q4:
	@mkdir -p models
	@echo "Downloading Q4 models (~2.8GB)..."
	@$(call hf_download,t5_encoder.gguf dit_Q4_0.gguf vocoder.gguf spiece.model)
	@mv models/dit_Q4_0.gguf models/dit.gguf 2>/dev/null || true
	@echo "Models downloaded to models/"
	@ls -lh models/

# Helper function for HF download
HF_REPO_URL := https://huggingface.co/audiohacking/dasheng-audiogen-gguf/resolve/main

define hf_download
	@if command -v hf >/dev/null 2>&1; then \
		for f in $(1); do hf download audiohacking/dasheng-audiogen-gguf $$f --local-dir models/; done; \
	elif command -v huggingface-cli >/dev/null 2>&1 && huggingface-cli download --help >/dev/null 2>&1; then \
		for f in $(1); do huggingface-cli download audiohacking/dasheng-audiogen-gguf $$f --local-dir models/; done; \
	elif command -v curl >/dev/null 2>&1; then \
		for f in $(1); do echo "Downloading $$f..."; curl -L -o models/$$f $(HF_REPO_URL)/$$f; done; \
	elif command -v wget >/dev/null 2>&1; then \
		for f in $(1); do echo "Downloading $$f..."; wget -O models/$$f $(HF_REPO_URL)/$$f; done; \
	else \
		echo "Error: No download tool found. Install curl, wget, or huggingface_hub[cli]"; \
		exit 1; \
	fi
endef

# Convert models to GGUF (alternative to downloading)
.PHONY: convert
convert:
	@mkdir -p models
	@echo "Converting T5 encoder..."
	python convert/convert_t5_encoder.py google/flan-t5-large -o models/t5_encoder.gguf --dtype f32
	@echo "Converting DiT (F16)..."
	python convert/convert_dit.py mispeech/Dasheng-AudioGen -o models/dit.gguf --dtype f16
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

# Convert with Q8 quantization
.PHONY: convert-q8
convert-q8:
	@mkdir -p models
	@echo "Converting T5 encoder..."
	python convert/convert_t5_encoder.py google/flan-t5-large -o models/t5_encoder.gguf --dtype f32
	@echo "Converting DiT (Q8_0)..."
	python convert/convert_dit.py mispeech/Dasheng-AudioGen -o models/dit.gguf --dtype q8_0
	@echo "Converting Vocoder..."
	python convert/convert_vocoder.py mispeech/dashengtokenizer -o models/vocoder.gguf
	@cp ~/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/*/spiece.model models/ 2>/dev/null || true
	@ls -lh models/

# Convert with Q4 quantization
.PHONY: convert-q4
convert-q4:
	@mkdir -p models
	@echo "Converting T5 encoder..."
	python convert/convert_t5_encoder.py google/flan-t5-large -o models/t5_encoder.gguf --dtype f32
	@echo "Converting DiT (Q4_0)..."
	python convert/convert_dit.py mispeech/Dasheng-AudioGen -o models/dit.gguf --dtype q4_0
	@echo "Converting Vocoder..."
	python convert/convert_vocoder.py mispeech/dashengtokenizer -o models/vocoder.gguf
	@cp ~/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/*/spiece.model models/ 2>/dev/null || true
	@ls -lh models/

# Quick test run
.PHONY: test-e2e
test-e2e: cpu
	$(BUILD_DIR_CPU)/dasheng-audiogen models/t5_encoder.gguf models/dit.gguf models/vocoder.gguf models/spiece.model \
		--caption "A dog barking" --steps 10 --output output/test.wav

# Show help
.PHONY: help
help:
	@echo "audiogen.cpp build targets:"
	@echo ""
	@echo "Build:"
	@echo "  make cpu              Build CPU-only version"
	@echo "  make metal            Build with Metal backend (default, macOS)"
	@echo "  make cuda             Build with CUDA backend"
	@echo "  make build-all        Build both CPU and Metal versions"
	@echo ""
	@echo "Models:"
	@echo "  make download-models     Download F16 GGUF models (~5.7GB)"
	@echo "  make download-models-q8  Download Q8 quantized models (~3.8GB)"
	@echo "  make download-models-q4  Download Q4 quantized models (~2.8GB)"
	@echo "  make convert             Convert models from original weights (F16)"
	@echo "  make convert-q8          Convert with Q8 quantization"
	@echo "  make convert-q4          Convert with Q4 quantization"
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
