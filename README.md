# dasheng-audiogen.cpp

Standalone GGML inference engine for [`mispeech/Dasheng-AudioGen`](https://huggingface.co/mispeech/Dasheng-AudioGen),
a 2B-parameter flow-matching text-to-audio model.

Pre-converted GGUF weights available at [`audiohacking/dasheng-audiogen-gguf`](https://huggingface.co/audiohacking/dasheng-audiogen-gguf).

## Quick Start

```bash
# Clone and build
git clone https://github.com/audiohacking/dasheng-audiogen.cpp
cd dasheng-audiogen.cpp
git submodule update --init --recursive
make metal    # or: make cpu

# Download pre-converted models (~6GB)
make download-models

# Generate audio
./build-metal/dasheng-audiogen models/t5_encoder.gguf models/dit.gguf models/vocoder.gguf models/spiece.model \
    --caption "A dog barking loudly" --output output.wav
```

## Build

```bash
make cpu          # CPU-only build
make metal        # Metal backend (macOS, default)
make cuda         # CUDA backend
make build-all    # Build all available backends
```

## Download Models

Pre-converted GGUF models are hosted on Hugging Face:

```bash
# Download all models (~6GB total)
make download-models

# Or manually with hf CLI:
hf download audiohacking/dasheng-audiogen-gguf --local-dir models/
```

### Convert Models Yourself

If you prefer to convert from the original weights:

```bash
# Install dependencies
pip install -r convert/requirements.txt

# Convert all models
make convert
```

## Usage

### Basic Generation

```bash
# Using --caption (recommended)
./build-metal/dasheng-audiogen models/t5_encoder.gguf models/dit.gguf models/vocoder.gguf models/spiece.model \
    --caption "A dog barking" --output dog.wav

# Using raw prompt (must start with <|caption|>)
./build-metal/dasheng-audiogen models/t5_encoder.gguf models/dit.gguf models/vocoder.gguf models/spiece.model \
    --prompt "<|caption|> A dog barking" --output dog.wav
```

### Prompt Composition

Combine multiple tags for richer audio generation:

```bash
./build-metal/dasheng-audiogen models/t5_encoder.gguf models/dit.gguf models/vocoder.gguf models/spiece.model \
    --caption "A peaceful morning" \
    --music "soft piano melody" \
    --env "birds chirping, gentle breeze" \
    --output morning.wav
```

Available tags:
- `--caption TEXT` - Main description (required)
- `--speech TEXT` - Speech characteristics
- `--asr TEXT` - Text to speak
- `--sfx TEXT` - Sound effects
- `--music TEXT` - Music description
- `--env TEXT` - Environment/ambience

### Complex Example

```bash
./build-metal/dasheng-audiogen models/t5_encoder.gguf models/dit.gguf models/vocoder.gguf models/spiece.model \
    --caption "A gritty detective narrating" \
    --speech "gritty deep male voice" \
    --asr "The city never sleeps, but it sure knows how to cry." \
    --sfx "heavy rain hitting pavement" \
    --music "melancholic solo saxophone" \
    --env "distant urban ambience" \
    --steps 25 --cfg 5.0 \
    --output noir_detective.wav
```

### Batch Processing

```bash
# Create a batch file (one prompt per line)
cat > prompts.txt << 'EOF'
<|caption|> A dog barking
<|caption|> Rain falling on a roof
<|caption|> Jazz music playing in a cafe
EOF

# Process all prompts
./build-metal/dasheng-audiogen models/t5_encoder.gguf models/dit.gguf models/vocoder.gguf models/spiece.model \
    --batch prompts.txt --output-dir output/
```

### Generation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--steps N` | 25 | Diffusion steps (more = higher quality) |
| `--cfg SCALE` | 3.0 | Guidance scale (higher = more prompt adherence) |
| `--sway COEF` | -1.0 | Sway sampling coefficient |
| `--seed N` | random | Random seed for reproducibility |
| `--threads N` | 4 | CPU threads |

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full pipeline breakdown.

## License

MIT
