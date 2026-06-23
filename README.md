# audiogen.cpp

High-performance C++17 GGML inference engine for local text-to-audio generation using [`AudioGen`](https://huggingface.co/mispeech/Dasheng-AudioGen) a 2B-parameter flow-matching model using pre-converted GGUF weights [`audiohacking/dasheng-audiogen-gguf`](https://huggingface.co/audiohacking/dasheng-audiogen-gguf) _(fp16, q8, q4)_

## Quick Start

```bash
# Clone and build
git clone https://github.com/audiohacking/audiogen.cpp
cd audiogen.cpp
git submodule update --init --recursive
make metal    # or: make cpu

# Download models (~6GB)
make download-models

# Generate audio
./build-metal/dasheng-audiogen \
    models/t5_encoder.gguf models/dit.gguf models/vocoder.gguf models/spiece.model \
    --caption "A dog barking loudly" --output dog.wav
```

## Performance

| Backend | 10s @ 25 Steps | Hardware |
|---------|----------|----------|
| Metal GPU | ~6s | Apple M3 Ultra |
| CPU | ~163s | Apple M3 Ultra (16 threads) |

## Build

```bash
make metal        # Metal backend (macOS)
make cpu          # CPU-only build
make cuda         # CUDA backend (Linux)
```

## Download Models

```bash
# F16 models (~5.7GB) - best quality
make download-models

# Q8 models (~3.8GB) - good quality, 33% smaller
make download-models-q8

# Q4 models (~2.8GB) - smallest, 51% smaller
make download-models-q4
```

Or manually:
```bash
hf download audiohacking/dasheng-audiogen-gguf --local-dir models/
```

## Usage

### Basic

```bash
./build-metal/dasheng-audiogen \
    models/t5_encoder.gguf models/dit.gguf models/vocoder.gguf models/spiece.model \
    --caption "Rain falling on a window" --output rain.wav
```

### Prompt Composition

Combine tags for richer generation:

```bash
./build-metal/dasheng-audiogen \
    models/t5_encoder.gguf models/dit.gguf models/vocoder.gguf models/spiece.model \
    --caption "A peaceful morning" \
    --music "soft piano melody" \
    --env "birds chirping" \
    --output morning.wav
```

Tags: `--caption`, `--speech`, `--asr`, `--sfx`, `--music`, `--env`

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--steps N` | 25 | Diffusion steps |
| `--duration SECS` | 10 | Audio duration in seconds |
| `--cfg SCALE` | 3.0 | Guidance scale |
| `--seed N` | random | Random seed |
| `--threads N` | 4 | CPU threads |

### Batch Processing

```bash
cat > prompts.txt << 'EOF'
<|caption|> A dog barking
<|caption|> Thunder rolling
EOF

./build-metal/dasheng-audiogen \
    models/t5_encoder.gguf models/dit.gguf models/vocoder.gguf models/spiece.model \
    --batch prompts.txt --output-dir output/
```

## Converting Models

To convert from original weights instead of downloading:

```bash
pip install -r convert/requirements.txt
make convert
```

## License

MIT
