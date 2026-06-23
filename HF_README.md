---
license: apache-2.0
tags:
  - audio
  - text-to-audio
  - gguf
  - ggml
base_model: mispeech/Dasheng-AudioGen
---

# Dasheng-AudioGen GGUF

GGUF-converted weights for [mispeech/Dasheng-AudioGen](https://huggingface.co/mispeech/Dasheng-AudioGen)


**Dasheng-AudioGen** is a unified audio generation model that can jointly synthesize **intelligible speech, music, sound effects, and environmental acoustics** from text descriptions.

<p align="center">
  <video
    src="https://github.com/user-attachments/assets/497f5688-8731-4830-8ee7-b9cf4234d900"
    controls
    autoplay
    muted
    loop
    playsinline
    width="85%">
  </video>
</p>


## Model Variants

| Variant | DiT Model | Total Size | Quality |
|---------|-----------|------------|---------|
| **F16** (default) | `dit.gguf` | ~5.7GB | Best |
| **Q8** | `dit_Q8_0.gguf` | ~3.8GB | Great |
| **Q4** | `dit_Q4_0.gguf` | ~2.8GB | Good |

All variants include the same T5 encoder, vocoder, and tokenizer.

## Files

| File | Description | Size |
|------|-------------|------|
| `t5_encoder.gguf` | T5 text encoder (F32) | 1.3GB |
| `dit.gguf` | DiT backbone (F16) | 4.1GB |
| `dit_Q8_0.gguf` | DiT backbone (Q8 quantized) | 2.2GB |
| `dit_Q4_0.gguf` | DiT backbone (Q4 quantized) | 1.2GB |
| `vocoder.gguf` | Vocoder decoder (F16) | 332MB |
| `spiece.model` | SentencePiece tokenizer | 792KB |

## Usage with audiogen.cpp

```bash
# Clone and build
git clone https://github.com/audiohacking/audiogen.cpp
cd audiogen.cpp
git submodule update --init --recursive
make metal  # or: make cpu

# Download models (choose one)
make download-models      # F16 (~5.7GB) - best quality
make download-models-q8   # Q8 (~3.8GB) - great quality
make download-models-q4   # Q4 (~2.8GB) - good quality

# Generate audio
./build-metal/dasheng-audiogen \
    models/t5_encoder.gguf models/dit.gguf models/vocoder.gguf models/spiece.model \
    --caption "A dog barking loudly" --output output.wav
```

## Prompt Format

Supports the same prompt tags as the original model:

| Flag | Tag | Description |
|------|-----|-------------|
| `--caption` | `<\|caption\|>` | Main audio description (required) |
| `--speech` | `<\|speech\|>` | Speech characteristics |
| `--asr` | `<\|asr\|>` | Text to speak |
| `--sfx` | `<\|sfx\|>` | Sound effects |
| `--music` | `<\|music\|>` | Music description |
| `--env` | `<\|env\|>` | Environment/ambience |

### Example: Complex multi-tag prompt

```bash
./build-metal/dasheng-audiogen \
    models/t5_encoder.gguf models/dit.gguf models/vocoder.gguf models/spiece.model \
    --caption "A gritty detective narrating" \
    --speech "gritty deep male voice" \
    --asr "The city never sleeps, but it sure knows how to cry." \
    --sfx "heavy rain hitting pavement" \
    --music "melancholic solo saxophone" \
    --env "distant urban ambience" \
    --output noir_detective.wav
```

## Generation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--steps N` | 25 | Number of diffusion steps |
| `--duration SECS` | 10 | Audio duration in seconds |
| `--cfg SCALE` | 3.0 | Classifier-free guidance scale |
| `--sway COEF` | -1.0 | Sway sampling coefficient |
| `--seed N` | random | Random seed for reproducibility |
| `--threads N` | 4 | CPU threads |

## Performance

| Backend | 25 Steps | Hardware |
|---------|----------|----------|
| Metal GPU | ~6s | Apple M3 Ultra |
| CPU | ~163s | Apple M3 Ultra (16 threads) |

## Original Model

This is a GGUF conversion of [mispeech/Dasheng-AudioGen](https://huggingface.co/mispeech/Dasheng-AudioGen). Please refer to the original model card for more details about the model architecture and training.

## License

Apache-2.0 - Same as the original model
