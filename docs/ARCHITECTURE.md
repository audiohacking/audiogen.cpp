# dasheng-audiogen.cpp architecture

Standalone GGML inference engine for `mispeech/Dasheng-AudioGen`, modeled on
[acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp). GGUF is used purely as
ggml's native tensor container — there is no dependency on llama.cpp's model registry
except for a small vendored T5 encoder-only graph (T5 is natively supported by llama.cpp
already, so we reuse that logic rather than reinventing it).

## Pipeline

1. **Text encoder** — `google/flan-t5-large` (`T5EncoderModel`). Standard architecture;
   handled by a minimal vendored T5 encoder-only ggml graph (no KV-cache/decoder/sampling),
   derived from llama.cpp's `LLM_ARCH_T5ENCODER` handling.
2. **Content projection** — `Linear(1024, 1024)` over the T5 hidden states
   (`content_encoder.text_encoder.proj`), part of Dasheng-AudioGen's own weights.
3. **Content adapter** — `CrossAttentionAdapter` (cross-attention from the content
   sequence to a fixed learned `instruction_embedding` vector) + `DurationPredictor`
   (Conv1d/ReLU/LayerNorm/Dropout stack) + global duration MLP + `Conv1d(1x1)` projection.
4. **Duration/alignment expansion** — monotonic-alignment expansion from predicted
   durations (cumsum -> alignment path -> matmul). Plain C++ over CPU buffers, not part of
   the ggml graph.
5. **DiT backbone** (`LayerFusionAudioDiT`) — 32 blocks (16 in / 1 mid / 16 out, U-Net
   skip connections), `embed_dim=1536`, 24 heads, GEGLU FFN (`mlp_ratio=4.0`), AdaLN time
   conditioning, shared RoPE, per-block cross-attention to content, per-block additive
   fusion of duration-aligned "time-aligned context". 1310 tensors total in the source
   safetensors (~1258 under `backbone.*`).
6. **Flow-matching Euler scheduler** — sway-sampling sigma schedule + Euler step,
   default `num_steps=25`, doubled batch for classifier-free guidance.
7. **Vocoder** (`mispeech/dashengtokenizer`, decode path only) — `ConvTranspose1d`
   upsampler (kernel=2, stride=2) -> `VocosModel` (12 ConvNeXt-style 1D conv blocks) ->
   `ISTFTHead` (Linear -> complex spectrogram -> overlap-add inverse STFT,
   `n_fft=1280`, `hop_length=320`). `n_fft=1280 = 2^8 x 5` is not a power of two, so the
   ISTFT uses a vendored mixed-radix FFT rather than a naive radix-2 implementation.

Reference HF source for both repos (AudioGen + tokenizer) lives under `reference/` for
ongoing comparison against the C++ implementation during development.

## Layout

```
third_party/      ggml submodule, header-only FFT lib
convert/          Python: HF safetensors -> our GGUF tensor schema
src/              C++: t5_encoder, dit, scheduler, vocoder, pipeline, main (CLI)
reference/        downloaded HF modeling source (ground truth for shapes/ops)
docs/             this file
```
