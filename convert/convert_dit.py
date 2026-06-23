#!/usr/bin/env python3
"""Convert mispeech/Dasheng-AudioGen's DiT backbone + content adapter weights
to a standalone GGUF file for the dasheng-audiogen.cpp engine.

Excludes content_encoder.text_encoder.* T5 weights (those go through
convert_t5_encoder.py, which targets llama.cpp's own T5 GGUF schema) and the
Vocos vocoder (convert_vocoder.py).
"""
import argparse
import json
import os

import gguf
import numpy as np
from safetensors import safe_open

ARCH = "dasheng_audiogen_dit"

# Tensor name prefixes carried over 1:1 from the HF checkpoint into GGUF.
TENSOR_PREFIXES = (
    "backbone.",
    "content_adapter.",
    "content_encoder.text_encoder.proj.",
)
# Top-level buffers consumed directly by content_adapter / generation that
# don't fall under any prefix above.
EXTRA_TENSORS = ("dummy_nta_embed", "dummy_ta_embed", "instruction_embedding")

META_INT_FIELDS = (
    "instruction_seq_len", "task_instruction_dim", "latent_dim", "content_dim",
    "dit_img_size", "dit_patch_size", "dit_in_chans", "dit_out_chans",
    "dit_embed_dim", "dit_depth", "dit_num_heads", "dit_ada_sola_rank",
    "dit_ada_sola_alpha", "dit_ta_context_dim", "dit_context_dim",
    "adapter_num_heads", "duration_predictor_filter_channels",
    "duration_predictor_n_layers", "duration_predictor_kernel_size",
    "downsampling_ratio", "sample_rate", "tokenizer_max_length",
)
META_FLOAT_FIELDS = ("dit_mlp_ratio", "frame_resolution", "duration_offset")
META_STR_FIELDS = (
    "dit_qk_norm", "dit_norm_layer", "dit_act_layer", "dit_time_fusion",
    "dit_ta_context_fusion", "dit_context_fusion", "dit_context_pe_method",
    "dit_pe_method", "dit_rope_mode", "dit_input_type",
)
META_BOOL_FIELDS = ("dit_context_norm", "dit_ta_context_norm", "use_zero_instruction")


def resolve_paths(src):
    if os.path.isdir(src):
        return (
            os.path.join(src, "config.json"),
            os.path.join(src, "model.safetensors"),
        )
    from huggingface_hub import hf_hub_download
    config_path = hf_hub_download(src, "config.json")
    weights_path = hf_hub_download(src, "model.safetensors")
    return config_path, weights_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="local checkpoint dir or HF repo id (mispeech/Dasheng-AudioGen)")
    ap.add_argument("-o", "--out", default="dit.gguf")
    ap.add_argument("--dtype", default="f16", choices=["f32", "f16", "q8_0", "q4_0"])
    args = ap.parse_args()

    config_path, weights_path = resolve_paths(args.src)
    with open(config_path) as f:
        config = json.load(f)

    writer = gguf.GGUFWriter(args.out, ARCH)
    writer.add_name("dasheng-audiogen-dit")

    for key in META_INT_FIELDS:
        writer.add_int32(f"dasheng_audiogen.{key}", int(config[key]))
    for key in META_FLOAT_FIELDS:
        writer.add_float32(f"dasheng_audiogen.{key}", float(config[key]))
    for key in META_STR_FIELDS:
        writer.add_string(f"dasheng_audiogen.{key}", str(config[key]))
    for key in META_BOOL_FIELDS:
        writer.add_bool(f"dasheng_audiogen.{key}", bool(config[key]))

    # Determine dtype and quantization settings
    use_quant = args.dtype in ("q8_0", "q4_0")
    if use_quant:
        quant_type = gguf.GGMLQuantizationType.Q8_0 if args.dtype == "q8_0" else gguf.GGMLQuantizationType.Q4_0
        np_dtype = np.float32  # Quantization expects f32 input
    else:
        np_dtype = np.float16 if args.dtype == "f16" else np.float32
        quant_type = None

    n_written = 0
    n_quantized = 0
    with safe_open(weights_path, framework="numpy") as f:
        for name in f.keys():
            if name == "instruction_lengths":
                writer.add_int32(
                    "dasheng_audiogen.instruction_lengths",
                    int(f.get_tensor(name)[0]),
                )
                continue
            if name == "dummy_param":
                continue  # zero-size training-only placeholder
            if not (name.startswith(TENSOR_PREFIXES) or name in EXTRA_TENSORS):
                continue
            tensor = f.get_tensor(name)

            # Determine if this tensor should be quantized
            # Only quantize 2D+ weight matrices, not biases/norms/small tensors
            # Also check that dimensions are compatible with quantization block size (32)
            # IMPORTANT: Only quantize backbone.* tensors (GPU), not content_adapter.* (CPU fallback)
            block_size = 32
            row_compatible = tensor.shape[-1] % block_size == 0 if tensor.ndim >= 1 else False

            should_quantize = (
                use_quant and
                name.startswith('backbone.') and  # Only backbone tensors (run on GPU)
                tensor.ndim >= 2 and
                tensor.size >= 256 and
                row_compatible and
                not any(s in name.lower() for s in ['bias', 'norm', 'ln_'])
            )

            if should_quantize:
                # Convert to f32 and quantize
                tensor = tensor.astype(np.float32)
                tensor = gguf.quants.quantize(tensor, quant_type)
                writer.add_tensor(name, tensor, raw_dtype=quant_type)
                n_quantized += 1
            else:
                # Keep 1-D tensors (biases, norms) in f32 for precision
                if tensor.dtype == np.float32 and np_dtype != np.float32 and tensor.ndim >= 2:
                    tensor = tensor.astype(np_dtype)
                writer.add_tensor(name, tensor)
            n_written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    if use_quant:
        print(f"wrote {n_written} tensors ({n_quantized} quantized to {args.dtype.upper()}) to {args.out}")
    else:
        print(f"wrote {n_written} tensors to {args.out}")


if __name__ == "__main__":
    main()
