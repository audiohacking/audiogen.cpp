#!/usr/bin/env python3
"""Convert google/flan-t5-large's encoder stack to a standalone GGUF file for
the dasheng-audiogen.cpp engine's vendored T5 encoder-only graph
(src/t5_encoder.cpp).

Only the encoder is needed (Dasheng-AudioGen never runs the T5 decoder), so
decoder.*, lm_head.* and the encoder-decoder shared bits beyond the token
embedding table are skipped.

Tensor naming is our own minimal scheme (not llama.cpp's GGUF vocab layout)
since src/t5_encoder.cpp is a small hand-written graph, not a llama.cpp
architecture-registry consumer:

  token_embd.weight              [vocab, d_model]
  enc.blk.{i}.attn_norm.weight   [d_model]
  enc.blk.{i}.attn_q.weight      [d_model, d_model]
  enc.blk.{i}.attn_k.weight      [d_model, d_model]
  enc.blk.{i}.attn_v.weight      [d_model, d_model]
  enc.blk.{i}.attn_o.weight      [d_model, d_model]
  enc.blk.0.attn_rel_b.weight    [num_buckets, n_head]   (block 0 only; the
                                  relative-position bias table is shared
                                  across all layers in T5)
  enc.blk.{i}.ffn_norm.weight    [d_model]
  enc.blk.{i}.ffn_gate.weight    [d_ff, d_model]   (wi_0, gated-gelu gate)
  enc.blk.{i}.ffn_up.weight      [d_ff, d_model]   (wi_1)
  enc.blk.{i}.ffn_down.weight    [d_model, d_ff]   (wo)
  enc.output_norm.weight         [d_model]
"""
import argparse
import json
import os

import gguf
import numpy as np
from safetensors import safe_open

ARCH = "dasheng_audiogen_t5enc"

META_INT_FIELDS = {
    "n_layer": "num_layers",
    "n_head": "num_heads",
    "d_model": "d_model",
    "d_kv": "d_kv",
    "d_ff": "d_ff",
    "vocab_size": "vocab_size",
    "relative_attention_num_buckets": "relative_attention_num_buckets",
    "relative_attention_max_distance": "relative_attention_max_distance",
}


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


def cast(tensor, np_dtype, force_f32=False):
    if force_f32:
        return tensor
    if tensor.dtype == np.float32 and np_dtype != np.float32 and tensor.ndim >= 2:
        return tensor.astype(np_dtype)
    return tensor


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="local checkpoint dir or HF repo id (google/flan-t5-large)")
    ap.add_argument("-o", "--out", default="t5-encoder.gguf")
    ap.add_argument("--dtype", default="f16", choices=["f32", "f16"])
    args = ap.parse_args()

    config_path, weights_path = resolve_paths(args.src)
    with open(config_path) as f:
        config = json.load(f)

    n_layer = config["num_layers"]

    writer = gguf.GGUFWriter(args.out, ARCH)
    writer.add_name("dasheng-audiogen-t5-encoder")
    for gguf_key, cfg_key in META_INT_FIELDS.items():
        writer.add_int32(f"t5enc.{gguf_key}", int(config[cfg_key]))
    writer.add_float32("t5enc.layer_norm_eps", float(config["layer_norm_epsilon"]))

    np_dtype = np.float16 if args.dtype == "f16" else np.float32

    n_written = 0
    with safe_open(weights_path, framework="numpy") as f:
        writer.add_tensor("token_embd.weight", cast(f.get_tensor("shared.weight"), np_dtype))
        n_written += 1

        for i in range(n_layer):
            prefix = f"encoder.block.{i}"
            mapping = [
                (f"{prefix}.layer.0.layer_norm.weight", f"enc.blk.{i}.attn_norm.weight"),
                (f"{prefix}.layer.0.SelfAttention.q.weight", f"enc.blk.{i}.attn_q.weight"),
                (f"{prefix}.layer.0.SelfAttention.k.weight", f"enc.blk.{i}.attn_k.weight"),
                (f"{prefix}.layer.0.SelfAttention.v.weight", f"enc.blk.{i}.attn_v.weight"),
                (f"{prefix}.layer.0.SelfAttention.o.weight", f"enc.blk.{i}.attn_o.weight"),
                (f"{prefix}.layer.1.layer_norm.weight", f"enc.blk.{i}.ffn_norm.weight"),
                (f"{prefix}.layer.1.DenseReluDense.wi_0.weight", f"enc.blk.{i}.ffn_gate.weight"),
                (f"{prefix}.layer.1.DenseReluDense.wi_1.weight", f"enc.blk.{i}.ffn_up.weight"),
                (f"{prefix}.layer.1.DenseReluDense.wo.weight", f"enc.blk.{i}.ffn_down.weight"),
            ]
            if i == 0:
                mapping.append((
                    f"{prefix}.layer.0.SelfAttention.relative_attention_bias.weight",
                    "enc.blk.0.attn_rel_b.weight",
                ))
            for src_name, dst_name in mapping:
                force_f32 = dst_name.endswith("attn_rel_b.weight")
                writer.add_tensor(dst_name, cast(f.get_tensor(src_name), np_dtype, force_f32))
                n_written += 1

        writer.add_tensor(
            "enc.output_norm.weight", cast(f.get_tensor("encoder.final_layer_norm.weight"), np_dtype)
        )
        n_written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"wrote {n_written} tensors to {args.out}")


if __name__ == "__main__":
    main()
