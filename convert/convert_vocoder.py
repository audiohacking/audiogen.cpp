#!/usr/bin/env python3
"""Convert mispeech/dashengtokenizer's decode path (upsampler + Vocos decoder)
to a standalone GGUF file for the dasheng-audiogen.cpp engine.

Only the decode path is needed (text-to-audio is decode-only): the
`encoder.*` tensors (audio -> tokens) are not part of this pipeline and are
skipped.
"""
import argparse
import json
import os

import gguf
import numpy as np
from safetensors import safe_open

ARCH = "dasheng_audiogen_vocoder"

TENSOR_PREFIXES = ("decoder.", "upsampler.")

META_INT_FIELDS = (
    "decoder_depth", "decoder_embed_dim", "decoder_intermediate_size",
    "embed_dim", "hop_length", "istft_hop", "istft_n_fft", "upsample_tokens",
)


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
    ap.add_argument("src", help="local checkpoint dir or HF repo id (mispeech/dashengtokenizer)")
    ap.add_argument("-o", "--out", default="vocoder.gguf")
    ap.add_argument("--dtype", default="f16", choices=["f32", "f16"])
    args = ap.parse_args()

    config_path, weights_path = resolve_paths(args.src)
    with open(config_path) as f:
        config = json.load(f)

    writer = gguf.GGUFWriter(args.out, ARCH)
    writer.add_name("dasheng-audiogen-vocoder")

    for key in META_INT_FIELDS:
        writer.add_int32(f"dasheng_audiogen.{key}", int(config[key]))

    np_dtype = np.float16 if args.dtype == "f16" else np.float32

    n_written = 0
    with safe_open(weights_path, framework="numpy") as f:
        for name in f.keys():
            if not name.startswith(TENSOR_PREFIXES):
                continue
            if name == "decoder.head.istft.window":
                # Hann window of size win_length == n_fft; recomputed in
                # src/vocoder.cpp rather than carried as a stored buffer.
                continue
            tensor = f.get_tensor(name)
            # Keep 1-D tensors (biases, norms, gamma) in f32 for precision;
            # only downcast 2-D+ weight matrices.
            if tensor.dtype == np.float32 and np_dtype != np.float32 and tensor.ndim >= 2:
                tensor = tensor.astype(np_dtype)
            writer.add_tensor(name, tensor)
            n_written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"wrote {n_written} tensors to {args.out}")


if __name__ == "__main__":
    main()
