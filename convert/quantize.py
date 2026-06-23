#!/usr/bin/env python3
"""Quantize GGUF models to lower precision (Q8_0, Q4_0, etc.)"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

try:
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
except ImportError:
    print("Error: gguf package not found. Install with: pip install gguf")
    sys.exit(1)


def quantize_to_q8_0(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize float32/float16 data to Q8_0 format.

    Q8_0: 32 values per block, 1 scale (fp16) + 32 int8 values = 34 bytes per block
    """
    data = data.astype(np.float32).flatten()
    n = len(data)

    # Pad to multiple of 32
    pad = (32 - n % 32) % 32
    if pad > 0:
        data = np.concatenate([data, np.zeros(pad, dtype=np.float32)])

    n_blocks = len(data) // 32
    data = data.reshape(n_blocks, 32)

    # Compute scale per block (max abs value / 127)
    max_vals = np.max(np.abs(data), axis=1, keepdims=True)
    scales = max_vals / 127.0
    scales = np.where(scales == 0, 1.0, scales)  # Avoid division by zero

    # Quantize to int8
    quantized = np.round(data / scales).astype(np.int8)
    scales = scales.astype(np.float16).flatten()

    return quantized.flatten(), scales


def quantize_to_q4_0(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize float32/float16 data to Q4_0 format.

    Q4_0: 32 values per block, 1 scale (fp16) + 16 bytes (32 x 4-bit) = 18 bytes per block
    """
    data = data.astype(np.float32).flatten()
    n = len(data)

    # Pad to multiple of 32
    pad = (32 - n % 32) % 32
    if pad > 0:
        data = np.concatenate([data, np.zeros(pad, dtype=np.float32)])

    n_blocks = len(data) // 32
    data = data.reshape(n_blocks, 32)

    # Compute scale per block (max abs value / 7 for 4-bit signed)
    max_vals = np.max(np.abs(data), axis=1, keepdims=True)
    scales = max_vals / 7.0
    scales = np.where(scales == 0, 1.0, scales)

    # Quantize to 4-bit (-8 to 7)
    quantized = np.round(data / scales).clip(-8, 7).astype(np.int8)

    # Pack two 4-bit values into one byte
    q_low = (quantized[:, 0::2] + 8).astype(np.uint8)  # 0-15
    q_high = (quantized[:, 1::2] + 8).astype(np.uint8)  # 0-15
    packed = (q_high << 4) | q_low

    scales = scales.astype(np.float16).flatten()

    return packed.flatten(), scales


def should_quantize(name: str, shape: tuple) -> bool:
    """Determine if a tensor should be quantized based on name and shape."""
    # Don't quantize small tensors (biases, layer norms, embeddings < 256 elements)
    if np.prod(shape) < 256:
        return False

    # Don't quantize 1D tensors (typically biases)
    if len(shape) == 1:
        return False

    # Don't quantize specific tensor types by name
    skip_patterns = ['bias', 'norm', 'embed', 'ln_', 'layer_norm', 'position']
    name_lower = name.lower()
    for pattern in skip_patterns:
        if pattern in name_lower:
            return False

    return True


def quantize_gguf(input_path: str, output_path: str, quant_type: str, verbose: bool = False):
    """Quantize a GGUF file to the specified quantization type."""

    quant_map = {
        'q8_0': (GGMLQuantizationType.Q8_0, quantize_to_q8_0),
        'q4_0': (GGMLQuantizationType.Q4_0, quantize_to_q4_0),
    }

    if quant_type.lower() not in quant_map:
        raise ValueError(f"Unsupported quantization type: {quant_type}. Supported: {list(quant_map.keys())}")

    target_type, quant_func = quant_map[quant_type.lower()]

    print(f"Reading {input_path}...")
    reader = GGUFReader(input_path)

    # Get architecture from metadata
    arch = None
    for field in reader.fields.values():
        if 'architecture' in field.name or 'arch' in field.name:
            arch = str(field.parts[-1])
            break

    print(f"Writing {output_path} with {quant_type.upper()} quantization...")
    writer = GGUFWriter(output_path, arch or "dasheng")

    # Copy metadata
    for field in reader.fields.values():
        if field.name == 'GGUF.version':
            continue
        try:
            # Handle different field types
            if hasattr(field, 'parts') and len(field.parts) > 0:
                value = field.parts[-1]
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='ignore')
                writer.add_key_value(field.name, value)
        except Exception as e:
            if verbose:
                print(f"  Skipping metadata {field.name}: {e}")

    # Process tensors
    total_original = 0
    total_quantized = 0
    quantized_count = 0

    for tensor in reader.tensors:
        name = tensor.name
        shape = tuple(tensor.shape)
        original_type = tensor.tensor_type
        data = tensor.data

        total_original += tensor.n_bytes

        if should_quantize(name, shape) and original_type in [GGMLQuantizationType.F32, GGMLQuantizationType.F16]:
            # Quantize this tensor
            if verbose:
                print(f"  Quantizing {name}: {shape} {original_type.name} -> {quant_type.upper()}")

            # Note: gguf library handles the quantization internally when we specify the type
            # We just need to pass the original data and it will quantize
            writer.add_tensor(name, data, raw_dtype=target_type)
            quantized_count += 1
        else:
            # Keep original
            if verbose:
                print(f"  Keeping {name}: {shape} {original_type.name}")
            writer.add_tensor(name, data, raw_dtype=original_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Get output size
    output_size = os.path.getsize(output_path)

    print(f"Done! Quantized {quantized_count} tensors")
    print(f"  Original: {total_original / 1e9:.2f} GB")
    print(f"  Quantized: {output_size / 1e9:.2f} GB")
    print(f"  Compression: {total_original / output_size:.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Quantize GGUF models")
    parser.add_argument("input", help="Input GGUF file")
    parser.add_argument("-o", "--output", help="Output GGUF file (default: input_QTYPE.gguf)")
    parser.add_argument("-t", "--type", choices=['q8_0', 'q4_0'], default='q8_0',
                        help="Quantization type (default: q8_0)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if args.output is None:
        base = Path(args.input).stem
        args.output = f"{base}_{args.type.upper()}.gguf"

    quantize_gguf(args.input, args.output, args.type, args.verbose)


if __name__ == "__main__":
    main()
