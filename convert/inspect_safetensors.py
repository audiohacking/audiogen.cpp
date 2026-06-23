#!/usr/bin/env python3
"""Dump tensor names/shapes/dtypes from a safetensors file's header without
downloading the full weights. Used to nail down the exact key mapping needed
by convert_dit.py / convert_vocoder.py before writing the converters.
"""
import argparse
import json
import struct
import sys
from collections import Counter


def read_header(path_or_url: str) -> dict:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        import requests

        r = requests.get(path_or_url, headers={"Range": "bytes=0-7"})
        n = struct.unpack("<Q", r.content)[0]
        r2 = requests.get(path_or_url, headers={"Range": f"bytes=8-{8 + n - 1}"})
        return json.loads(r2.content)
    with open(path_or_url, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="local path or URL to a .safetensors file")
    ap.add_argument("--prefix-depth", type=int, default=2, help="group tensors by first N dot-separated path components")
    ap.add_argument("--grep", default=None, help="only show tensor names containing this substring")
    args = ap.parse_args()

    header = read_header(args.src)
    header.pop("__metadata__", None)

    if args.grep:
        for k, v in sorted(header.items()):
            if args.grep in k:
                print(k, v["dtype"], v["shape"])
        return

    counts = Counter()
    for k in header:
        parts = k.split(".")
        counts[".".join(parts[: args.prefix_depth])] += 1

    print(f"total tensors: {len(header)}")
    for prefix, c in sorted(counts.items()):
        print(f"{prefix:60s} {c}")


if __name__ == "__main__":
    sys.exit(main())
