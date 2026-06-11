"""Export Nielsen MNIST tensors as a single safetensors file.

This writes the safetensors layout directly instead of depending on the Python
package, so the generated file documents the exact minimal format Mulberry will
read later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from export_mnist_raw_tensors import (
    DEFAULT_MNIST_DATA,
    DEFAULT_NETWORK_JSON,
    loadNetwork,
    loadTestSample,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data/mnist-784-30-10.safetensors"


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Nielsen MNIST tensors as one safetensors file."
    )
    parser.add_argument(
        "--network-json",
        type=Path,
        default=DEFAULT_NETWORK_JSON,
        help="Path to mnist-784-30-10.json.",
    )
    parser.add_argument(
        "--mnist-data",
        type=Path,
        default=DEFAULT_MNIST_DATA,
        help="Path to mnist.pkl.gz.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="MNIST test_data sample index to export as x.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the output safetensors file.",
    )
    return parser.parse_args()


def float32Payload(value: np.ndarray) -> bytes:
    tensor = np.ascontiguousarray(value, dtype=np.dtype("<f4"))
    return tensor.tobytes(order="C")


def writeSafetensors(path: Path, tensors: list[tuple[str, np.ndarray]]) -> int:
    header: dict[str, dict[str, object]] = {}
    payloads: list[bytes] = []
    offset = 0

    for name, value in tensors:
        payload = float32Payload(value)
        nextOffset = offset + len(payload)
        header[name] = {
            "dtype": "F32",
            "shape": list(value.shape),
            "data_offsets": [offset, nextOffset],
        }
        payloads.append(payload)
        offset = nextOffset

    headerBytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(len(headerBytes).to_bytes(8, "little"))
        file.write(headerBytes)
        for payload in payloads:
            file.write(payload)

    return path.stat().st_size


def main() -> None:
    args = parseArgs()

    w1, b1, w2, b2 = loadNetwork(args.network_json)
    x, label = loadTestSample(args.mnist_data, args.sample_index)

    tensors = [
        ("w1", w1),
        ("b1", b1),
        ("w2", w2),
        ("b2", b2),
        ("x", x),
    ]
    byteCount = writeSafetensors(args.output, tensors)

    print(f"{args.output}: safetensors, {byteCount} bytes")
    for name, value in tensors:
        print(f"{name}: F32{list(value.shape)}")
    print(f"label: {label}")


if __name__ == "__main__":
    main()
