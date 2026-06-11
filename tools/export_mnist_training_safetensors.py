"""Export a small Nielsen MNIST training subset as one safetensors file.

The current Mulberry runtime reads named tensors by expected type. Until the
language has tensor slicing or a dataset iterator, each sample is exported as
its own pair of tensors: train_x_0, train_y_0, train_x_1, train_y_1, ...
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import warnings
from pathlib import Path

import numpy as np

from export_mnist_raw_tensors import (
    DEFAULT_MNIST_DATA,
    DEFAULT_NETWORK_JSON,
    loadNetwork,
)
from export_mnist_safetensors import writeSafetensors


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data/mnist-784-30-10-training.safetensors"


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Nielsen MNIST training tensors as one safetensors file."
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
        "--sample-count",
        type=int,
        default=10,
        help="Number of training samples to export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the output safetensors file.",
    )
    return parser.parse_args()


def loadTrainingData(path: Path) -> tuple[np.ndarray, np.ndarray]:
    numpyExceptions = getattr(np, "exceptions", np)
    warningClass = getattr(numpyExceptions, "VisibleDeprecationWarning", Warning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", warningClass)
        with gzip.open(path, "rb") as file:
            trainingData, _validationData, _testData = pickle.load(
                file, encoding="latin1"
            )

    return trainingData


def oneHotLabel(label: int) -> np.ndarray:
    value = np.zeros((10, 1), dtype=np.float32)
    value[label, 0] = 1.0
    return value


def main() -> None:
    args = parseArgs()
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")

    w1, b1, w2, b2 = loadNetwork(args.network_json)
    trainingInputs, trainingLabels = loadTrainingData(args.mnist_data)
    if args.sample_count > len(trainingLabels):
        raise ValueError(
            f"--sample-count {args.sample_count} exceeds "
            f"{len(trainingLabels)} training samples"
        )

    tensors = [
        ("w1", w1),
        ("b1", b1),
        ("w2", w2),
        ("b2", b2),
    ]
    for index in range(args.sample_count):
        x = np.asarray(trainingInputs[index], dtype=np.float32).reshape((784, 1))
        y = oneHotLabel(int(trainingLabels[index]))
        tensors.append((f"train_x_{index}", x))
        tensors.append((f"train_y_{index}", y))

    byteCount = writeSafetensors(args.output, tensors)

    print(f"{args.output}: safetensors, {byteCount} bytes")
    for name, value in tensors:
        print(f"{name}: F32{list(value.shape)}")


if __name__ == "__main__":
    main()
