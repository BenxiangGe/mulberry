"""Export a small Nielsen MNIST training subset as one safetensors file.

The current Mulberry runtime reads named tensors by expected type. Training and
test subsets are exported as batch tensors and Mulberry uses tensor.sliceFirst()
to view individual samples until a real dataset iterator exists.
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
        "--test-count",
        type=int,
        default=10,
        help="Number of test samples to export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the output safetensors file.",
    )
    return parser.parse_args()


def loadMnistData(
    path: Path,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    numpyExceptions = getattr(np, "exceptions", np)
    warningClass = getattr(numpyExceptions, "VisibleDeprecationWarning", Warning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", warningClass)
        with gzip.open(path, "rb") as file:
            trainingData, _validationData, testData = pickle.load(
                file, encoding="latin1"
            )

    return trainingData, testData


def oneHotLabels(labels: np.ndarray, count: int) -> np.ndarray:
    values = np.zeros((count, 10, 1), dtype=np.float32)
    for index in range(count):
        values[index, int(labels[index]), 0] = 1.0
    return values


def main() -> None:
    args = parseArgs()
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    if args.test_count <= 0:
        raise ValueError("--test-count must be positive")

    w1, b1, w2, b2 = loadNetwork(args.network_json)
    (trainingInputs, trainingLabels), (testInputs, testLabels) = loadMnistData(
        args.mnist_data
    )
    if args.sample_count > len(trainingLabels):
        raise ValueError(
            f"--sample-count {args.sample_count} exceeds "
            f"{len(trainingLabels)} training samples"
        )
    if args.test_count > len(testLabels):
        raise ValueError(
            f"--test-count {args.test_count} exceeds {len(testLabels)} test samples"
        )

    tensors = [
        ("w1", w1),
        ("b1", b1),
        ("w2", w2),
        ("b2", b2),
    ]
    trainX = np.asarray(trainingInputs[: args.sample_count], dtype=np.float32)
    trainX = trainX.reshape((args.sample_count, 784, 1))
    trainY = oneHotLabels(trainingLabels, args.sample_count)
    testX = np.asarray(testInputs[: args.test_count], dtype=np.float32)
    testX = testX.reshape((args.test_count, 784, 1))
    testY = oneHotLabels(testLabels, args.test_count)
    tensors.extend([
        ("train_x", trainX),
        ("train_y", trainY),
        ("test_x", testX),
        ("test_y", testY),
    ])

    byteCount = writeSafetensors(args.output, tensors)

    print(f"{args.output}: safetensors, {byteCount} bytes")
    for name, value in tensors:
        print(f"{name}: F32{list(value.shape)}")


if __name__ == "__main__":
    main()
