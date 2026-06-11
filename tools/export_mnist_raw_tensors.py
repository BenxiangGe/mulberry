"""Export Nielsen MNIST tensors as raw little-endian Float32 files.

The generated files intentionally contain only element bytes. Mulberry source
types provide dtype and shape when reading these files.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import warnings
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NETWORK_JSON = REPO_ROOT / "data/mnist-784-30-10.json"
DEFAULT_MNIST_DATA = REPO_ROOT / "data/mnist.pkl.gz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/mnist-784-30-10-raw"


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Nielsen MNIST tensors as raw Float32 files."
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
        help="MNIST test_data sample index to export as x.f32.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for raw tensor output files.",
    )
    return parser.parse_args()


def validateMatrix(name: str, value: np.ndarray, shape: tuple[int, int]) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} has shape {value.shape}, expected {shape}")


def loadNetwork(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as file:
        network = json.load(file)

    sizes = network["sizes"]
    if sizes != [784, 30, 10]:
        raise ValueError(f"expected network sizes [784, 30, 10], got {sizes}")

    w1 = np.asarray(network["weights"][0], dtype=np.float32)
    w2 = np.asarray(network["weights"][1], dtype=np.float32)
    b1 = np.asarray(network["biases"][0], dtype=np.float32)
    b2 = np.asarray(network["biases"][1], dtype=np.float32)

    validateMatrix("w1", w1, (30, 784))
    validateMatrix("w2", w2, (10, 30))
    validateMatrix("b1", b1, (30, 1))
    validateMatrix("b2", b2, (10, 1))
    return w1, b1, w2, b2


def loadTestSample(path: Path, sampleIndex: int) -> tuple[np.ndarray, int]:
    numpyExceptions = getattr(np, "exceptions", np)
    warningClass = getattr(numpyExceptions, "VisibleDeprecationWarning", Warning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", warningClass)
        with gzip.open(path, "rb") as file:
            _trainingData, _validationData, testData = pickle.load(
                file, encoding="latin1"
            )

    testInputs, testLabels = testData
    x = np.asarray(testInputs[sampleIndex], dtype=np.float32).reshape((784, 1))
    y = int(testLabels[sampleIndex])
    validateMatrix("x", x, (784, 1))
    return x, y


def writeFloat32Tensor(path: Path, value: np.ndarray) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = np.ascontiguousarray(value, dtype=np.dtype("<f4"))
    path.write_bytes(tensor.tobytes(order="C"))
    return tensor.nbytes


def report(path: Path, typeName: str, byteCount: int) -> None:
    print(f"{path}: {typeName}, {byteCount} bytes")


def main() -> None:
    args = parseArgs()
    outputDir = args.output_dir

    w1, b1, w2, b2 = loadNetwork(args.network_json)
    x, label = loadTestSample(args.mnist_data, args.sample_index)

    outputs = [
        ("w1.f32", "Float32[30, 784]", w1),
        ("b1.f32", "Float32[30, 1]", b1),
        ("w2.f32", "Float32[10, 30]", w2),
        ("b2.f32", "Float32[10, 1]", b2),
        ("x.f32", "Float32[784, 1]", x),
    ]

    for fileName, typeName, value in outputs:
        path = outputDir / fileName
        byteCount = writeFloat32Tensor(path, value)
        report(path, typeName, byteCount)

    labelPath = outputDir / "label.txt"
    labelPath.write_text(f"{label}\n", encoding="utf-8")
    print(f"{labelPath}: UInt64 label = {label}")


if __name__ == "__main__":
    main()
