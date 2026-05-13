"""Generate a Cherry source file with Nielsen MNIST inference data.

Run from the Cherry repository root:

    python3 tools/generate_inference_mnist1.py

The default command reads `data/mnist-784-30-10.json` and
`data/mnist.pkl.gz`, then writes `examples/dl/inference_mnist1.cherry`.
If `mnist.pkl.gz` lives elsewhere, pass a relative path with `--mnist-data`.
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
DEFAULT_OUTPUT = REPO_ROOT / "examples/dl/inference_mnist1.cherry"
DEFAULT_MNIST_DATA = REPO_ROOT / "data/mnist.pkl.gz"


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate inference_mnist1.cherry from Nielsen MNIST data."
    )
    parser.add_argument(
        "--network-json",
        type=Path,
        default=DEFAULT_NETWORK_JSON,
        help="Path to mnist-784-30-10.json. Defaults to data/mnist-784-30-10.json.",
    )
    parser.add_argument(
        "--mnist-data",
        type=Path,
        default=DEFAULT_MNIST_DATA,
        help="Path to mnist.pkl.gz. Defaults to data/mnist.pkl.gz.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="MNIST test_data sample index to emit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Generated Cherry source path.",
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
    warningClass = getattr(np.exceptions, "VisibleDeprecationWarning", Warning)
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


def formatPositiveFloat32(value: np.float32) -> str:
    value = np.float32(value)
    if not np.isfinite(value):
        raise ValueError(f"cannot emit non-finite Float32 value: {value}")
    if value == np.float32(0.0):
        return "0.0"

    text = format(float(value), ".9g")
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text


def formatFloat32(value: np.float32) -> str:
    value = np.float32(value)
    if value < np.float32(0.0):
        return f"-{formatPositiveFloat32(np.float32(-value))}"
    return formatPositiveFloat32(value)


def formatMatrixLiteral(matrix: np.ndarray, indent: str = "    ") -> str:
    rows = []
    for row in matrix:
        elements = ", ".join(formatFloat32(value) for value in row)
        rows.append(f"{indent}[{elements}]")
    return "[\n" + ",\n".join(rows) + "\n  ]"


def emitVariable(name: str, matrix: np.ndarray, isConst: bool = False) -> str:
    rows, cols = matrix.shape
    literal = formatMatrixLiteral(matrix)
    keyword = "const" if isConst else "var"
    return f"  {keyword} {name}: Float32[{rows}, {cols}] = {literal};"


def formatPathForComment(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def writeCherrySource(
    output: Path,
    networkJson: Path,
    mnistData: Path,
    sampleIndex: int,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    x: np.ndarray,
    y: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    body = "\n\n".join(
        [
            emitVariable("w1", w1, isConst=True),
            emitVariable("b1", b1, isConst=True),
            emitVariable("w2", w2, isConst=True),
            emitVariable("b2", b2, isConst=True),
            emitVariable("x", x, isConst=True),
            f"  const y: UInt64 = {y};",
            "  var z1: Float32[30, 1] = matadd(matmul(w1, x), b1);",
            "  var a1: Float32[30, 1] = sigmoid(z1);",
            "  var z2: Float32[10, 1] = matadd(matmul(w2, a1), b2);",
            "  var a2: Float32[10, 1] = sigmoid(z2);",
            "  var pred: UInt64 = argmax(a2);",
            "  print(pred);",
            "  0",
        ]
    )

    output.write_text(
        "\n".join(
            [
                "# Generated by tools/generate_inference_mnist1.py.",
                f"# network-json: {formatPathForComment(networkJson)}",
                f"# mnist-data: {formatPathForComment(mnistData)}",
                f"# sample-index: {sampleIndex}",
                "",
                "fn main(): UInt64 {",
                body,
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parseArgs()
    networkJson = args.network_json
    mnistData = args.mnist_data

    w1, b1, w2, b2 = loadNetwork(networkJson)
    x, y = loadTestSample(mnistData, args.sample_index)

    writeCherrySource(
        args.output,
        networkJson,
        mnistData,
        args.sample_index,
        w1,
        b1,
        w2,
        b2,
        x,
        y,
    )
    print(f"wrote {args.output}")
    print("w1: Float32[30, 784]")
    print("b1: Float32[30, 1]")
    print("w2: Float32[10, 30]")
    print("b2: Float32[10, 1]")
    print("x: Float32[784, 1]")
    print(f"y: UInt64 = {y}")


if __name__ == "__main__":
    main()
