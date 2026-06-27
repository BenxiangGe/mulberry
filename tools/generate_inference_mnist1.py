"""Generate a Mulberry source file with Nielsen MNIST inference data.

Run from the Mulberry repository root:

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
        help="Generated Mulberry source path.",
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


def emitVariable(
    name: str,
    matrix: np.ndarray,
    isConst: bool = False,
    typeAnnotation: str | None = None,
) -> str:
    rows, cols = matrix.shape
    literal = formatMatrixLiteral(matrix)
    keyword = "const" if isConst else "var"
    if typeAnnotation is None:
        typeAnnotation = f"Float32[{rows}, {cols}]"
    return f"  {keyword} {name}: {typeAnnotation} = {literal};"


def formatPathForComment(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def writeMulberrySource(
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
            emitVariable("w1", w1),
            emitVariable("b1", b1),
            emitVariable("w2", w2),
            emitVariable("b2", b2),
            "\n".join(
                [
                    "  var w1h: Tensor<Float32> = tensor.pack(w1);",
                    "  var b1h: Tensor<Float32> = tensor.pack(b1);",
                    "  var w2h: Tensor<Float32> = tensor.pack(w2);",
                    "  var b2h: Tensor<Float32> = tensor.pack(b2);",
                    "",
                    "  var weights: List<Tensor<Float32>> = [w1h, w2h];",
                    "  var biases: List<Tensor<Float32>> = [b1h, b2h];",
                ]
            ),
            emitVariable("activationValue", x, typeAnnotation="Float32[?, ?]"),
            "  var activation: Tensor<Float32> = tensor.pack(activationValue);",
            f"  const y: UInt64 = {y};",
            "\n".join(
                [
                    "  for layer in 0 .. weights.size() {",
                    "    activation = nn.sigmoid(nn.matadd(",
                    "        nn.matmul(weights[layer], activation),",
                    "        biases[layer]));",
                    "    ()",
                    "  };",
                ]
            ),
            "  var pred: UInt64 = nn.argmax(activation);",
            "  io.print(pred);",
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
                "# Requires the external mulberry.nn package.",
                "# Generated as regression material while NN is detached from core.",
                "",
                "import mulberry.nn;",
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

    writeMulberrySource(
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
    print("activation: Float32[?, ?]")
    print(f"y: UInt64 = {y}")


if __name__ == "__main__":
    main()
