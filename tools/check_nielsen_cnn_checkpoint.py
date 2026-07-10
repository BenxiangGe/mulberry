"""Validate a Mulberry Nielsen CNN mini-batch checkpoint with NumPy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from check_cnn_backward import conv2d_backward, max_pool2d_backward
from export_nielsen_cnn_safetensors import conv2dNchw, maxPool2dNchw, softmax


PARAMETER_NAMES = (
    "conv1_weight",
    "conv1_bias",
    "classifier_weight",
    "classifier_bias",
)


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Mulberry Nielsen CNN SGD checkpoint."
    )
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--regularization", type=float, default=0.0)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--report-evaluation", action="store_true")
    return parser.parse_args()


def readSafetensors(path: Path) -> dict[str, np.ndarray]:
    contents = path.read_bytes()
    if len(contents) < 8:
        raise ValueError(f"{path} is too short to be a safetensors file")

    headerSize = int.from_bytes(contents[:8], byteorder="little")
    dataOffset = 8 + headerSize
    header = json.loads(contents[8:dataOffset].decode("utf-8"))
    tensors = {}
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        if descriptor["dtype"] != "F32":
            raise ValueError(f"{name} has unsupported dtype {descriptor['dtype']}")

        begin, end = descriptor["data_offsets"]
        value = np.frombuffer(
            contents,
            dtype=np.dtype("<f4"),
            count=(end - begin) // np.dtype("<f4").itemsize,
            offset=dataOffset + begin,
        )
        tensors[name] = value.reshape(descriptor["shape"]).copy()
    return tensors


def reluBackward(
    inputValue: np.ndarray, outputGradient: np.ndarray
) -> np.ndarray:
    return np.where(inputValue > np.float32(0.0), outputGradient, 0.0).astype(
        np.float32
    )


def cnnActivations(
    parameters: dict[str, np.ndarray],
    inputValue: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    convWeight = parameters["conv1_weight"]
    convBias = parameters["conv1_bias"]
    classifierWeight = parameters["classifier_weight"]

    convolved = conv2dNchw(inputValue, convWeight, convBias)
    pooled = maxPool2dNchw(convolved)
    activated = np.maximum(pooled, np.float32(0.0))
    features = activated.reshape((activated.shape[0], -1))
    logits = (
        features @ classifierWeight + parameters["classifier_bias"]
    ).astype(np.float32)
    return logits, convolved, pooled, activated


def cnnLogits(
    parameters: dict[str, np.ndarray], inputValue: np.ndarray
) -> np.ndarray:
    logits, _convolved, _pooled, _activated = cnnActivations(
        parameters, inputValue
    )
    return logits


def cnnProbabilities(
    parameters: dict[str, np.ndarray], inputValue: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logits, convolved, pooled, activated = cnnActivations(
        parameters, inputValue
    )
    return softmax(logits).astype(np.float32), convolved, pooled, activated


def softmaxCrossEntropy(
    logits: np.ndarray, expected: np.ndarray
) -> np.float32:
    expectedRows = expected.reshape(logits.shape)
    rowMax = np.max(logits, axis=1, keepdims=True)
    exponentialSum = np.sum(
        np.exp(logits - rowMax), axis=1, dtype=np.float32
    )
    logSumExp = (
        rowMax[:, 0] + np.log(exponentialSum).astype(np.float32)
    ).astype(np.float32)
    weightedLogits = np.sum(
        expectedRows * logits, axis=1, dtype=np.float32
    )
    return np.float32(
        np.mean(logSumExp - weightedLogits, dtype=np.float32)
    )


def meanCrossEntropy(
    parameters: dict[str, np.ndarray],
    inputs: np.ndarray,
    labels: np.ndarray,
) -> np.float32:
    return softmaxCrossEntropy(cnnLogits(parameters, inputs), labels)


def cnnGradient(
    parameters: dict[str, np.ndarray],
    inputValue: np.ndarray,
    expected: np.ndarray,
) -> dict[str, np.ndarray]:
    convWeight = parameters["conv1_weight"]
    classifierWeight = parameters["classifier_weight"]
    probabilities, convolved, pooled, activated = cnnProbabilities(
        parameters, inputValue
    )
    features = activated.reshape((activated.shape[0], -1))

    scoreGradient = (probabilities - expected).astype(np.float32)
    classifierWeightGradient = (features.T @ scoreGradient).astype(np.float32)
    featureGradient = (scoreGradient @ classifierWeight.T).astype(np.float32)
    activationGradient = featureGradient.reshape(activated.shape)
    poolGradient = reluBackward(pooled, activationGradient)
    convOutputGradient = max_pool2d_backward(
        convolved,
        poolGradient,
        (2, 2),
        (0, 0, 0, 0),
        (2, 2),
    ).astype(np.float32)
    _, convWeightGradient, convBiasGradient = conv2d_backward(
        inputValue,
        convWeight,
        convOutputGradient,
        (0, 0, 0, 0),
        (1, 1),
        (1, 1),
    )

    return {
        "conv1_weight": convWeightGradient.astype(np.float32),
        "conv1_bias": convBiasGradient.astype(np.float32),
        "classifier_weight": classifierWeightGradient,
        "classifier_bias": scoreGradient,
    }


def shuffleOrder(order: list[int], seed: int) -> list[int]:
    result = order.copy()
    state = seed
    index = len(result)
    while index > 1:
        state = (state * 1103515245 + 12345) % 2147483648
        swapIndex = state % index
        result[index - 1], result[swapIndex] = (
            result[swapIndex],
            result[index - 1],
        )
        index -= 1
    return result


def expectedCheckpoint(
    fixture: dict[str, np.ndarray],
    learningRate: float,
    batchSize: int,
    epochs: int,
    regularization: float,
    shuffleSeed: int | None,
) -> dict[str, np.ndarray]:
    sampleCount = fixture["train_input"].shape[0]
    if batchSize <= 0:
        batchSize = sampleCount
    if sampleCount <= 0 or epochs <= 0:
        raise ValueError("sample count and epochs must be positive")

    parameters = {name: fixture[name].copy() for name in PARAMETER_NAMES}
    order = list(range(sampleCount))
    eta = np.float32(learningRate)
    regularizationValue = np.float32(regularization)
    trainingSize = np.float32(sampleCount)
    weightScale = np.float32(
        np.float32(1.0)
        - np.float32(eta * regularizationValue) / trainingSize
    )

    for epoch in range(epochs):
        if shuffleSeed is not None:
            order = shuffleOrder(order, shuffleSeed + epoch)

        for batchStart in range(0, sampleCount, batchSize):
            batch = order[batchStart : batchStart + batchSize]
            gradientSum = {
                name: np.zeros_like(parameters[name]) for name in PARAMETER_NAMES
            }
            for sample in batch:
                gradient = cnnGradient(
                    parameters,
                    fixture["train_input"][sample : sample + 1],
                    fixture["train_label"][sample],
                )
                for name in PARAMETER_NAMES:
                    gradientSum[name] = (
                        gradientSum[name] + gradient[name]
                    ).astype(np.float32)

            gradientScale = np.float32(eta / np.float32(len(batch)))
            for name in PARAMETER_NAMES:
                parameterScale = (
                    weightScale if name.endswith("weight") else np.float32(1.0)
                )
                parameters[name] = (
                    parameters[name] * parameterScale
                    - gradientSum[name] * gradientScale
                ).astype(np.float32)

    return parameters


def correctCount(
    parameters: dict[str, np.ndarray],
    inputs: np.ndarray,
    labels: np.ndarray,
) -> int:
    probabilities, _convolved, _pooled, _activated = cnnProbabilities(
        parameters, inputs
    )
    predictions = probabilities.argmax(axis=1)
    expected = labels.reshape((labels.shape[0], -1)).argmax(axis=1)
    return int(np.count_nonzero(predictions == expected))


def main() -> None:
    args = parseArgs()
    fixture = readSafetensors(args.fixture)
    checkpoint = readSafetensors(args.checkpoint)
    expected = expectedCheckpoint(
        fixture,
        args.learning_rate,
        args.batch_size,
        args.epochs,
        args.regularization,
        args.shuffle_seed,
    )

    hasMetrics = "metrics" in checkpoint
    if args.report_evaluation or hasMetrics:
        initial = {name: fixture[name] for name in PARAMETER_NAMES}
        beforeTrain = correctCount(
            initial, fixture["train_input"], fixture["train_label"]
        )
        afterTrain = correctCount(
            expected, fixture["train_input"], fixture["train_label"]
        )
        beforeTest = correctCount(
            initial, fixture["test_input"], fixture["test_label"]
        )
        afterTest = correctCount(
            expected, fixture["test_input"], fixture["test_label"]
        )
        beforeTrainLoss = meanCrossEntropy(
            initial, fixture["train_input"], fixture["train_label"]
        )
        afterTrainLoss = meanCrossEntropy(
            expected, fixture["train_input"], fixture["train_label"]
        )
        beforeTestLoss = meanCrossEntropy(
            initial, fixture["test_input"], fixture["test_label"]
        )
        afterTestLoss = meanCrossEntropy(
            expected, fixture["test_input"], fixture["test_label"]
        )

        if afterTrain <= beforeTrain or afterTest <= beforeTest:
            raise AssertionError("bounded CNN accuracy did not improve")
        if afterTrainLoss >= beforeTrainLoss or afterTestLoss >= beforeTestLoss:
            raise AssertionError("bounded CNN cross-entropy did not improve")

        if args.report_evaluation:
            print(
                f"NumPy CNN train correct: {beforeTrain} -> {afterTrain} / "
                f"{len(fixture['train_input'])}"
            )
            print(
                "NumPy CNN train loss: "
                f"{beforeTrainLoss:.6f} -> {afterTrainLoss:.6f}"
            )
            print(
                f"NumPy CNN test correct: {beforeTest} -> {afterTest} / "
                f"{len(fixture['test_input'])}"
            )
            print(
                "NumPy CNN test loss: "
                f"{beforeTestLoss:.6f} -> {afterTestLoss:.6f}"
            )

        if hasMetrics:
            expectedMetrics = np.array(
                [
                    beforeTrainLoss,
                    afterTrainLoss,
                    beforeTestLoss,
                    afterTestLoss,
                ],
                dtype=np.float32,
            )
            if checkpoint["metrics"].shape != expectedMetrics.shape:
                raise AssertionError(
                    "metrics shape "
                    f"{checkpoint['metrics'].shape} != {expectedMetrics.shape}"
                )
            np.testing.assert_allclose(
                checkpoint["metrics"], expectedMetrics,
                rtol=2.0e-4, atol=2.0e-5
            )

    for name in PARAMETER_NAMES:
        if checkpoint[name].shape != expected[name].shape:
            raise AssertionError(
                f"{name} shape {checkpoint[name].shape} != {expected[name].shape}"
            )
        np.testing.assert_allclose(
            checkpoint[name], expected[name], rtol=2.0e-4, atol=2.0e-5
        )

    print("NumPy Nielsen CNN mini-batch checkpoint: ok")


if __name__ == "__main__":
    main()
