"""Export a deterministic Nielsen-sized CNN reference fixture.

The topology follows the Chapter 6 ``omit_FC`` experiment's real MNIST
dimensions, with ReLU selected as the convolution activation:

    NCHW input -> OIHW conv -> 2x2 max pool -> ReLU -> flatten -> softmax

The original repository does not save trained CNN parameters.  The parameters
here are deterministic reference data, not a pretrained Nielsen checkpoint.
NumPy computes the expected probabilities independently so Mulberry can verify
its complete safetensors inference path against them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from export_mnist_raw_tensors import DEFAULT_MNIST_DATA
from export_mnist_safetensors import writeSafetensors
from export_mnist_training_safetensors import loadMnistData


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data/nielsen-cnn-relu.safetensors"
DEFAULT_SEED = 20260710
FILTER_SHAPE = (20, 1, 5, 5)
POOL_SHAPE = (2, 2)
CLASS_COUNT = 10


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a deterministic Nielsen-sized CNN fixture."
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
        help="First MNIST test_data sample index to export.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=1,
        help="Number of MNIST training_data samples to export.",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=1,
        help="Number of consecutive MNIST test_data samples to export.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for deterministic reference parameters.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the output safetensors file.",
    )
    return parser.parse_args()


def conv2dNchw(
    inputValue: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    batchSize, inputChannels, inputHeight, inputWidth = inputValue.shape
    outputChannels, weightChannels, kernelHeight, kernelWidth = weight.shape
    if inputChannels != weightChannels:
        raise ValueError("convolution input and weight channels must match")
    if bias.shape != (outputChannels,):
        raise ValueError("convolution bias must contain one value per filter")

    outputHeight = inputHeight - kernelHeight + 1
    outputWidth = inputWidth - kernelWidth + 1
    output = np.empty(
        (batchSize, outputChannels, outputHeight, outputWidth),
        dtype=np.float32,
    )
    for row in range(outputHeight):
        for column in range(outputWidth):
            window = inputValue[
                :, :, row : row + kernelHeight, column : column + kernelWidth
            ]
            values = np.tensordot(
                window,
                weight,
                axes=((1, 2, 3), (1, 2, 3)),
            )
            output[:, :, row, column] = values + bias
    return output


def maxPool2dNchw(inputValue: np.ndarray) -> np.ndarray:
    batchSize, channels, inputHeight, inputWidth = inputValue.shape
    poolHeight, poolWidth = POOL_SHAPE
    outputHeight = inputHeight // poolHeight
    outputWidth = inputWidth // poolWidth
    cropped = inputValue[
        :, :, : outputHeight * poolHeight, : outputWidth * poolWidth
    ]
    windows = cropped.reshape(
        batchSize,
        channels,
        outputHeight,
        poolHeight,
        outputWidth,
        poolWidth,
    )
    return windows.max(axis=(3, 5))


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - value.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def oneHotLabels(labels: np.ndarray) -> np.ndarray:
    values = np.zeros((len(labels), 1, CLASS_COUNT), dtype=np.float32)
    for index, label in enumerate(labels):
        values[index, 0, int(label)] = 1.0
    return values


def createFixture(
    mnistData: Path,
    sampleIndex: int,
    sampleCount: int,
    testCount: int,
    seed: int,
) -> tuple[list[tuple[str, np.ndarray]], int, int]:
    if sampleCount <= 0:
        raise ValueError("sample count must be positive")
    if sampleIndex < 0 or testCount <= 0:
        raise ValueError("sample index must be non-negative and test count positive")

    (trainingInputs, trainingLabels), (testInputs, testLabels) = loadMnistData(
        mnistData
    )
    if sampleCount > len(trainingLabels):
        raise ValueError("sample count exceeds the MNIST training split")
    if sampleIndex + testCount > len(testLabels):
        raise ValueError("requested test range exceeds the MNIST test split")

    trainInput = np.asarray(
        trainingInputs[:sampleCount], dtype=np.float32
    ).reshape((sampleCount, 1, 28, 28))
    trainLabels = np.asarray(trainingLabels[:sampleCount], dtype=np.int64)
    testInput = np.asarray(
        testInputs[sampleIndex : sampleIndex + testCount], dtype=np.float32
    ).reshape((testCount, 1, 28, 28))
    selectedTestLabels = np.asarray(
        testLabels[sampleIndex : sampleIndex + testCount], dtype=np.int64
    )

    rng = np.random.RandomState(seed)
    filterCount = FILTER_SHAPE[0]
    filterElementCount = FILTER_SHAPE[0] * FILTER_SHAPE[2] * FILTER_SHAPE[3]
    poolElementCount = POOL_SHAPE[0] * POOL_SHAPE[1]
    convScale = np.sqrt(poolElementCount / filterElementCount)
    convWeight = rng.normal(0.0, convScale, FILTER_SHAPE).astype(np.float32)
    convBias = rng.normal(0.0, 1.0, (filterCount,)).astype(np.float32)

    # Nielsen adds each channel's bias after pooling. A channel-constant bias
    # commutes with max pooling, so the package's fused conv bias is equivalent.
    referenceInput = testInput[:1]
    convOutput = conv2dNchw(referenceInput, convWeight, convBias)
    pooled = maxPool2dNchw(convOutput)
    activated = np.maximum(pooled, np.float32(0.0))
    features = activated.reshape((1, -1))

    classifierShape = (features.shape[1], CLASS_COUNT)
    classifierScale = np.sqrt(1.0 / CLASS_COUNT)
    classifierWeight = rng.normal(
        0.0, classifierScale, classifierShape
    ).astype(np.float32)
    classifierBias = rng.normal(0.0, 1.0, (1, CLASS_COUNT)).astype(np.float32)
    logits = features @ classifierWeight + classifierBias
    probabilities = softmax(logits).astype(np.float32)
    prediction = int(probabilities.argmax(axis=1)[0])

    trainLabel = oneHotLabels(trainLabels)
    testLabel = oneHotLabels(selectedTestLabels)
    tensors = [
        ("input", testInput[:1]),
        ("conv1_weight", convWeight),
        ("conv1_bias", convBias),
        ("classifier_weight", classifierWeight),
        ("classifier_bias", classifierBias),
        ("reference_probabilities", probabilities[:1]),
        ("label", testLabel[0]),
        ("train_input", trainInput),
        ("train_label", trainLabel),
        ("test_input", testInput),
        ("test_label", testLabel),
    ]
    return tensors, prediction, int(selectedTestLabels[0])


def main() -> None:
    args = parseArgs()
    tensors, prediction, label = createFixture(
        args.mnist_data,
        args.sample_index,
        args.sample_count,
        args.test_count,
        args.seed,
    )
    byteCount = writeSafetensors(args.output, tensors)

    print(f"{args.output}: deterministic reference fixture, {byteCount} bytes")
    print("topology: NCHW conv -> 2x2 pool -> ReLU -> flatten -> softmax")
    for name, value in tensors:
        print(f"{name}: F32{list(value.shape)}")
    print(f"reference prediction: {prediction}")
    print(f"MNIST label: {label}")
    print("parameters: deterministic fixture, not a trained Nielsen checkpoint")


if __name__ == "__main__":
    main()
