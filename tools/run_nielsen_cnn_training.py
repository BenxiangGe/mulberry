"""Build and run a configurable Nielsen CNN native training executable."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from export_mnist_raw_tensors import DEFAULT_MNIST_DATA


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIVER = REPO_ROOT / "build/release/bin/mulberry-driver"
EXPORTER = REPO_ROOT / "tools/export_nielsen_cnn_safetensors.py"
DEFAULT_CHECKPOINT = Path("/tmp/mulberry_nielsen_cnn_trained.safetensors")

TRAIN_SOURCE = """\
import mulberry.nn;
import safetensors;

fn main(): UInt64 {
  const file =
      safetensors.open("data/nielsen-cnn-relu.safetensors");
  var parameters = nn.CnnParameters {
    safetensors.read(file, "conv1_weight"),
    safetensors.read(file, "conv1_bias"),
    safetensors.read(file, "classifier_weight"),
    safetensors.read(file, "classifier_bias")
  };
  var train = nn.datasetFromBatches(
      safetensors.read(file, "train_input"),
      safetensors.read(file, "train_label"));
  var test = nn.datasetFromBatches(
      safetensors.read(file, "test_input"),
      safetensors.read(file, "test_label"));
  safetensors.close(file);

  const batchSize = @BATCH_SIZE@;
  const epochs = @EPOCHS@;
  const eta = @ETA@;
  const regularization = @REGULARIZATION@;
  const beforeTrain =
      nn.cnnEvaluateDataset(parameters, train);
  const beforeTest =
      nn.cnnEvaluateDataset(parameters, test);
  const beforeTrainLoss =
      nn.cnnMeanCrossEntropy(parameters, train);
  const beforeTestLoss =
      nn.cnnMeanCrossEntropy(parameters, test);
  for epoch in 0 .. epochs {
    parameters = nn.cnnTrainEpoch(
        parameters, train, batchSize, eta, regularization, @SEED@ + epoch);
  }
  const afterTrain =
      nn.cnnEvaluateDataset(parameters, train);
  const afterTest =
      nn.cnnEvaluateDataset(parameters, test);
  const afterTrainLoss =
      nn.cnnMeanCrossEntropy(parameters, train);
  const afterTestLoss =
      nn.cnnMeanCrossEntropy(parameters, test);
  var metrics = tensor.from([
    beforeTrainLoss, afterTrainLoss, beforeTestLoss, afterTestLoss
  ]);

  var names = list.from([
    "conv1_weight", "conv1_bias", "classifier_weight", "classifier_bias",
    "metrics"
  ]);
  var values = list.from([
    parameters.convWeight, parameters.convBias,
    parameters.classifierWeight, parameters.classifierBias, metrics
  ]);
  safetensors.write("checkpoint.safetensors", names, values);

  io.println(train.size());
  io.println(test.size());
  io.println(nn.batchCount(train.size(), batchSize));
  io.println(epochs);
  io.println(beforeTrain.correct);
  io.println(beforeTrain.accuracyBasisPoints);
  io.println(afterTrain.correct);
  io.println(afterTrain.accuracyBasisPoints);
  io.println(beforeTest.correct);
  io.println(beforeTest.accuracyBasisPoints);
  io.println(afterTest.correct);
  io.println(afterTest.accuracyBasisPoints);
  return 0;
}
"""


def positiveInt(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def positiveFloat(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return result


def nonnegativeFloat(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError(
            "value must be finite and non-negative"
        )
    return result


def seedValue(value: str) -> int:
    result = int(value)
    if result < 0 or result > 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("seed must fit in UInt32")
    return result


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run configurable Nielsen CNN training as a native executable."
    )
    parser.add_argument("--sample-count", type=positiveInt, default=10)
    parser.add_argument("--test-count", type=positiveInt, default=10)
    parser.add_argument("--batch-size", type=positiveInt, default=2)
    parser.add_argument("--epochs", type=positiveInt, default=3)
    parser.add_argument("--eta", type=positiveFloat, default=0.01)
    parser.add_argument(
        "--regularization", type=nonnegativeFloat, default=0.01
    )
    parser.add_argument("--seed", type=seedValue, default=12345)
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    parser.add_argument("--mnist-data", type=Path, default=DEFAULT_MNIST_DATA)
    return parser.parse_args()


def renderSource(args: argparse.Namespace) -> str:
    return (
        TRAIN_SOURCE.replace("@BATCH_SIZE@", str(args.batch_size))
        .replace("@EPOCHS@", str(args.epochs))
        .replace("@ETA@", repr(args.eta))
        .replace("@REGULARIZATION@", repr(args.regularization))
        .replace("@SEED@", str(args.seed))
    )


def main() -> None:
    args = parseArgs()
    driver = args.driver.expanduser().resolve()
    mnistData = args.mnist_data.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not driver.is_file():
        raise FileNotFoundError(driver)
    if not mnistData.is_file():
        raise FileNotFoundError(mnistData)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mulberry-cnn-training-") as temp:
        root = Path(temp)
        dataDir = root / "data"
        dataDir.mkdir()
        fixture = dataDir / "nielsen-cnn-relu.safetensors"
        source = root / "training.mulberry"
        executable = root / "training"
        temporaryCheckpoint = root / "checkpoint.safetensors"

        print(
            f"exporting {args.sample_count} train and {args.test_count} test samples",
            file=sys.stderr,
            flush=True,
        )
        exportStart = time.perf_counter()
        subprocess.run(
            [
                sys.executable,
                str(EXPORTER),
                "--mnist-data",
                str(mnistData),
                "--sample-count",
                str(args.sample_count),
                "--test-count",
                str(args.test_count),
                "--seed",
                str(args.seed),
                "--output",
                str(fixture),
            ],
            cwd=REPO_ROOT,
            check=True,
            stdout=sys.stderr,
        )
        print(
            f"fixture exported in {time.perf_counter() - exportStart:.2f} s",
            file=sys.stderr,
        )

        source.write_text(renderSource(args), encoding="utf-8")
        print(
            "compiling optimized native executable", file=sys.stderr, flush=True
        )
        compileStart = time.perf_counter()
        subprocess.run(
            [str(driver), "--opt", "-o", str(executable), str(source)],
            cwd=root,
            check=True,
            stdout=sys.stderr,
        )
        print(
            f"executable compiled in {time.perf_counter() - compileStart:.2f} s",
            file=sys.stderr,
        )

        print("running Nielsen CNN training", file=sys.stderr, flush=True)
        trainingStart = time.perf_counter()
        subprocess.run([str(executable)], cwd=root, check=True)
        print(
            f"native training completed in "
            f"{time.perf_counter() - trainingStart:.2f} s",
            file=sys.stderr,
        )
        if not temporaryCheckpoint.is_file():
            raise RuntimeError("training did not produce a checkpoint")
        shutil.copyfile(temporaryCheckpoint, checkpoint)

    print(f"checkpoint: {checkpoint}", file=sys.stderr)


if __name__ == "__main__":
    main()
