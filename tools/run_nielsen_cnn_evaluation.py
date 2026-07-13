"""Build and run native evaluation for a saved Nielsen CNN checkpoint."""

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
DEFAULT_CHECKPOINT = REPO_ROOT / "data/nielsen-cnn-trained-full.safetensors"
EXPORTER = REPO_ROOT / "tools/export_nielsen_cnn_safetensors.py"

EVALUATE_SOURCE = """\
import mulberry.nn;
import safetensors;

fn main(): UInt64 {
  const checkpoint = safetensors.open("data/checkpoint.safetensors");
  const parameters = nn.CnnParameters {
    safetensors.read(checkpoint, "conv1_weight"),
    safetensors.read(checkpoint, "conv1_bias"),
    safetensors.read(checkpoint, "classifier_weight"),
    safetensors.read(checkpoint, "classifier_bias")
  };
  safetensors.close(checkpoint);

  const fixture = safetensors.open("data/test.safetensors");
  var test = nn.datasetFromBatches(
      safetensors.read(fixture, "test_input"),
      safetensors.read(fixture, "test_label"));
  safetensors.close(fixture);

  const evaluation = nn.cnnEvaluateDataset(parameters, test);
  const loss = nn.cnnMeanCrossEntropy(parameters, test);
  io.println(test.size());
  io.println(evaluation.correct);
  io.println(evaluation.accuracyBasisPoints);
  io.println(loss);
  return 0;
}
"""


def positiveInt(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def finiteFloat(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("value must be finite")
    return result


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Nielsen CNN checkpoint natively."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-count", type=positiveInt, default=10000)
    parser.add_argument("--expected-correct", type=int)
    parser.add_argument("--expected-loss", type=finiteFloat)
    parser.add_argument("--loss-tolerance", type=finiteFloat, default=1.0e-6)
    parser.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    parser.add_argument("--mnist-data", type=Path, default=DEFAULT_MNIST_DATA)
    return parser.parse_args()


def parseEvaluation(output: str) -> tuple[int, int, int, float]:
    lines = output.splitlines()
    if len(lines) != 4:
        raise RuntimeError(
            f"evaluation produced {len(lines)} lines, expected 4:\n{output}"
        )
    try:
        testCount = int(lines[0])
        correct = int(lines[1])
        accuracyBasisPoints = int(lines[2])
        loss = float(lines[3])
    except ValueError as error:
        raise RuntimeError(f"invalid evaluation output:\n{output}") from error
    if not math.isfinite(loss):
        raise RuntimeError(f"evaluation produced non-finite loss: {loss}")
    return testCount, correct, accuracyBasisPoints, loss


def verifyEvaluation(
    args: argparse.Namespace,
    testCount: int,
    correct: int,
    accuracyBasisPoints: int,
    loss: float,
) -> None:
    if testCount != args.test_count:
        raise RuntimeError(
            f"test count mismatch: {testCount} != {args.test_count}"
        )
    expectedBasisPoints = correct * 10000 // testCount
    if accuracyBasisPoints != expectedBasisPoints:
        raise RuntimeError(
            "accuracy basis points mismatch: "
            f"{accuracyBasisPoints} != {expectedBasisPoints}"
        )
    if args.expected_correct is not None and correct != args.expected_correct:
        raise RuntimeError(
            f"correct count mismatch: {correct} != {args.expected_correct}"
        )
    if args.expected_loss is not None and not math.isclose(
        loss,
        args.expected_loss,
        rel_tol=0.0,
        abs_tol=args.loss_tolerance,
    ):
        raise RuntimeError(
            f"loss mismatch: {loss} != {args.expected_loss} "
            f"within {args.loss_tolerance}"
        )


def main() -> None:
    args = parseArgs()
    checkpoint = args.checkpoint.expanduser().resolve()
    driver = args.driver.expanduser().resolve()
    mnistData = args.mnist_data.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not driver.is_file():
        raise FileNotFoundError(driver)
    if not mnistData.is_file():
        raise FileNotFoundError(mnistData)
    if args.loss_tolerance < 0.0:
        raise ValueError("loss tolerance must be non-negative")

    with tempfile.TemporaryDirectory(prefix="mulberry-cnn-evaluation-") as temp:
        root = Path(temp)
        dataDir = root / "data"
        dataDir.mkdir()
        temporaryCheckpoint = dataDir / "checkpoint.safetensors"
        fixture = dataDir / "test.safetensors"
        source = root / "evaluation.mulberry"
        executable = root / "evaluation"
        shutil.copyfile(checkpoint, temporaryCheckpoint)

        print(
            f"exporting {args.test_count} MNIST test samples",
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
                "1",
                "--test-count",
                str(args.test_count),
                "--output",
                str(fixture),
            ],
            cwd=REPO_ROOT,
            check=True,
            stdout=sys.stderr,
        )
        print(
            f"test fixture exported in {time.perf_counter() - exportStart:.2f} s",
            file=sys.stderr,
        )

        source.write_text(EVALUATE_SOURCE, encoding="utf-8")
        print(
            "compiling optimized native evaluator", file=sys.stderr, flush=True
        )
        compileStart = time.perf_counter()
        subprocess.run(
            [str(driver), "--opt", "-o", str(executable), str(source)],
            cwd=root,
            check=True,
            stdout=sys.stderr,
        )
        print(
            f"evaluator compiled in {time.perf_counter() - compileStart:.2f} s",
            file=sys.stderr,
        )

        print("running Nielsen CNN evaluation", file=sys.stderr, flush=True)
        evaluationStart = time.perf_counter()
        completed = subprocess.run(
            [str(executable)],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        testCount, correct, accuracyBasisPoints, loss = parseEvaluation(
            completed.stdout
        )
        verifyEvaluation(
            args, testCount, correct, accuracyBasisPoints, loss
        )
        print(
            f"native evaluation completed in "
            f"{time.perf_counter() - evaluationStart:.2f} s",
            file=sys.stderr,
        )

    print(f"test samples: {testCount}")
    print(f"correct: {correct}")
    print(f"accuracy: {accuracyBasisPoints / 100.0:.2f}%")
    print(f"cross-entropy: {loss:.9g}")


if __name__ == "__main__":
    main()
