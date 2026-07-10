"""Benchmark Nielsen CNN compilation, JIT, and native execution paths."""

from __future__ import annotations

import argparse
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIVER = REPO_ROOT / "build/release/bin/mulberry-driver"
EXPORTER = REPO_ROOT / "tools/export_nielsen_cnn_safetensors.py"

EVALUATE_SOURCE_TEMPLATE = """\
import mulberry.nn;
import safetensors;

fn main(): UInt64 {
  const file: safetensors.TensorFile =
      safetensors.open("data/nielsen-cnn-relu.safetensors");
  const parameters: nn.CnnParameters = nn.CnnParameters {
    safetensors.read(file, "conv1_weight"),
    safetensors.read(file, "conv1_bias"),
    safetensors.read(file, "classifier_weight"),
    safetensors.read(file, "classifier_bias")
  };
  var test: nn.TensorDataset = nn.datasetFromBatches(
      safetensors.read(file, "test_input"),
      safetensors.read(file, "test_label"));
  safetensors.close(file);
@EVALUATION@
  return 0;
}
"""

EVALUATION_BODIES = {
    "accuracy": """\
  const evaluation: nn.Evaluation = nn.cnnEvaluateDataset(parameters, test);
  io.print(evaluation.correct);""",
    "loss": """\
  const loss: Float32 = nn.cnnMeanCrossEntropy(parameters, test);
  var valid: UInt64 = 0;
  if loss >= 0.0 {
    valid = 1;
  }
  io.print(valid);""",
    "accuracy+loss": """\
  const evaluation: nn.Evaluation = nn.cnnEvaluateDataset(parameters, test);
  const loss: Float32 = nn.cnnMeanCrossEntropy(parameters, test);
  var valid: UInt64 = 0;
  if loss >= 0.0 {
    valid = 1;
  }
  io.print(evaluation.correct);
  io.print(valid);""",
}

TRAIN_SOURCE = """\
import mulberry.nn;
import safetensors;

fn main(): UInt64 {
  const file: safetensors.TensorFile =
      safetensors.open("data/nielsen-cnn-relu.safetensors");
  var parameters: nn.CnnParameters = nn.CnnParameters {
    safetensors.read(file, "conv1_weight"),
    safetensors.read(file, "conv1_bias"),
    safetensors.read(file, "classifier_weight"),
    safetensors.read(file, "classifier_bias")
  };
  var train: nn.TensorDataset = nn.datasetFromBatches(
      safetensors.read(file, "train_input"),
      safetensors.read(file, "train_label"));
  var test: nn.TensorDataset = nn.datasetFromBatches(
      safetensors.read(file, "test_input"),
      safetensors.read(file, "test_label"));
  safetensors.close(file);

  for epoch in 0 .. @EPOCHS@ {
    parameters = nn.cnnTrainEpoch(
        parameters, train, @BATCH_SIZE@, 0.01, 0.01, 12345 + epoch);
  }
  io.print(nn.cnnPredict(parameters, test.input(0)));
  return 0;
}
"""


def parseList(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Mulberry Nielsen CNN compilation and execution."
    )
    parser.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    parser.add_argument("--samples", type=parseList, default=parseList("1,2,10"))
    parser.add_argument(
        "--batch-sizes", type=parseList, default=parseList("1,2,5")
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--native-repeats", type=int, default=11)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--opt", action="store_true")
    return parser.parse_args()


def run(command: list[str], workdir: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=workdir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr}"
        )


def medianSeconds(action: Callable[[], None], repeats: int) -> float:
    # Warm filesystem, dynamic libraries, and page cache before recording.
    action()
    measurements = []
    for _ in range(repeats):
        start = time.perf_counter()
        action()
        measurements.append(time.perf_counter() - start)
    return statistics.median(measurements)


def driverCommand(driver: Path, opt: bool, arguments: list[str]) -> list[str]:
    command = [str(driver)]
    if opt:
        command.append("--opt")
    command.extend(arguments)
    return command


def writeSources(
    root: Path, batchSizes: list[int], epochs: int
) -> tuple[dict[str, Path], dict[int, Path]]:
    evaluateSources = {}
    for name, body in EVALUATION_BODIES.items():
        source = root / f"evaluate-{name.replace('+', '-and-')}.mulberry"
        contents = EVALUATE_SOURCE_TEMPLATE.replace("@EVALUATION@", body)
        source.write_text(contents, encoding="utf-8")
        evaluateSources[name] = source

    trainSources = {}
    for batchSize in batchSizes:
        source = root / f"train-b{batchSize}.mulberry"
        contents = TRAIN_SOURCE.replace("@BATCH_SIZE@", str(batchSize)).replace(
            "@EPOCHS@", str(epochs)
        )
        source.write_text(contents, encoding="utf-8")
        trainSources[batchSize] = source
    return evaluateSources, trainSources


def createFixtures(root: Path, sampleCounts: list[int]) -> dict[int, Path]:
    workdirs = {}
    for sampleCount in sampleCounts:
        workdir = root / f"samples-{sampleCount}"
        dataDir = workdir / "data"
        dataDir.mkdir(parents=True)
        output = dataDir / "nielsen-cnn-relu.safetensors"
        print(f"preparing {sampleCount}-sample fixture", file=sys.stderr)
        run(
            [
                sys.executable,
                str(EXPORTER),
                "--sample-count",
                str(sampleCount),
                "--test-count",
                str(sampleCount),
                "--output",
                str(output),
            ],
            REPO_ROOT,
        )
        workdirs[sampleCount] = workdir
    return workdirs


def compileExecutable(
    driver: Path, opt: bool, source: Path, output: Path, workdir: Path
) -> None:
    run(driverCommand(driver, opt, ["-o", str(output), str(source)]), workdir)


def formatMilliseconds(seconds: float) -> str:
    return f"{seconds * 1000.0:.2f}"


def main() -> None:
    args = parseArgs()
    if args.epochs <= 0 or args.repeats <= 0 or args.native_repeats <= 0:
        raise ValueError("epochs and repeat counts must be positive")
    if args.cpu is not None:
        if args.cpu < 0:
            raise ValueError("CPU index must be non-negative")
        if not hasattr(os, "sched_getaffinity"):
            raise RuntimeError("CPU pinning is not supported on this platform")
        availableCpus = os.sched_getaffinity(0)
        if args.cpu not in availableCpus:
            raise ValueError(f"CPU {args.cpu} is not available")
        os.sched_setaffinity(0, {args.cpu})
    driver = args.driver.resolve()
    if not driver.is_file():
        raise FileNotFoundError(driver)

    with tempfile.TemporaryDirectory(prefix="mulberry-cnn-benchmark-") as temp:
        root = Path(temp)
        evaluateSources, trainSources = writeSources(
            root, args.batch_sizes, args.epochs
        )
        workdirs = createFixtures(root, args.samples)
        largestWorkdir = workdirs[max(args.samples)]

        objectPath = root / "train.o"
        executablePath = root / "train"
        compileBatchSize = min(
            args.batch_sizes, key=lambda value: abs(value - 2)
        )
        compileSource = trainSources[compileBatchSize]

        print("measuring compiler stages", file=sys.stderr)
        compilerStages = {
            "typecheck": medianSeconds(
                lambda: run(
                    driverCommand(driver, args.opt, ["-typecheck", str(compileSource)]),
                    largestWorkdir,
                ),
                args.repeats,
            ),
            "lowered MLIR": medianSeconds(
                lambda: run(
                    driverCommand(
                        driver, args.opt, ["-dump=lowered-mlir", str(compileSource)]
                    ),
                    largestWorkdir,
                ),
                args.repeats,
            ),
            "object": medianSeconds(
                lambda: run(
                    driverCommand(
                        driver, args.opt, [f"-c={objectPath}", str(compileSource)]
                    ),
                    largestWorkdir,
                ),
                args.repeats,
            ),
            "executable": medianSeconds(
                lambda: compileExecutable(
                    driver, args.opt, compileSource, executablePath, largestWorkdir
                ),
                args.repeats,
            ),
        }

        print("building native benchmark executables", file=sys.stderr)
        evaluateExecutables = {}
        for name, source in evaluateSources.items():
            executable = root / f"evaluate-{name.replace('+', '-and-')}"
            compileExecutable(driver, args.opt, source, executable, largestWorkdir)
            evaluateExecutables[name] = executable
        trainExecutables = {}
        for batchSize, source in trainSources.items():
            executable = root / f"train-b{batchSize}"
            compileExecutable(driver, args.opt, source, executable, largestWorkdir)
            trainExecutables[batchSize] = executable

        evaluationRows = []
        for sampleCount in args.samples:
            workdir = workdirs[sampleCount]
            for name, source in evaluateSources.items():
                print(
                    f"measuring {name} evaluation with {sampleCount} samples",
                    file=sys.stderr,
                )
                executable = evaluateExecutables[name]
                jit = medianSeconds(
                    lambda source=source, cwd=workdir: run(
                        driverCommand(driver, args.opt, [str(source)]), cwd
                    ),
                    args.repeats,
                )
                native = medianSeconds(
                    lambda executable=executable, cwd=workdir: run(
                        [str(executable)], cwd
                    ),
                    args.native_repeats,
                )
                evaluationRows.append((sampleCount, name, jit, native))

        trainingRows = []
        for sampleCount in args.samples:
            workdir = workdirs[sampleCount]
            for batchSize in args.batch_sizes:
                print(
                    f"measuring training with {sampleCount} samples, batch {batchSize}",
                    file=sys.stderr,
                )
                source = trainSources[batchSize]
                executable = trainExecutables[batchSize]
                jit = medianSeconds(
                    lambda source=source, cwd=workdir: run(
                        driverCommand(driver, args.opt, [str(source)]), cwd
                    ),
                    args.repeats,
                )
                native = medianSeconds(
                    lambda executable=executable, cwd=workdir: run(
                        [str(executable)], cwd
                    ),
                    args.native_repeats,
                )
                updates = math.ceil(sampleCount / batchSize) * args.epochs
                trainingRows.append((sampleCount, batchSize, updates, jit, native))

        print("# Nielsen CNN benchmark")
        print()
        print(f"- platform: `{platform.platform()}`")
        print(f"- machine: `{platform.machine()}`")
        print(f"- logical CPUs: `{os.cpu_count()}`")
        if hasattr(os, "sched_getaffinity"):
            affinity = ",".join(
                str(cpu) for cpu in sorted(os.sched_getaffinity(0))
            )
            print(f"- CPU affinity: `{affinity}`")
        print(f"- compiler/JIT repeats: median of {args.repeats} warm runs")
        print(
            f"- native repeats: median of {args.native_repeats} warm runs"
        )
        print(f"- compiler `--opt`: `{args.opt}`")
        print(f"- training epochs: `{args.epochs}`")
        print()
        print("## Compiler stages")
        print()
        print("| stage | median ms |")
        print("| --- | ---: |")
        for stage, seconds in compilerStages.items():
            print(f"| {stage} | {formatMilliseconds(seconds)} |")
        print()
        print("## Evaluation")
        print()
        print(
            "| samples | workload | JIT total ms | native ms | "
            "native ms/sample |"
        )
        print("| ---: | --- | ---: | ---: | ---: |")
        for sampleCount, name, jit, native in evaluationRows:
            print(
                f"| {sampleCount} | {name} | {formatMilliseconds(jit)} | "
                f"{formatMilliseconds(native)} | "
                f"{formatMilliseconds(native / sampleCount)} |"
            )
        print()
        print(f"## {args.epochs}-epoch training")
        print()
        print(
            "| samples | batch | updates | JIT total ms | native ms | "
            "native ms/sample/epoch |"
        )
        print("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for sampleCount, batchSize, updates, jit, native in trainingRows:
            perSampleEpoch = native / (sampleCount * args.epochs)
            print(
                f"| {sampleCount} | {batchSize} | {updates} | "
                f"{formatMilliseconds(jit)} | {formatMilliseconds(native)} | "
                f"{formatMilliseconds(perSampleEpoch)} |"
            )


if __name__ == "__main__":
    main()
