# 深度学习示例

这个目录下的文件现在主要作为独立 `mulberry.nn` package 的回归和设计素材。

Core Mulberry 目前仍然保留 tensor literal、`Tensor<T>`、`List<T>`、raw file IO
和 safetensors IO。神经网络 primitive，例如 `nn.matmul`、`nn.sigmoid`、
`nn.argmax` 等，已经从 core compiler 中移到独立 `mulberry.nn` package。

现在的路径是：

- 示例通过 `import mulberry.nn` 使用 NN package。
- `stdlib/mulberry/nn.mulberry` 里只声明普通 extern package function。
- `packages/nn` 里提供 `mulberry_nn` dialect、op、pass 和 lowering。
- Sema、MLIRGen 和 LowerMulberry 不再硬编码 NN primitive。

当前 `MulberryNNPackage` 已经能通过 package pass 处理 source-level
`nn.matmul(...)` 这类普通 package call，并把最终的 `mulberry_nn.*` op lower 到
`linalg` / `arith` / `math`。driver 通过 bundled package registry 检测
`import mulberry.nn`，自动加载 `MulberryNNPackage` 并接入 NN package pipeline，
所以 inference 示例可以作为普通 `mulberry-driver file.mulberry` 的默认 JIT smoke。

`matmul` 和 `transpose` 当前只实现 rank-2 tensor，刚好覆盖 Nielsen MNIST
推理/训练需要的路径。后续如果要支持更通用的 broadcasting、batch matmul 或
GPU backend，应该继续放在 `mulberry.nn` package 边界后面，而不是塞回
Sema、MLIRGen 或 LowerMulberry。

## Nielsen 尺寸 CNN inference

`inference_nielsen_cnn_safetensors.mulberry` 使用 Chapter 6 `omit_FC` 实验的
真实 MNIST 尺寸，并选择 ReLU 作为卷积层 activation：

```text
[1,1,28,28] -> Conv[20,1,5,5] -> Pool[2,2] -> ReLU
              -> [1,2880] -> Linear[2880,10] -> Softmax
```

先生成确定性的 NumPy reference fixture，再运行 Mulberry：

```bash
/usr/bin/python3 tools/export_nielsen_cnn_safetensors.py
./build/release/bin/mulberry-driver \
  examples/dl/inference_nielsen_cnn_safetensors.mulberry
```

原始 `conv.py` 不保存 CNN checkpoint，因此这个文件的参数是固定 seed 的 reference
fixture，不是 Nielsen 的已训练权重。exporter 会用 NumPy 独立计算 reference
probabilities；Mulberry 示例输出自己的 prediction、NumPy reference prediction 和
MNIST label。

## Nielsen 尺寸 CNN mini-batch training

`training_nielsen_cnn_safetensors.mulberry` 复用同一套 CNN topology，并通过
`nn.cnnTrainEpoch()` 完成 deterministic shuffle、tail batch、逐样本 gradient sum、
L2 weight decay 和 scaled mini-batch update。当前示例仍然使用固定 seed 参数和一个
很小且彼此独立的 MNIST training/test 子集，目的是验证完整训练基础设施和准确率
与 loss 方向，不冒充收敛后的模型。`nn.softmaxCrossEntropy()` 直接对 logits 使用
stable log-sum-exp；`nn.cnnMeanCrossEntropy()` 在 dataset 上汇总平均 loss。

生成 10 个训练样本、10 个测试样本并运行三个 epoch：

```bash
/usr/bin/python3 tools/export_nielsen_cnn_safetensors.py \
  --sample-count 10 --test-count 10
./build/release/bin/mulberry-driver \
  examples/dl/training_nielsen_cnn_safetensors.mulberry
```

训练后的四个参数 tensor 和 `[beforeTrainLoss, afterTrainLoss,
beforeTestLoss, afterTestLoss]` metrics tensor 会写入
`/tmp/mulberry_nielsen_cnn_trained.safetensors`。可以用独立 NumPy reference 逐元素
复算同样的 shuffle、batch、SGD 更新和 stable cross-entropy：

```bash
/usr/bin/python3 tools/check_nielsen_cnn_checkpoint.py \
  --fixture data/nielsen-cnn-relu.safetensors \
  --checkpoint /tmp/mulberry_nielsen_cnn_trained.safetensors \
  --batch-size 2 --epochs 3 --learning-rate 0.01 \
  --regularization 0.01 --shuffle-seed 12345 --report-evaluation
```

固定 fixture 的 convergence smoke 中，training correct count 为 `0 -> 7`，loss 为
`9.470685 -> 0.923963`；独立 test correct count 为 `1 -> 2`，loss 为
`9.038157 -> 3.126420`。测试会同时要求 accuracy 改善、loss 下降，并逐元素比较
Mulberry checkpoint 与 NumPy reference。这些小样本数字只用于锁定训练方向与实现
一致性，不代表模型质量。

`tools/benchmark_nielsen_cnn.py` 会分别测量 accuracy、stable cross-entropy、两者
组合以及 training，并把 compiler/JIT 与短 native 进程使用的重复次数分开。可复现的
固定 CPU 基线见 [Nielsen CNN 性能基线](../../docs/CnnPerformance.md)。
