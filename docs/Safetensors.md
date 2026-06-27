# Safetensors 数据文件

本文档记录 Mulberry 当前推荐的数据文件方向。raw `.f32` 文件只是
bootstrap/debug 格式；深度学习推理和后续训练数据优先使用 safetensors。

官方格式说明见：

- https://github.com/huggingface/safetensors
- https://huggingface.co/docs/safetensors/metadata_parsing

## 为什么选择 Safetensors

safetensors 很适合当前 Mulberry 的需求：

- 单文件可以保存多个 tensor，例如 `w1`、`b1`、`w2`、`b2`、`x`。
- 文件开头有 JSON header，记录 tensor 名字、dtype、shape 和 byte offsets。
- tensor payload 是连续 raw bytes，没有 pickle opcode，也不需要 zip/gzip。
- 可以只读取某个 tensor，不需要把整个文件都读入内存。
- 文件格式足够简单，可以先用 C++ runtime 实现，后续再用 Mulberry script 实现。

## 文件 Layout

safetensors 文件由三段组成：

```text
8-byte little-endian header length
JSON header
contiguous tensor payload bytes
```

JSON header 里的每个 tensor entry 类似：

```json
{
  "w1": {
    "dtype": "F32",
    "shape": [30, 784],
    "data_offsets": [0, 94080]
  }
}
```

`data_offsets` 是相对于 payload 开始位置的 offset。真实文件 offset 是：

```text
8 + headerLength + data_offsets[0]
```

## MNIST 导出

当前可以用下面的脚本导出 Nielsen MNIST 推理所需的单文件 safetensors：

```sh
python3 tools/export_mnist_safetensors.py
```

默认输出文件：

```text
data/mnist-784-30-10.safetensors
```

文件包含：

```text
w1  F32[30, 784]
b1  F32[30, 1]
w2  F32[10, 30]
b2  F32[10, 1]
x   F32[784, 1]
```

这个导出工具直接写 safetensors layout，不依赖 Python `safetensors` package。
Mulberry runtime 读取同一个文件格式。

这些文件和示例现在作为外部 NN package 的回归素材保留。当前已经有独立
`MulberryNNPackage`，driver 通过 bundled package registry 检测
`import mulberry.nn`，自动加载 package，并接入 source-level `nn.matmul(...)`
到 `mulberry_nn.*` / `linalg` 的 lowering。
下面的命令已经重新成为默认 JIT smoke：

```sh
./build/release/bin/mulberry-driver examples/dl/inference_mnist_safetensors.cherry
```

`test_data[0]` 的期望预测结果是：

```text
7
```

也可以导出一个小的 training 子集：

```sh
python3 tools/export_mnist_training_safetensors.py
```

默认输出文件：

```text
data/mnist-784-30-10-training.safetensors
```

文件包含初始网络参数，以及按名字展开的 training 样本：

```text
w1         F32[30, 784]
b1         F32[30, 1]
w2         F32[10, 30]
b2         F32[10, 1]
train_x_0  F32[784, 1]
train_y_0  F32[10, 1]
train_x_1  F32[784, 1]
train_y_1  F32[10, 1]
...
train_x_9  F32[784, 1]
train_y_9  F32[10, 1]
```

`train_y_N` 是 one-hot label。当前 Mulberry 还没有 tensor slice 或 dataset
iterator，所以 training bootstrap 先把每个样本导成独立 named tensor。后续如果
语言补齐 dataset 抽象，可以改成 `train_x: Float32[?, 784, 1]` 和
`train_y: Float32[?, 10, 1]` 这样的批量布局。

这个 smoke 使用默认导出的 `10` 个 training 样本跑 `30` 个 epoch 的 per-sample SGD，然后读取
`data/mnist-784-30-10.safetensors` 里的 `x` 做一次 inference，期望输出是 `7`。
输出层 delta 采用 Nielsen `network2.py` 默认的 CrossEntropy 形式：
`delta = a - y`。mini-batch、shuffle、L2 regularization 和保存训练结果后续再补。

```sh
./build/release/bin/mulberry-driver examples/dl/training_mnist_safetensors.cherry
```

## Mulberry API

第一版固定返回 `Tensor<Float32>`，不做裸类型推断：

```cherry
const file: File = io.open("data/mnist-784-30-10.safetensors", "rb");

var w1: Tensor<Float32> = io.readTensor(file, "w1");
var b1: Tensor<Float32> = io.readTensor(file, "b1");
var w2: Tensor<Float32> = io.readTensor(file, "w2");
var b2: Tensor<Float32> = io.readTensor(file, "b2");
var x: Tensor<Float32> = io.readTensor(file, "x");

io.close(file);
```

safetensors header 提供 dtype、concrete shape 和 payload offset。当前推荐直接使用
`Tensor<Float32>` header：

```cherry
var w1: Tensor<Float32> = io.readTensor(file, "w1");
```

含义是：

- stdlib 签名要求返回 `Tensor<Float32>`。
- runtime 检查 safetensors dtype 是 `F32`。
- runtime 从 header 读出实际 shape，例如 `[30, 784]`。
- runtime 分配 payload、sizes 和 strides buffer，最终返回 `Tensor<Float32>` header。

## 分层设计

`io.readTensor()` 不应该让 Mulberry script 解析 JSON。当前阶段的正确分层是：

```text
Mulberry source
  -> stdlib: io.readTensor(file, name)
  -> MLIRGen: ordinary func.call @std.io.readTensor
  -> Lowering: ordinary extern call @mulberry_safetensor_read_tensor_f32
  -> Runtime: parse header, allocate Tensor<Float32> header buffers, read payload
```

当前 runtime helper 每次 `io.readTensor(file, name)` 都 parse 一次 header。
MNIST 推理只会读 `w1`、`b1`、`w2`、`b2`、`x`，重复 parse header 的开销可以接受。
这样可以避免提前引入 `TensorFile` handle、cache 和 lifetime 设计。

后续如果需要优化，再增加：

```cherry
const tensors: TensorFile = openTensorFile("data/mnist-784-30-10.safetensors");
var w1: Tensor<Float32> = io.readTensor(tensors, "w1");
```

## Runtime Helper 责任

C++ runtime helper 负责：

- `seek(file, 0)`。
- 读前 8 bytes，得到 JSON header length。
- 读 JSON header。
- 查找 tensor name。
- 检查 dtype。
- 读取 `shape` 和 `data_offsets`。
- 定位 payload offset。
- 分配 payload、sizes 和 strides buffer。
- 返回 `Tensor<Float32>` header。

compiler/lowering 负责把 stdlib `io.readTensor(file, name)` 降成普通 extern
runtime call，不再插入 safetensors-specific 隐藏调用。

## 当前限制

- 只支持 `Float32` / safetensors `F32`。
- 只支持 Tensor，不支持 List 或 struct 直接从 safetensors 读取。
- `io.readTensor()` 统一返回 `Tensor<Float32>` header。
- 不支持 `var w = io.readTensor(file, "w1");` 这种裸推断。
- training 数据当前使用 `train_x_N` / `train_y_N` named tensor bootstrap 布局，
  还不是最终 dataset API。
- shape mismatch、dtype mismatch、找不到 tensor 先 fail-fast。
- header 每次读取都会重复 parse，不做 cache。
- 暂不支持写 safetensors。

## 后续方向

training 跑通以后，可以把 safetensors reader 作为 Mulberry 语言能力增强任务。那时
再用 Mulberry script 实现 header parsing 会自然推动这些真实能力：

- byte / string indexing
- slice
- string compare
- integer parsing
- dynamic List
- Map 或 record-like dictionary
- error handling
- 精确 file seek/read

当前阶段先用 C++ runtime 托底，避免为了 JSON parsing 过早扩张语言核心。
