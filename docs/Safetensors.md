# Safetensors 数据文件

本文档记录 Mulberry 下一阶段的数据文件方向。当前 raw `.f32` 文件只是 bootstrap
格式；后续深度学习推理和训练数据优先使用 safetensors。

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
后续 Mulberry runtime 会读取同一个文件格式。

## Mulberry API

第一版使用 expected type，不做裸类型推断：

```cherry
const file: File = open("data/mnist-784-30-10.safetensors", "rb");

var w1: Float32[?, ?] = readTensor(file, "w1");
var b1: Float32[?, ?] = readTensor(file, "b1");
var w2: Float32[?, ?] = readTensor(file, "w2");
var b2: Float32[?, ?] = readTensor(file, "b2");
var x: Float32[?, ?] = readTensor(file, "x");

close(file);
```

expected type 提供 element type 和 rank。safetensors header 提供 concrete shape。

例如：

```cherry
var w1: Float32[?, ?] = readTensor(file, "w1");
```

含义是：

- Mulberry expected type 要求 `Float32` element。
- expected rank 是 2。
- runtime 从 header 读出实际 shape，例如 `[30, 784]`。
- runtime 检查 dtype 是 `F32`，rank 是 2。
- lowering 使用实际 shape 分配动态 Tensor storage。
- runtime 把 payload bytes 读入这个 Tensor。

静态和动态维度可以混合：

```cherry
var w1: Float32[30, ?] = readTensor(file, "w1");
```

这会额外检查 header 里的第一维必须是 `30`。

## 分层设计

`readTensor()` 不应该让 Mulberry script 解析 JSON。当前阶段的正确分层是：

```text
Mulberry source
  -> Sema: readTensor() 必须有 expected Tensor type
  -> MLIRGen: mulberry.safetensor.read
  -> Lowering: runtime helper + tensor allocation + payload read
```

第一版 runtime helper 可以每次 `readTensor(file, name)` 都 parse 一次 header。
MNIST 推理只会读 `w1`、`b1`、`w2`、`b2`、`x`，重复 parse header 的开销可以接受。
这样可以避免提前引入 `TensorFile` handle、cache 和 lifetime 设计。

后续如果需要优化，再增加：

```cherry
const tensors: TensorFile = openTensorFile("data/mnist-784-30-10.safetensors");
var w1: Float32[?, ?] = readTensor(tensors, "w1");
```

## Runtime Helper 责任

C++ runtime helper 负责：

- `seek(file, 0)`。
- 读前 8 bytes，得到 JSON header length。
- 读 JSON header。
- 查找 tensor name。
- 检查 dtype。
- 检查 rank 和静态维度。
- 读取 `shape` 和 `data_offsets`。
- 定位 payload offset。
- 把 payload bytes 读入 lowering 已经分配好的 Tensor data pointer。

compiler/lowering 负责：

- 根据 expected type 知道 element type 和 rank。
- 根据 runtime 返回的 concrete shape 分配动态 Tensor storage。
- 把 Tensor data pointer 传给 runtime。
- 返回 Tensor value。

## 第一版限制

- 只支持 `Float32` / safetensors `F32`。
- 只支持 Tensor，不支持 List 或 struct 直接从 safetensors 读取。
- 必须有 expected Tensor type。
- 不支持 `var w = readTensor(file, "w1");` 这种裸推断。
- shape mismatch、dtype mismatch、找不到 tensor 先 fail-fast。
- header 每次读取可以重复 parse，不做 cache。
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
