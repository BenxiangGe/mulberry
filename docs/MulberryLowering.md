# Mulberry Lowering

本文档记录当前 Mulberry lowering 的边界、ABI 形状和仍然保留的限制。核心原则是：
MLIRGen 只生成自洽的高层 Mulberry IR；函数边界和后端 ABI 由独立 lowering pass
处理，不能把 workaround 塞回 MLIRGen。

## 当前边界

`--lower-mulberry` 现在做的是 storage-level lowering；完整 driver pipeline 会继续
lower 到 LLVM dialect、LLVM IR，并在当前正向子集上执行 JIT：

- `mulberry.tensor` lower 成 MLIR `memref`，方便 `cherry_nn` 继续 lower 到
  `linalg`、`math`、`arith` 和 `memref`。
- `mulberry.list<T>` 在函数内部 lower 成本地 list storage。
- scalar `List<T>` 和 `List<Tensor>` 穿过受支持的函数边界时，会先 rewrite 成
  descriptor 语义，再 lower 成 LLVM descriptor。
- scalar 和 record storage 在能直接映射的地方使用 LLVM dialect。
- `-dump=mlir-llvm`、`-dump=llvm` 和 JIT 已经打开，用来验证当前支持的正向路径。
- object generation 仍然关闭，后续需要单独设计 emission path。

## IR 分层

`!mulberry.list<T>` 是源语言语义层的 `List<T>` value。MLIRGen 为用户可见的
list 生成这个类型。

`!mulberry.list_storage<T>` 是 lowering 中使用的连续元素 storage。它是实现细节，
不是函数 ABI。

`!mulberry.list_desc<T>` 是 descriptor-stage value。它表示一个可以穿过函数边界的
first-class list descriptor。

逻辑 descriptor 是：

```text
{ length: index, data: !mulberry.list_storage<T> }
```

后端 ABI descriptor 是：

```text
{ length: i64, data: ptr }
```

注意 `length` 永远是运行时元素个数，不是 element type。因此
`List<UInt64>`、`List<Float32>`、`List<Float32[?, ?]>` 的 descriptor 外壳都是：

```mlir
!llvm.struct<(i64, ptr)>
```

element type 的差异体现在 `data` 指向的连续 element ABI storage 里。由于 LLVM
dialect 使用 opaque pointer，`data` 字段打印出来统一是 `ptr`。

## Tensor ABI

Mulberry Tensor 是可写数组，rank 静态，dimension 可以是静态或动态。Tensor ABI
descriptor 当前按 rank 固定：

```text
TensorABI<T, rank> = {
  data: ptr<#llvm.address_space<0>>,
  sizes: array<rank x i64>,
  strides: array<rank x i64>
}
```

`data` 指向连续元素 storage；`sizes` 是每个维度的运行时大小；`strides` 是按
row-major layout 访问元素时的步长。这个 layout 借鉴 MLIR memref descriptor，
但没有直接把完整 memref ABI 暴露成 Mulberry 语言 ABI。

`tensor.desc_pack` 把本地 Tensor/memref 打包成这个 ABI descriptor。当前 lowering
会从 memref 中提取 data pointer、sizes 和 strides。

`tensor.desc_unpack` 做反方向：从 Tensor ABI descriptor 重建可被后续 Tensor /
`cherry_nn` op 使用的 memref view。当前实现使用：

- `llvm.extractvalue` 取出 descriptor 的 data、sizes 和 strides。
- `ptr.from_ptr` 把 ABI data pointer 变成 MLIR ptr dialect value。
- `memref.reinterpret_cast` 用 metadata 重建 ranked memref view。
- `memref.memory_space_cast` 回到普通 tensor lowering 使用的 memref type。

这里有一个重要约束：`desc_unpack` 不拥有 data 的生命周期。它只重建 view；data
必须由函数参数、返回 descriptor、heap list storage 或未来 runtime 保证仍然有效。

`!mulberry.tensor_handle` 和 `tensor.handle_from_desc` 仍然存在，但现在不是
`List<Tensor>` boundary 的主路径。当前主路径是：

```text
list.desc_get -> tensor.desc_unpack -> Tensor/memref value
```

`tensor_handle` 继续作为 future reconstruction/runtime handle 的实验性 IR，普通
`--lower-mulberry` 仍然对它 fail-fast。

## 字符串 ABI

Mulberry `String` 是源语言 builtin value，不是用户可见的 record。当前 lowering
把 `!mulberry.string` 转成一个 LLVM descriptor：

```text
StringABI = { length: i64, data: ptr }
```

`length` 是源码字符串的字节数，不包含结尾的 `\0`。`data` 指向只读字节数据。

字符串字面量会 materialize 成一个 internal constant LLVM global byte array，并在
末尾额外放一个 `\0`：

```text
"data/mnist.bin" -> global bytes "data/mnist.bin\0"
```

这个额外的 NUL 不属于 Mulberry `String` 的长度语义，只是 ABI/runtime 便利：
后续 `open/read/write/close` 这类 runtime API 如果走 C ABI，可以直接复用同一个
data pointer 作为 C string。

## List ABI

`List<T>` 的 ABI 外壳固定为：

```text
ListABI<T> = { length: i64, data: ptr }
```

`data` 指向连续的 element ABI storage：

```text
List<UInt64>        -> data: ptr<i64>
List<Float32[?, ?]> -> data: ptr<TensorABI<f32, 2>>
List<List<UInt64>>  -> data: ptr<ListABI<i64>>
```

当前实现已经支持 `List<Tensor>` 需要的 descriptor-backed storage：

```text
!mulberry.list_storage<!mulberry.tensor_desc<...>>
  -> ptr to contiguous Tensor ABI descriptors
```

普通函数内部的本地 `List<Tensor>` 仍可以用 memref-backed storage 作为 lowering
实现细节，以服务 for-loop 推理和 linalg lowering。这条本地 storage 路径不是函数
ABI。

## Boundary Preparation

`prepare-mulberry-boundaries` 是函数边界 rewrite pass。它负责把 source-level
`List<T>` 边界改写成 descriptor 边界，避免 MLIRGen 关心 ABI。

函数边界包括：

- `func.func` 参数。
- `func.func` 返回值。
- `func.call` 参数。
- `func.call` 返回值。
- `func.return` operand。

Tensor list 的 rewrite 形态是：

```mlir
func.func @f(%xs: !mulberry.list<!mulberry.tensor<?x?xf32>>)
```

变成：

```mlir
func.func @f(
    %xs: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>)
```

callee body 里的 list 操作同步改写：

```text
list.size %xs
  -> list.desc_length %xs -> arith.index_cast

list.get %xs[%i]
  -> list.desc_get %xs[%i] -> tensor.desc_unpack
```

scalar list 的形态更直接：

```text
List<UInt64>
  -> list_desc<i64>
  -> LLVM {length: i64, data: ptr}

list.get %xs[%i]
  -> list.desc_get %xs[%i]
  -> extract data pointer -> gep -> load scalar element
```

call site 同步插入 descriptor materialization：

```text
call @f(%list)
  -> %desc = mulberry.list.to_desc %list
  -> call @f(%desc)
```

这个 pass 只在能同步改写 callee signature、callee body、call site 和 return
operands 时动手。否则 fail-fast，避免生成半转换 IR。

## 当前正向支持

当前已经支持的核心路径：

- 函数内部 Tensor、List、`cherry_nn` lowering。
- Tensor 函数参数和返回值 lower 成 memref function boundary。
- 显式 `list_desc<TensorDesc>` 的 ABI projection 和 lowering。
- source-level scalar `List<T>` 参数和返回边界，其中 `T` 当前支持 `UInt64`、
  `Float32` 和 `index` 这类可直接 lower 的 scalar type。
- source-level `List<Tensor>` 参数边界，在所有 symbol use 都是 direct
  `func.call` 且 body use 只需要 `list.size/list.get` 时 rewrite。
- source-level `List<Tensor>` 返回边界，在 return value 直接来自本地
  `mulberry.list.create`，且 caller result use 可 rewrite 时支持。
- caller 侧拿到 `List<Tensor>` return 后，可以执行 `size(result)`，也可以执行
  `result[i]` 并把得到的 Tensor 继续交给 `cherry_nn` op。

源码级正向例子：

```cherry
fn make(): List<Float32[?, ?]> {
  const w: Float32[2, 2] = [[1.0, 0.0], [0.0, 1.0]];
  [w]
}

fn main(): UInt64 {
  const x: Float32[2, 1] = [[0.2], [0.8]];
  argmax(matmul(make()[0], x))
}
```

lowering 后的关键路径是：

```text
make() -> !llvm.struct<(i64, ptr)>
main(): call @make
main(): extract data pointer
main(): load TensorABI descriptor
main(): tensor.desc_unpack -> memref
main(): linalg.generic / scf / LLVM lowering
```

## Ownership

返回 `List<T>` 时，descriptor 的 `data` 不能指向 callee-local storage。scalar list
的本地 storage 是 memref，`List<Tensor>` 返回路径里的 descriptor-backed storage
当前是 LLVM stack alloca；二者都不能直接作为返回 descriptor 的 owned data。

因此 boundary rewrite 会插入：

```mlir
mulberry.list.escape_storage
```

`escape_storage` 本身只是 ownership marker，不直接改变 storage。真正的 heap ABI
data 在 `mulberry.list.desc_pack` lowering 里 materialize：

1. 按 element ABI type 计算每个元素的大小。
2. `mulberry_boehm_malloc(length * sizeof(element ABI))` 分配 Boehm-managed
   data buffer。
3. 从本地 storage 逐个复制 element 到 heap buffer。
4. 生成 `{ length: i64, data: ptr }` descriptor。

当前采用明确标注的临时策略：

- 支持 element ABI 可 lower 的返回 list，包括当前的 scalar list 和
  `List<TensorDesc>`。
- lowering 使用 Boehm GC 分配返回 descriptor 的 heap data。
- 不再生成 `mulberry.list.dealloc` 或 `mulberry.list.desc_dealloc`。实际回收交给
  Boehm GC。

本地 `List<T>` storage 当前不再做自动 dealloc。旧实现会在 `--lower-mulberry`
里分析本地 list literal 的 use，再插入内部 `mulberry.list.dealloc`，这让 lowering
pass 过早承担了 lifetime 分析。Boehm-only 方向接受当前正向 JIT 路径里的少量
local list storage 不回收，后续如果需要精确 lifetime，再单独设计。

这不是最终 ownership 方案，只是为了避免返回悬空 pointer，同时让当前正向路径可以
完整 lower。当前策略故意很窄：callee 不释放参数，callee 不释放返回 operand；
返回 descriptor 的 data lifetime 由 Boehm GC 管理。

## 仍然限制

以下场景仍然不应该硬凑：

- 任意来源的 `List<T>` value 直接 materialize 成 descriptor。当前 `list.to_desc`
  仍然主要支持本地 `list.create`。
- nested list、record field 里的 list 穿过函数边界。
- external function 的 `List<T>` boundary。
- indirect symbol use、function pointer 或跨 module boundary。
- 完整 ownership，包括 arena、precise GC、move/copy 语义。
- `tensor_handle` lowering。
- unsupported source patterns 的完整 LLVM/JIT 支持。
- object generation。

这些限制是有意保留的。缺少底层 ABI 或 ownership 设计时，继续 fail-fast 比生成
表面可用、实际语义不成立的 IR 更好。

## 借鉴 `std::vector<T>`

Mulberry 不需要完整 C++ template 语言特性，但 `List<T>` 的 lowering 可以借鉴
monomorphization 思想：

- `List<T>` 在语义层是泛型容器。
- `List<UInt64>`、`List<Float32[?, ?]>`、`List<List<UInt64>>` 在 lowering/ABI
  层是不同 concrete type。
- 每个 concrete `List<T>` 都递归计算自己的 element ABI、descriptor layout 和
  load/store lowering。

可以把当前 `List<T>` 理解成语言内建的 `std::vector<T>` 简化版本：

```text
用户层：List<T>
语义层：泛型容器类型
lowering 层：按 T 计算 element ABI
ABI 层：{ length, data: ptr<element ABI> }
```

当前不引入 `capacity`、allocator、异常安全、move/copy 语义或用户级 generic
function。等 training 需要更完整的容器语义时，再单独设计。

## 下一步

R3.8.88 已完成：Nielsen for-loop 推理脚本可以通过 `--dump=lowered-mlir`，
lowered IR 使用 `scf`、`memref`、`linalg`、`arith` 和 `math`，不再残留
`mulberry` 或 `cherry_nn` op。

R3.8.89 已完成：scalar `List<T>` 函数边界可以通过 descriptor ABI lowering，
当前覆盖 `UInt64`、`Float32` 和 `index` 这类 scalar element。

R3.8.90 已完成：本地 list storage 曾支持内部显式 dealloc，`List<Tensor>` 返回
descriptor 的 heap data 曾使用 caller-side `free`。这只是历史阶段，不是完整
runtime ownership。

R3.8.91 已完成：返回的 scalar `List<T>` 和 `List<Tensor>` 统一走 escaping
descriptor storage，`desc_pack` 会复制到 Boehm-managed ABI data。

R3.8.92 已完成：当前支持的正向路径可以继续 lower 到 LLVM dialect / LLVM IR，
并通过 JIT 执行。Nielsen for-loop 推理脚本可以直接运行，输出 `7`。

G3 已完成：接受 Boehm-only 方向后，删除 local `List<T>` auto dealloc 插入逻辑。
G4 已完成：删除 `mulberry.list.dealloc` 和 `mulberry.list.desc_dealloc`。List
descriptor data 的生命周期由 Boehm GC 管理，不再保留显式 cleanup op。

R3.8.93：暂时不扩展 object generation；先增加 training 需要的 `matsub`、
`hadamard`、`sigmoidPrime` 等 `cherry_nn` ops，并保持 for-loop 推理 JIT 路径稳定。

R3.8.94：等 training script 需要时，再补更完整的 ownership / dealloc / runtime
策略，而不是提前为负向场景堆 workaround。
