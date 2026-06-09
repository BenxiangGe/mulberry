# Mulberry Lowering

本文档记录当前 lowering 边界，以及 `List<T>` ABI 的设计方向。文档刻意保持
简短：核心目标是防止 MLIRGen 和 lowering 再次滑向偶然可用的后端 workaround。

## 当前边界

`--lower-mulberry` 把高层 Mulberry IR lower 到 storage-level MLIR：

- `mulberry.tensor` 变成 `memref`。
- `mulberry.list` 在支持的场景下变成本地 list storage。
- `cherry_nn` operations 变成 `linalg`、`math`、`arith` 和 `memref`。
- scalar 和 record storage 在映射直接的地方可以使用 LLVM dialect。

这不是最终的 Mulberry-to-LLVM ABI lowering。`-dump=mlir-llvm`、
`-dump=llvm`、JIT 和 object generation 都保持关闭，直到这个 pipeline 被单独
设计清楚。

## List 分层

`!mulberry.list<T>` 是源语言层面的 semantic value。MLIRGen 应该为用户可见的
`List<T>` value 生成这个类型。

`!mulberry.list_storage<T>` 是 storage-level lowering 过程中产生的本地连续元素
存储。它是实现细节，不能变成 function-boundary ABI。

`!mulberry.list_desc<T>` 是未来 list boundary 使用的 descriptor-stage value。
它表示一个 first-class list value，字段严格只有：

- `length`：运行时元素数量。
- `data`：本地 storage 或后端 data handle。

本地的 `list_desc` pack/projection 可以在 `--lower-mulberry` 内部 fold 掉。
传递或返回 `list_desc` 仍然是非法的，直到后端 ABI 被设计清楚。

## Descriptor Conversion 设计

`list_desc` 需要分成两个概念来看：

- 逻辑 descriptor：`{length: index, data: !mulberry.list_storage<T>}`。
- 后端 ABI descriptor：`{length: backend index, data: ptr<element ABI>}`。

当前 `!mulberry.list_desc<T>` 表达的是逻辑 descriptor。它适合描述
`List<T>` 作为一个 first-class value 的语义，但还不是 LLVM-compatible ABI。
原因是 `data` 现在仍然可能是 `list_storage`，而 `list_storage` 当前会 lower 到
`memref<?x...>`。`memref` value 不能直接放进 LLVM struct 字段里。

后端 ABI descriptor 的 `data` 字段应该是一个普通后端指针，指向连续的元素 ABI
storage。元素 ABI 类型按元素语义决定：

- `List<UInt64>`：`data` 指向连续的 `i64` 元素。
- `List<Float32[?, ?]>`：`data` 指向连续的 Tensor ABI descriptor。
- `List<List<UInt64>>`：`data` 指向连续的 List ABI descriptor。
- `List<struct T>`：需要等 record storage 可以安全嵌入 descriptor 后再支持。

注意 `{length, data}` 里的 `length` 不是 list element type。比如
`List<UInt64>` 和 `List<Float32>` 的 descriptor type 都是：

```mlir
!llvm.struct<(i64, ptr)>
```

其中第一个字段 `i64` 永远表示运行时长度。`UInt64`、`Float32` 等 element type
的差异体现在 `data` 指向的连续 storage 里，而不是体现在 descriptor struct 的
第一个字段里。由于 LLVM dialect 当前使用 opaque pointer，`data` 字段打印出来是
统一的 `ptr`。

因此，最终 conversion 不应该是 `list_desc` 直接变成包含 `memref` 字段的
`LLVMStructType`。正确顺序是：

1. 先决定元素类型的 ABI 表示。
2. 把本地 storage handle finalization 成 backend-legal pointer 或 descriptor。
3. 再把 `{length, data}` 组成后端 record。
4. 最后才允许这个 record 穿过 `func.func`、`func.call` 和 `func.return`。

这里的关键设计约束是：`List<T>` 在 Mulberry lowering 层仍然是一个 descriptor
value。低级 LLVM ABI 最后是否拆成多个参数、是否按值传 struct、是否传指针，是
更后面的 backend ABI 决策，不能反向污染 MLIRGen。

R3.8.39 当前只加入了 type-level helper，用来描述可行的后端 ABI record 形状。
它没有注册到 `MulberryTypeConverter`。原因是：类型能转换并不表示 value 已经能
materialize。`list_storage` 变成后端 pointer 之前，`List<T>` function boundary
仍然必须 fail-fast。

R3.8.40 加入了 value-level helper，用来把已经 lower 后的 local list storage
提取成 backend data pointer，并把 `{length, data}` 组装成 backend ABI record。
这些 helper 仍然没有接入 function/call/return conversion，因为它们只处理
storage 到 ABI value 的局部 materialization，还没有解决 ownership、生命周期、
return ABI 和 nested descriptor 的完整规则。

R3.8.41 加入了一个显式的局部测试路径：
`mulberry.list.desc_to_abi`。它要求输入来自本地 `mulberry.list.desc_pack`，
然后把这个 logical descriptor materialize 成后端 ABI record：
`!llvm.struct<(i64, ptr)>`。当前只支持 scalar element list，例如
`List<UInt64>`。

这里有一个刻意保留的限制：`desc_to_abi` 不是 function-boundary lowering。
`--lower-mulberry` 在转换前会显式拒绝带有 `List<T>`、`list_storage<T>` 或
`list_desc<T>` 的 `func.func`、`func.call`、`func.return`。这样可以测试 ABI
record 的局部组装逻辑，又不会让普通函数边界误以为已经有完整 List ABI。

后面真正启用函数参数 lowering 时，`ListABILayout.descriptorType` 会是函数签名里
使用的后端类型。比如：

```mlir
func.func @f(%xs: !mulberry.list<i64>)
```

会被转换为：

```mlir
func.func @f(%xs: !llvm.struct<(i64, ptr)>)
```

其中 `!llvm.struct<(i64, ptr)>` 就是 `ListABILayout.descriptorType`。函数体里的
`size(xs)`、`xs[i]` 等操作再从这个 descriptor value 里 extract `length` 和
`data` 字段。当前还没有启用这一步，只是先把 descriptor type 和局部 value
materialization 路径验证清楚。

R3.8.42 明确保持 `List<Tensor>` 的 ABI descriptor materialization fail-fast。
函数内部的 `List<Tensor>` 仍然可以 lower 到本地 storage：

```mlir
memref<?xmemref<...>>
```

但这只是 storage-level lowering，不是稳定函数 ABI。`List<Tensor>` 作为函数参数、
返回值、record field 或嵌套 list element 时，需要先有 Tensor ABI descriptor。
没有这个设计前，不能把 `List<Tensor>` 直接 materialize 成 `{length, ptr}`，
否则 `ptr` 指向什么会变成隐式 workaround。

R3.8.43 的设计结论是：`List<Tensor>` 的 element ABI 应该优先使用 Tensor ABI
descriptor，而不是裸 `memref` handle。原因是 Tensor descriptor 是普通后端 value，
可以作为 `List<Tensor>` 的连续 element storage，也可以递归放进 record field 或
其它 descriptor。裸 `memref` handle 仍然是 storage-level value，直接把它塞进
`List<Tensor>` ABI 会重新引入 “memref value 如何嵌入 LLVM aggregate” 的问题。

因此后续正确顺序应该是：

1. 设计 Tensor ABI descriptor type，例如固定或动态 rank 的 `{data, sizes, ...}`。
2. 增加本地 Tensor descriptor materialization，验证 tensor value 可以变成后端
   legal descriptor value。
3. 再让 `List<Tensor>` 的 `data` 指向连续的 Tensor ABI descriptor storage。
4. 最后才允许 `List<Tensor>` 穿过 `func.func`、`func.call`、`func.return`。

R3.8.44 的 Tensor ABI descriptor 方向：

Mulberry Tensor 的 rank 是静态的，只有每个维度的 size 可以是静态或动态。因此
Tensor ABI descriptor 不需要一开始就做成动态 rank object。每个 concrete
`Tensor<T, rank>` 可以有自己的 descriptor type：

```text
TensorABI<T, rank> = {
  data: ptr,
  sizes:  index[rank],
  strides: index[rank]
}
```

这里的 `data` 指向连续元素 storage；`sizes` 是每个维度的运行时大小；`strides`
描述按 row-major layout 访问元素时的步长。这个 layout 借鉴 MLIR memref
descriptor，但不直接把完整 memref descriptor 暴露成 Mulberry ABI。原因是 memref
descriptor 还包含 allocated pointer、aligned pointer、offset 等 ownership 和
view 相关细节；这些是 backend/runtime lowering 需要的信息，不应该过早变成
Mulberry 语言 ABI。

例如：

```text
Float32[2, 3]  -> { data: ptr, sizes: [2, 3], strides: [3, 1] }
Float32[?, ?]  -> { data: ptr, sizes: [m, n], strides: [n, 1] }
```

`List<Tensor>` 的 element ABI 应该使用这个 Tensor descriptor。这样
`List<Float32[?, ?]>` 的 list descriptor 仍然是 `{length, data}`，其中 `data`
指向连续的 Tensor ABI descriptor 数组：

```text
List<Float32[?, ?]> -> {
  length: i64,
  data: ptr<TensorABI<f32, 2>>
}
```

这个设计仍然需要后续实现两个步骤：先增加本地 Tensor descriptor materialization，
再让 `List<Tensor>` 存储连续 Tensor descriptor。没有这两个步骤前，当前的
`List<Tensor>` ABI lowering 继续 fail-fast。

R3.8.45 增加了 descriptor-stage Tensor IR 骨架：

- `!mulberry.tensor_desc<...>` 表示 Tensor ABI descriptor value。
- `mulberry.tensor.desc_pack` 表示从 Tensor value 打包出逻辑 descriptor。
- `mulberry.tensor.desc_to_abi` 表示把 Tensor descriptor materialize 成后端 ABI
  record。

这一步只建立 IR 语义和 verifier，不接入 `--lower-mulberry` conversion。原因是
Tensor descriptor materialization 需要明确如何从 memref storage 中提取 data
pointer、sizes 和 strides；在这之前不能让 `List<Tensor>` ABI 通过裸 memref
handle 偷跑。

R3.8.46 增加了局部 Tensor descriptor materialization：
`tensor.desc_pack` 在 `--lower-mulberry` 中负责把已经 lower 后的 memref storage
组装成 `{data, sizes, strides}` 后端 record，`tensor.desc_to_abi` 只作为局部 marker
被替换掉。`tensor_desc` 作为函数参数或返回值仍然 fail-fast；这一步只验证本地
value-level ABI record 可以干净生成。

R3.8.47 验证了一个重要负例：不能把 `List<Tensor>` 的 ABI storage 简单改成
`memref<?x TensorABI>`。MLIR `memref` 的 element type 有限制，
`memref<?x !llvm.struct<...>>` 和
`memref<?x !mulberry.tensor_desc<...>>` 都不是合法的 memref element type。

因此当前函数内部的 `List<Tensor>` 仍然保留为 storage-level lowering：

```mlir
memref<?xmemref<...>>
```

这条路径只服务于本地 for-loop 推理和 `linalg` lowering，不能成为函数边界 ABI。
真正支持 `List<Tensor>` boundary 前，需要先设计独立的 list runtime storage，
例如用 backend pointer 指向连续 descriptor storage，并提供明确的 descriptor
load/store/reconstruct 规则。

R3.8.48 增加了这个独立 storage 的第一块：显式的
`list_storage<TensorDesc>` 不再通过 `memref` lowering，而是 lower 成一个
LLVM pointer，指向连续的 Tensor ABI descriptor 元素。

```text
!mulberry.list_storage<!mulberry.tensor_desc<...>>
  -> ptr to TensorABI descriptor array
```

这一步只支持显式 descriptor storage，不改变当前高层 `List<Tensor>` codegen。
也就是说，函数内部推理代码里的 `List<Tensor>` 仍然走
`memref<?xmemref<...>>`，而 ABI 方向的 descriptor storage 已经有了单独路径。
后面要做的是让 boundary lowering 在需要时把 Tensor value pack 成
`tensor_desc`，再放进这条 descriptor-backed list storage。

R3.8.49 把这条 descriptor-backed storage 接到本地 `list_desc` ABI
materialization。对于 scalar list，`list.desc_pack` 仍然从 `memref` storage
提取 data pointer；对于 `list_storage<TensorDesc>`，storage 已经是后端 pointer，
所以 `{length, data}` 里的 `data` 直接使用这个 pointer。

这一步仍然不是函数边界支持。它只证明：如果一个 list 的 data 已经是连续 Tensor
ABI descriptor storage，那么本地 `list.desc_to_abi` 可以干净地产生
`!llvm.struct<(i64, ptr)>`。

R3.8.50 增加了 descriptor-backed `list_desc` 的 ABI projection 路径。
对于 `list_desc<TensorDesc>`，`desc_pack` 会先 lower 成后端 ABI record：

```mlir
!llvm.struct<(i64, ptr)>
```

随后 `list.desc_length` 和 `list.desc_data` 分别 lower 成：

```text
llvm.extractvalue desc[0]  // length
llvm.extractvalue desc[1]  // data pointer
```

这一步仍然不打开 `List<T>` 函数边界。它只验证：当一个 `List<T>` descriptor
已经是后端 ABI record 时，函数体里需要的 `length` 和 `data` 字段可以干净地
投影出来。scalar `List<T>` 当前仍然优先走本地 `desc_pack` fold，因为它的 data
字段来自 memref-backed storage；如果从 `{length, ptr}` 反向重建
`memref<?xT>`，就会重新引入不明确的 memref descriptor reconstruction 规则。

R3.8.51 打开了一个窄的函数参数路径：显式的
`list_desc<TensorDesc>` 可以作为 `func.func` 参数和 `func.call` operand lower 成
List ABI descriptor：

```mlir
!mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
  -> !llvm.struct<(i64, ptr)>
```

这条路径只允许调用方已经把 tensor pack 成 Tensor ABI descriptor，并把这些
descriptor 放进 pointer-backed list storage。callee 侧通过 R3.8.50 的
`extractvalue` projection 读取 `length` 和 `data`。

这一步仍然不支持：

- `List<T>` source-level function boundary。
- scalar `list_desc<T>` 参数。
- `list_desc<T>` 返回值。
- `tensor_desc` 直接作为函数参数或返回值。

原因是返回值还需要明确 ownership/lifetime；scalar list 的 `{length, ptr}` 到
当前 memref-backed storage 的 reconstruction 规则也还没有设计。当前只验证
`List<Tensor>` ABI 所需的最小 descriptor 参数通路。

R3.8.52 增加了显式 bridge op：

```mlir
mulberry.list.to_desc %list
  : !mulberry.list<!mulberry.tensor<?x?xf32>>
    -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
```

这个 op 表达的是：把 high-level `List<Tensor>` materialize 成 ABI-ready
`list_desc<TensorDesc>`。它不是普通 type cast，也不是 TypeConverter 的隐式规则。
当前 lowering 只支持 `%list` 来自本地 `mulberry.list.create`：

1. 为每个 tensor element 生成 `tensor.desc_pack`。
2. 分配 pointer-backed `list_storage<TensorDesc>`。
3. 把 Tensor ABI descriptor 写入该 storage。
4. 再生成 `list.desc_pack` 得到 `{length, data}` List ABI descriptor。

这个限制是有意保留的。任意 `List<Tensor>` value 可能来自函数参数、record field
或其它存储位置；这些场景需要先有明确的 descriptor reconstruction 和 ownership
规则，不能在 lowering 里偷偷假设。

R3.8.53 明确了 source-level `List<Tensor>` 函数参数暂时不能只靠 call-site bridge
打开。原因是函数边界有两侧：

```mlir
func.func @f(%xs: !mulberry.list<!mulberry.tensor<?x?xf32>>)
call @f(%list) : (!mulberry.list<!mulberry.tensor<?x?xf32>>) -> ...
```

如果只在 caller 侧把 `%list` 改成 `mulberry.list.to_desc`，callee 的函数签名仍然是
`!mulberry.list<Tensor>`，类型对不上。如果同时把 callee 参数改成
`!mulberry.list_desc<!mulberry.tensor_desc<...>>`，函数体里的 `xs[i]` 又不再能直接
产生 high-level Tensor value：它只能从 `{length, data}` 里读出一个 Tensor ABI
descriptor。这个 descriptor 后续要么继续作为 ABI value 传播，要么需要一个明确的
`TensorDesc -> Tensor/memref` reconstruction 规则。

因此下一步不应该在 MLIRGen 里增加 “遇到函数参数就特殊改类型” 的模式。更干净的
路径是增加一个独立的 boundary preparation 阶段：

1. MLIRGen 继续生成源语言语义最直接的 `!mulberry.list<Tensor>`。
2. boundary preparation 只处理明确支持的函数边界，把参数类型改成
   `!mulberry.list_desc<TensorDesc>`，并在 call site 显式插入 `list.to_desc`。
3. callee body 中的 `size(xs)` 改成 descriptor length projection。
4. callee body 中的 `xs[i]` 先改成 descriptor element load。
5. 只有在定义清楚 `TensorDesc -> Tensor/memref` reconstruction 后，才能让这个
   element 被 `cherry_nn` ops 当作普通 Tensor 使用。

这一步的结论是：当前显式 MLIR `list_desc<TensorDesc>` 参数 lowering 是正确的
底层能力；source-level `List<Tensor>` 参数 lowering 还缺一个独立 preparation pass
和 Tensor descriptor reconstruction 规则。不能把这两件事塞进 call lowering 里
硬凑。

R3.8.54 增加了 `prepare-mulberry-boundaries` pass 骨架。这个 pass 暂时只识别
source-level `List<Tensor>` 函数边界并 fail-fast，不做实际转换，也不接入
`--dump=lowered-mlir` 默认 pipeline。它的作用是把未来 boundary preparation 的入口
固定下来，避免把函数签名 rewrite、call-site bridge、callee body rewrite 和
descriptor reconstruction 混进 `MLIRGen` 或 `lower-mulberry`。

R3.8.55 增加了 `mulberry.tensor.desc_unpack` 作为 callee body rewrite 需要的
显式 IR 入口：

```mlir
mulberry.tensor.desc_unpack %desc
    : !mulberry.tensor_desc<?x?xf32> -> !mulberry.tensor<?x?xf32>
```

它和 `tensor.desc_pack` 对称，用来表达从 Tensor descriptor 重新得到 Tensor value。
当前只定义 IR 语义和 verifier，不实现 lowering。原因是 descriptor reconstruction
涉及 allocation、ownership 和从 ABI record 恢复 memref/tensor handle 的规则；
这些不能隐式塞进 `lower-mulberry`。后续 boundary preparation 可以先把 `xs[i]`
rewrite 成 `list.desc_data -> list.load -> tensor.desc_unpack`，然后再单独实现
`desc_unpack` 的 lowering。

R3.8.56 增加了 `mulberry.list.desc_get`：

```mlir
mulberry.list.desc_get %desc[%i]
    : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
      -> !mulberry.tensor_desc<?x?xf32>
```

它是 callee body rewrite 的 descriptor-level list indexing 入口。lowering 前它会
展开为 `list.desc_data + list.load`，因此不增加新的 ABI 语义。未来
`prepare-mulberry-boundaries` 可以把 source-level `xs[i]` rewrite 成：

```text
list.desc_get -> tensor.desc_unpack
```

这一步仍然只处理显式 descriptor IR，不打开 source-level `List<Tensor>` 函数边界。

R3.8.57 明确保持 `tensor.desc_unpack` lowering fail-fast。当前 Tensor ABI
descriptor 是：

```text
{ data: ptr, sizes: array<rank x i64>, strides: array<rank x i64> }
```

这个 layout 足够从 memref pack 出后端 ABI value，但还不足以在 `lower-mulberry`
里重新构造高层 memref/Tensor value。MLIR 的 `memref.reinterpret_cast` 需要一个已有的
memref base source；它不能从裸 LLVM `ptr` 直接生成 memref descriptor。因此
`desc_unpack` 需要后续单独设计 Tensor ABI reconstruction/runtime handle，不能在这里
用 `ptr -> memref` 的隐式 workaround 硬凑。

R3.8.58 补充了更贴近真实 callee rewrite 的 fail-fast 覆盖：

```mlir
%loaded = mulberry.list.desc_get %desc[%i]
    : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
      -> !mulberry.tensor_desc<?x?xf32>
%tensor = mulberry.tensor.desc_unpack %loaded
    : !mulberry.tensor_desc<?x?xf32> -> !mulberry.tensor<?x?xf32>
```

这条路径表达的是：函数参数已经变成 `list_desc<TensorDesc>` 后，函数体里
`xs[i]` 先得到 Tensor descriptor，然后再尝试恢复成普通 Tensor value。当前仍然
必须 fail-fast。原因是 `list.desc_get` 只能读出一个后端 descriptor value；
它不能自动证明这个 descriptor 的 data pointer、shape、stride 和 lifetime 足够
重建一个合法的 Mulberry Tensor/memref。

因此现在允许的是：

```text
list.desc_get -> tensor.desc_to_abi
```

也就是把 Tensor descriptor 继续作为 ABI record 传播。现在不允许的是：

```text
list.desc_get -> tensor.desc_unpack -> tensor/cherry_nn ops
```

后者需要先设计真正的 Tensor handle 或 runtime reconstruction 规则。这个测试的
目的就是防止后续 boundary preparation 把 `xs[i]` 误改成“看起来能 lower”的
Tensor value。

R3.8.59 对 `TensorDesc -> Tensor` reconstruction 先确定设计边界，不在
`lower-mulberry` 里硬实现。

当前 `TensorDesc` 的 ABI record 是：

```text
{ data: ptr, sizes: array<rank x i64>, strides: array<rank x i64> }
```

它适合做函数边界 ABI value，也适合保存到 `List<TensorDesc>` 的 contiguous storage
里。但它还不是一个完整的 Mulberry Tensor handle。缺失的信息包括：

- `data` 的 ownership/lifetime 由谁负责。
- `data` 是否一定指向 contiguous row-major storage。
- descriptor 是否允许 non-zero offset。
- descriptor 是否需要 allocated pointer / aligned pointer 的区别。
- descriptor 重新进入 Tensor/cherry_nn ops 时，应该恢复成 memref，还是保持成
  另一种 runtime handle。

因此不能把 `tensor.desc_unpack` 简单 lowering 成：

```text
ptr + sizes + strides -> memref
```

这会伪造一个 MLIR memref source。MLIR memref 的 lowering/finalization 本身有一套
descriptor 规则；从裸 LLVM `ptr` 逆向构造 memref 会把 ownership、alignment、
offset 和 lifetime 都藏起来，后面一定会变成 workaround。

更干净的方向是把问题拆成两层：

1. `TensorDesc` 继续表示 ABI descriptor，也就是跨函数边界和存进
   `List<TensorDesc>` 的后端 record。
2. 另设 Tensor handle/reconstruction 设计，用一个明确 IR 类型或 runtime API 表达
   “这个 descriptor 重新成为可被 tensor/cherry_nn ops 使用的 Tensor value”。

候选路线：

- `!mulberry.tensor_handle<...>`：descriptor unpack 得到 handle，后续 tensor/cherry_nn
  ops 改为接受 handle 或在 lowering 前统一把 Tensor value 表示成 handle。
- runtime reconstruction API：`tensor.desc_unpack` lowering 成 runtime call，runtime
  返回一个稳定 Tensor handle。
- 完整 memref descriptor ABI：把 ABI record 改成能直接描述 MLIR memref 的完整
  descriptor，再设计从该 descriptor 到后续 lowering 的合法通路。

当前倾向第一条或第二条。第三条看似直接，但很容易重新把 MLIR memref ABI 暴露成
Mulberry 语言 ABI，而且对 dynamic tensor、list storage 和函数返回 ownership 都不够
清爽。

所以现在的规则保持不变：

```text
允许：Tensor -> tensor.desc_pack -> tensor.desc_to_abi
允许：List<Tensor> -> list.to_desc -> list_desc<TensorDesc> -> list.desc_get
禁止：tensor.desc_unpack lowering
```

下一步如果要真正打开 `tensor.desc_unpack`，应该先增加一个最小 Tensor handle IR
设计，而不是在 `LowerMulberry.cpp` 里直接写 `ptr -> memref`。

R3.8.60 增加最小 `TensorHandle` IR 草图：

```mlir
!mulberry.tensor_handle<?x?xf32>

%handle = mulberry.tensor.handle_from_desc %desc
    : !mulberry.tensor_desc<?x?xf32>
      -> !mulberry.tensor_handle<?x?xf32>
```

`TensorHandle` 是 reconstruction-stage value。它表达的是：“这个 descriptor 已经进入
可恢复 Tensor 的阶段”，但它还不是 `!mulberry.tensor`，也不是 MLIR `memref`。这样做
的目的，是把三种语义拆开：

- `!mulberry.tensor`：高层 Mulberry Tensor value，当前本地 lowering 到 memref。
- `!mulberry.tensor_desc`：函数边界和 `List<Tensor>` storage 使用的 ABI descriptor。
- `!mulberry.tensor_handle`：未来 runtime 或 reconstruction pass 返回的 Tensor handle。

当前只允许 parse/verify `tensor_handle` 和 `tensor.handle_from_desc`。`lower-mulberry`
仍然 fail-fast，因为还没有 runtime handle layout、ownership 和具体 tensor/cherry_nn
op 接受 handle 的规则。

R3.8.61 明确 `tensor_handle` 目前不能被普通 Tensor op 或 `cherry_nn` op 消费：

```text
禁止：tensor.handle_from_desc -> mulberry.tensor.dim/load/store
禁止：tensor.handle_from_desc -> cherry_nn.matmul/sigmoid/argmax
```

这不是能力缺失的偶然结果，而是有意保留的语义边界。`tensor_handle` 不是
`!mulberry.tensor` 的子类型，也不是一个可以隐式 cast 的 memref。后续如果要让
`List<Tensor>` 函数参数里的 `xs[i]` 参与 NN 计算，应该增加显式 bridge/pass，例如：

```text
list.desc_get -> tensor.handle_from_desc -> tensor.handle_to_value/runtime op
```

或者整体把 `tensor/cherry_nn` lowering 改为 handle-aware。不能直接把
`CherryNN_AnyTensorType` 放宽成 `TensorType | TensorHandleType`，否则会把“本地
Tensor value”和“跨边界重建中的 handle”混成同一种语义。

R3.8.62 补齐 `prepare-mulberry-boundaries` 的边界 fail-fast 覆盖。这个 pass 未来要
处理的是函数边界，不只是 `func.func` 的签名：

```text
func.func 参数
func.func 返回类型
func.call 参数
func.call 返回值
func.return 返回值
```

这一步仍然不开始 rewrite。原因是真正打开 source-level `List<Tensor>` 参数需要同时改
三类东西：callee signature、call site 和 callee body。如果只改 call site，会和
callee signature 类型不匹配；如果只改 signature，body 里的 `xs[i]` 又需要
`list.desc_get -> tensor.handle_from_desc` 以及后续 Tensor handle 消费模型。先把所有
边界点的 fail-fast 测清楚，后面做 rewrite 时才不会漏掉某一侧。

R3.8.63 明确不做 call-site-only rewrite。也就是说，`prepare-mulberry-boundaries`
当前不会把：

```mlir
call @f(%xs) : (!mulberry.list<!mulberry.tensor<?x?xf32>>) -> ()
```

单独改成：

```mlir
%desc = mulberry.list.to_desc %xs
call @f(%desc) : (!mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>) -> ()
```

原因是这只改了 caller，没有改 callee signature，也没有改 callee body。生成这种半改
IR 会让函数声明和 call type 不一致，或者让 body 里的 `xs[i]` 没有合法语义。

当前 `prepare-mulberry-boundaries` 只接受已经显式处于 descriptor boundary 的 IR：

```mlir
func.func @f(%xs: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>)
```

真正的 source-level `List<Tensor>` boundary rewrite 必须作为一个完整步骤处理：

```text
callee signature rewrite
call-site list.to_desc insertion
callee body xs.size / xs[i] rewrite
Tensor handle reconstruction policy
```

这四件事要一起设计，不能先在 call-site 塞一个局部 workaround。

R3.8.64 固化完整 boundary rewrite 的目标 IR 形态。source-level 代码大致是：

```mlir
func.func @f(%xs: !mulberry.list<!mulberry.tensor<?x?xf32>>) {
  %size = mulberry.list.size %xs : !mulberry.list<!mulberry.tensor<?x?xf32>>
  %tensor = mulberry.list.get %xs[%i]
      : !mulberry.list<!mulberry.tensor<?x?xf32>> -> !mulberry.tensor<?x?xf32>
}
```

完整 boundary preparation 后，callee 侧目标形态应该是：

```mlir
func.func @f(%xs: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>) {
  %size = mulberry.list.desc_length %xs
      : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
  %desc = mulberry.list.desc_get %xs[%i]
      : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
        -> !mulberry.tensor_desc<?x?xf32>
  %handle = mulberry.tensor.handle_from_desc %desc
      : !mulberry.tensor_desc<?x?xf32>
        -> !mulberry.tensor_handle<?x?xf32>
}
```

caller 侧目标形态是：

```mlir
%desc = mulberry.list.to_desc %xs
    : !mulberry.list<!mulberry.tensor<?x?xf32>>
      -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
call @f(%desc)
    : (!mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>) -> ()
```

这里仍然不产生可被 `cherry_nn` 消费的 Tensor value。`tensor_handle` 只是重建阶段的
中间 value。后续要么增加 `tensor.handle_to_value/runtime op`，要么让
`tensor/cherry_nn` lowering 变成 handle-aware。这个目标 IR 测试的作用，是给未来
`prepare-mulberry-boundaries` 的 rewrite 提供稳定形状，而不是提前打开 lowering。

R3.8.65 明确不做 callee-signature-only rewrite。把：

```mlir
func.func @f(%xs: !mulberry.list<!mulberry.tensor<?x?xf32>>)
```

单独改成：

```mlir
func.func @f(%xs: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>)
```

仍然是不完整的。原因是 entry block argument 的所有 use 都还是 source-level list
语义：

```text
mulberry.list.size %xs
mulberry.list.get %xs[%i]
```

这些 use 必须同步 rewrite 成：

```text
mulberry.list.desc_length %xs
mulberry.list.desc_get %xs[%i] -> mulberry.tensor.handle_from_desc
```

同时 caller 侧也必须插入 `mulberry.list.to_desc`，否则 call op 和 callee signature
类型不匹配。因此后续真正实现时，第一块不应该叫 “callee signature rewrite”，而应该是
“单函数 boundary rewrite skeleton”：先支持没有 caller 的 private function，且只处理
`size/get` 两种 body use；再扩展到 call-site rewrite。

R3.8.66 打开了第一块非常窄的 skeleton：只处理没有 caller 的 private function
参数，并且 body 里这个参数只能被 `mulberry.list.size` 直接使用。rewrite 后：

```mlir
func.func private @f(%xs: !mulberry.list<!mulberry.tensor<?x?xf32>>) -> i64 {
  %size = mulberry.list.size %xs
      : !mulberry.list<!mulberry.tensor<?x?xf32>>
  return %size : i64
}
```

变成：

```mlir
func.func private @f(%xs: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>) -> i64 {
  %length = mulberry.list.desc_length %xs
      : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
  %size = arith.index_cast %length : index to i64
  return %size : i64
}
```

这一步仍然不支持 call site，也不支持 `list.get`。`list.get` 需要继续变成
`list.desc_get -> tensor.handle_from_desc`，而当前 `tensor_handle` 还不能被普通
Tensor op 或 `cherry_nn` op 消费。如果现在强行改 `list.get`，就会再次制造
“看起来改了边界，实际后续没有语义”的半转换 IR。

R3.8.67 把 `list.get` 的目标 IR 打开了一小步，但只允许结果没有 users 的情况：

```mlir
%tensor = mulberry.list.get %xs[%i]
    : !mulberry.list<!mulberry.tensor<?x?xf32>>
      -> !mulberry.tensor<?x?xf32>
```

可以 rewrite 成：

```mlir
%desc = mulberry.list.desc_get %xs[%i]
    : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
      -> !mulberry.tensor_desc<?x?xf32>
%handle = mulberry.tensor.handle_from_desc %desc
    : !mulberry.tensor_desc<?x?xf32>
      -> !mulberry.tensor_handle<?x?xf32>
```

但是如果原来的 `%tensor` 有任何 users，例如：

```mlir
%dim = mulberry.tensor.dim %tensor, %i : !mulberry.tensor<?x?xf32>
```

仍然必须 fail-fast。原因是 `tensor_handle` 不是 `tensor`，不能偷偷替换给
`tensor.dim`、`tensor.load/store` 或 `cherry_nn`。后续真正打开这条路径时，需要先
决定是增加 `tensor.handle_to_value`，还是让 Tensor/cherry_nn lowering 直接理解
handle。

R3.8.68 把 private function 的 call-site 参数 rewrite 接了起来，但仍然只处理
没有 `List<Tensor>` 返回值、且 call op 本身没有 results 的场景。形态是：

```mlir
%list = mulberry.list.create(%tensor)
    : (!mulberry.tensor<?x?xf32>)
      -> !mulberry.list<!mulberry.tensor<?x?xf32>>
call @use_tensor_list(%list)
    : (!mulberry.list<!mulberry.tensor<?x?xf32>>) -> ()

func.func private @use_tensor_list(
    %xs: !mulberry.list<!mulberry.tensor<?x?xf32>>) {
  %size = mulberry.list.size %xs
      : !mulberry.list<!mulberry.tensor<?x?xf32>>
  return
}
```

会变成：

```mlir
%desc = mulberry.list.to_desc %list
    : !mulberry.list<!mulberry.tensor<?x?xf32>>
      -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
call @use_tensor_list(%desc)
    : (!mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>) -> ()

func.func private @use_tensor_list(
    %xs: !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>) {
  %length = mulberry.list.desc_length %xs
      : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
  %size = arith.index_cast %length : index to i64
  return
}
```

这个 rewrite 分成 preflight 和 rewrite 两步：先确认所有 direct `func.call` 和
callee body 都能同步转换，再真正插入 `list.to_desc`、重建 `func.call`、修改 callee
signature。这样可以避免“call site 已经改了，但 callee body 后面失败”的半转换 IR。

仍然不支持：

- callee 返回 `List<Tensor>`。
- call 上有 argument/result attributes。
- callee body 里 `list.get` 的结果被 Tensor op 或 `cherry_nn` 消费。

R3.8.69 放开了普通 call result。只要 result 里不包含 source-level
`List<Tensor>`，call-site rewrite 可以保留原 result types，并把旧 call result 的 uses
替换成新 call result：

```mlir
%result = call @use_tensor_list(%list)
    : (!mulberry.list<!mulberry.tensor<?x?xf32>>) -> i64
return %result : i64
```

会变成：

```mlir
%desc = mulberry.list.to_desc %list
    : !mulberry.list<!mulberry.tensor<?x?xf32>>
      -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
%result = call @use_tensor_list(%desc)
    : (!mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>) -> i64
return %result : i64
```

这一步没有打开 `List<Tensor>` 返回值。函数返回 list 仍然需要和 `func.return` 的
value rewrite、descriptor ownership/ABI 一起设计，不能只改 call op。

R3.8.70 固化了 `List<Tensor>` 返回值的目标 IR，但仍然不打开 source-level rewrite。
目标形态不是让函数继续返回 `!mulberry.list<!mulberry.tensor<...>>`，而是让函数边界
直接返回 descriptor：

```mlir
func.func @make_tensor_list(%n: index, %m: index)
    -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>> {
  %tensor = mulberry.tensor.alloc(%n, %m) : !mulberry.tensor<?x?xf32>
  %list = mulberry.list.create(%tensor)
      : (!mulberry.tensor<?x?xf32>)
        -> !mulberry.list<!mulberry.tensor<?x?xf32>>
  %desc = mulberry.list.to_desc %list
      : !mulberry.list<!mulberry.tensor<?x?xf32>>
        -> !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
  return %desc : !mulberry.list_desc<!mulberry.tensor_desc<?x?xf32>>
}
```

真正把 source-level：

```mlir
func.func @make_tensor_list(...) -> !mulberry.list<!mulberry.tensor<?x?xf32>> {
  ...
  return %list : !mulberry.list<!mulberry.tensor<?x?xf32>>
}
```

rewrite 成 descriptor return，必须同时处理：

- callee function result type。
- callee `func.return` operands。
- caller `func.call` result type。
- call result 的后续 users，也就是拿到 descriptor 后如何恢复成高层 list 或保持
  descriptor 语义。
- descriptor 的 ownership/生命周期，避免返回指向失效局部 storage 的假 descriptor。

最后一点是最危险的：`list.to_desc` 当前只描述 IR 形状，不等于已经解决 runtime
ownership。如果强行打开 source-level list return，很容易返回一个看起来合法、实际
data 指针生命周期不清楚的 descriptor。因此这一步只保留 target IR 和 fail-fast
边界，不做 rewrite。

R3.8.71 只清理 boundary preparation 的诊断，不改变语义。诊断里显式使用
`source-level List<Tensor>`，是为了区分两类东西：

- `!mulberry.list<!mulberry.tensor<...>>`：源语言层的 list value。它穿过函数
  参数、返回值或 call result 时，仍然需要完整 rewrite 和 ownership 设计。
- `!mulberry.list_desc<!mulberry.tensor_desc<...>>`：已经准备好的 descriptor
  boundary value。它是当前允许的目标 IR 形态，不应该被 source-level list 的
  fail-fast 诊断误伤。

所以 `source-level List<Tensor> function return boundary preparation is not
implemented yet` 表示“从源语言 list return 自动改写到 descriptor return 还没做”，
不是说显式 descriptor return 不能通过。

R3.8.72 补了 mixed-result regression 测试，目的是把 preflight 边界卡死：

- `func.call` 返回 `(i64, source-level List<Tensor>)` 必须 fail-fast。即使其中
  有普通 scalar result，也不能只保留 scalar、偷偷放过 source list result。
- `func.func` 返回 `(i64, list_desc<TensorDesc>)` 是合法 target IR。descriptor
  已经是 boundary value，不应该被 source-level list 的 fail-fast 检查拦住。

这一步仍然不打开 source-level list return rewrite。它只是防止后续实现时把 mixed
results 做成半转换 IR。

## 借鉴 C++ Template / `std::vector<T>`

Mulberry 不应该实现完整的 C++ template 语言特性。完整 template 会引入用户级
template declaration、template argument deduction、partial specialization、
overload resolution、name mangling、实例化缓存、递归实例化防护和复杂诊断。这些
都不是当前 `List<T>` 需要解决的问题。

真正值得借鉴的是 C++ template 背后的 monomorphization 思想：

- `List<T>` 在语义层是泛型容器类型。
- `List<UInt64>`、`List<Float32[?, ?]>`、`List<List<UInt64>>` 在 lowering/ABI
  层都是不同的 concrete type。
- 每个 concrete `List<T>` 都有自己的 element ABI、descriptor layout 和
  load/store lowering。

可以把 Mulberry 的 `List<T>` 理解成语言内建的 `std::vector<T>` 简化版本：

```text
List<UInt64>          -> { length, data: ptr<i64> }
List<Float32[?, ?]>   -> { length, data: ptr<TensorABI> }
List<List<UInt64>>    -> { length, data: ptr<ListABI<i64>> }
```

这里暂时只保留 `{length, data}`，不引入 `capacity`、allocator、ownership、
move/copy 语义等 `std::vector<T>` 的复杂部分。

这条路线和 Java generic 的 type erasure 不同。Mulberry 不需要把所有 `List<T>`
erase 成统一 object layout，也不需要为 scalar 做 boxing。它应该保留静态类型，
在 lowering/ABI 层根据 concrete element type 递归计算 descriptor。

因此当前设计可以概括为：

```text
用户层：List<T>
语义层：泛型容器类型
lowering 层：按 T 计算 element ABI
ABI 层：{ length, data: ptr<element ABI> }
```

如果将来需要用户级 generic function，再单独考虑受约束的 generic / trait /
concept 设计。当前 `List<T>` 不应提前引入完整 template 系统。

## 函数边界

函数边界指 value 作为参数或返回值穿过 `func.func`、`func.call` 或
`func.return`。

Tensor boundary 目前会 lower 到 `memref`，因为在当前阶段 MLIR 有直接的 memref
function type 表示。

List boundary 目前保持 fail-fast。把 `List<T>` 直接 lower 成 `memref<?xT>` 会把
本地 storage 暴露成 ABI。把 `List<Tensor>` lower 成包含 `memref` 字段的 LLVM
struct 也不合法：`memref` value 必须先 finalization 成 LLVM-compatible
representation，才能嵌入 LLVM aggregate。

## 下一步 ABI 方向

R3.8.41 可以开始把 descriptor helper 接到一个局部、非 boundary 的测试路径，但仍要避免
shortcut。
预期路径是：

1. 保留 `!mulberry.list<T>` 作为高层 value。
2. 当 list 必须穿过边界时，materialize `!mulberry.list_desc<T>`。
3. 把 tensor 和 list storage handle finalization 成 LLVM-compatible fields。
4. 只有当每个字段都是合法后端 value 之后，才把 descriptor lower 成后端 record。
5. 只有 descriptor ABI 存在之后，才启用 `func.func`、`func.call` 和
   `func.return` 的 list conversion。
