# `imb` REPL

`imb` 是 Mulberry 的交互式执行环境。它使用 JIT 编译每次提交的输入，并
保留当前 session 的变量、函数、struct、trait 和 import，适合快速试验语法、
检查类型和调试标准库代码。

## 启动

在仓库根目录构建 `imb`：

```sh
make mulberry-build
```

启动：

```sh
./build/release/bin/imb
```

默认提示符是 `imb > `。输入未完成的表达式或声明后，REPL 使用 `... ` 作为
多行提示符。按 `Ctrl-C` 可以取消当前的多行输入。

REPL submission 的最后一个 declaration 或 statement 可以用输入结束代替分号，
所以交互输入通常可以省略末尾的 `;`。普通 `.mulberry` 文件仍然遵循正常语法，
例如文件中的 `import io;` 需要分号。

## 求值

有非 `Unit` 返回值的表达式会自动打印结果：

```mulberry
1 + 2
// 3

var answer = 40
answer + 2
// 42
```

声明和返回 `Unit` 的调用不会额外打印内部的数值结果。REPL 中的 `io.print()`
会在输出后补一个换行；`io.println()` 自己写入换行。批处理程序的 `print()`
行为不受这个 REPL 专用处理影响。

成功提交的定义会留在当前 session 中，后续输入可以直接使用：

```mulberry
fn double(value: UInt64): UInt64 { return value + value; }
double(21)
// 42
```

如果编译、JIT 初始化或求值失败，这次 submission 不会污染当前语义 session。

## 内部命令

内部命令以 `:` 开头。命令参数用空格分隔；`:help` 可以查看全部命令，也可以
查看某个命令的 usage。

| 命令 | 作用 |
| --- | --- |
| `:help`, `:h` | 列出命令；`:help <command>` 查看命令 usage |
| `:pwd` | 打印当前工作目录 |
| `:cd <path>` | 修改当前工作目录，支持绝对路径和相对路径 |
| `:type <expression>`, `:t` | 打印表达式的 Mulberry 类型 |
| `:load <path>`, `:l` | 把文件加载到当前 session |
| `:quit`, `:q`, `:exit` | 退出 `imb` |

例子：

```text
imb > :type 1 + 2
UInt64
imb > :pwd
/work/demo
imb > :cd examples
imb > :load ../stdlib-demo.mulberry
```

`:type` 只检查表达式，不执行表达式。缺少参数或使用未知命令时，`imb` 会打印
usage 或诊断信息并继续运行。

## Import 和 Load

普通 `import` 使用 Mulberry 语法，不是内部命令：

```mulberry
import a.b.c
c.answer()
```

当前工作目录是本地模块根目录。在 `/work/demo` 中，`import a.b.c` 会加载
`/work/demo/a/b/c.mulberry`。该文件必须声明匹配的 package：

```mulberry
package a.b.c;
```

`import io`、`import std.foo` 等普通 import 也可以在 REPL 中使用；REPL 输入的
最后一个 declaration 可以省略分号。

`:load <path>` 按当前工作目录解析路径，并把整个文件作为一次 submission 加入
当前 session。文件的编译和初始化成功后，文件中的变量、函数、trait 和 import
才会全部生效；如果失败，这些新语义声明不会保留。文件执行过程中已经发生的
文件写入、打印等外部运行时副作用不能回滚。`:cd` 会同时影响后续的 import 和
`:load` 路径解析。

## 编辑、补全和 History

`imb` 使用 Isocline 提供交互式行编辑。支持常用的 Emacs 风格编辑键、Tab 补全、
ghost suggestion 和多行输入。

- Tab 可以补全关键字、类型、prelude 名称、当前 session 中的变量和函数，以及
  struct member 和 package member。
- 出现灰色的历史或名称 suggestion 时，按右箭头接受 suggestion。
- History 保存在 `$HOME/.mulberry/mulberry-history`，而不是直接放在 home 目录下。
- 没有输入内容时，Up/Down 进行普通历史导航。
- 输入非空内容且它不是完整的历史项时，按 Up 会用当前内容在 history 中搜索，
  不要求历史项从输入开头匹配。
- 输入已经是完整历史项时，按 Up 恢复普通的 older-entry 导航，避免历史搜索
  把导航锁在同一条命令上。
- `Ctrl-R` 和 `Ctrl-S` 可以使用 Isocline 的历史搜索。

历史项会跨进程保存。重复项不会重复写入；启动新的 `imb` 后仍然可以使用之前
的历史和补全。

## 一个完整例子

```text
$ ./build/release/bin/imb
imb > var x = 123
imb > x + 1
124
imb > :t x
UInt64
imb > import io
imb > io.print("hello")
hello
imb > :q
```
