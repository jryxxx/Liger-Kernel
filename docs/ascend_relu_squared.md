# Ascend ReLU²

本实现面向 Liger Kernel v0.8.2，使用 Triton-Ascend 编写前向和反向内核，并用 C++ 主机适配层减少启动和 autograd 开销。

## 环境

本次验证使用 Ascend 910B3、CANN 9.1.0、Python 3.11.4、torch 2.9.0、torch_npu 2.9.0.post6、Triton-Ascend 3.2.2。主机适配层需要支持 C++17 的编译器和 Ninja。

先加载当前 CANN 安装目录的 `set_env.sh`，激活对应 Python 环境，然后执行：

```bash
python -m pip install ninja
python -m pip install --no-deps -e .
```

Liger v0.8.2 的现有 NPU 依赖声明使用较早的版本；此处使用 `--no-deps` 保留任务规定的版本组合。主机适配层和 Triton 内核在首次使用时编译并缓存，后续调用复用编译结果。可以用 `MAX_JOBS=2` 限制主机适配层的编译并行度。

## 接口

```python
import torch
import torch_npu
from liger_kernel.transformers.relu_squared import LigerReLUSquared

x = torch.randn(2048, 4096, device="npu", dtype=torch.bfloat16, requires_grad=True)
y = LigerReLUSquared()(x)
y.backward(torch.ones_like(y))
```

前向为 `torch.square(torch.relu(x))`，反向为平方与 ReLU 的链式梯度。支持 FP16、BF16、FP32，保留输入形状和 dtype；支持标量、空张量和任意维度。非连续输入及梯度按上游接口转为连续存储。

反向保留平方梯度的 dtype 舍入顺序。条件选择实现 ReLU，使 NaN、Inf、非正输入及低精度溢出的行为与参考路径一致。

## 实现

- 前向和反向分别使用一个 Triton 内核，连续元素分块为 8192，尾部使用 mask，偏移采用 int64。
- C++ 适配层负责 ATen 分配、autograd 节点、保存张量及版本检查；数值计算由 Triton 完成。
- 按元素数、dtype、设备缓存主机上下文，上限 128 项。上下文保存编译状态，不保存请求 Tensor。活跃 autograd 图按正常规则保存反向所需输入。
- 原生入口附加在 Triton-Ascend 自动生成的启动器中，复用其 `_launch` 函数，包括工作区、任务队列和 NPU profiling 逻辑；每次调用读取实际地址和当前 stream。
- 普通路径省去 Python 参数解析。动态 launch hook、Tensor profiler 和 compile-only 模式保留相应后端路径。
- Python 缓存、C++ 桥和生成入口均针对上述版本组合验证。升级 Triton-Ascend 或 PyTorch 后需要重新执行本页测试。

## 验证

```bash
python -m pytest -q test/transformers/test_relu_squared.py test/transformers/test_relu_squared_ascend.py
LIGER_BENCH_TARGET=relu_check python benchmark/scripts/benchmark_relu_squared.py --overwrite
LIGER_BENCH_TARGET=relu_check python benchmark/scripts/benchmark_relu_squared.py --sweep-mode model_config --overwrite
python -m ruff check src/liger_kernel/ops/relu_squared.py src/liger_kernel/ops/_relu_squared_launcher.py test/transformers/test_relu_squared_ascend.py setup.py
```

上游原始测试、断言、容差和 benchmark 均保持原样。默认长度及模型配置 sweep 覆盖 33 个速度配置，以及 11 个 full 内存配置。性能判定逐项比较 Liger / PyTorch 耗时；完整数值和多轮结果见随附测试报告。

补充测试覆盖边界形状、布局、特殊数值、数值梯度、关闭梯度、重复反向、输入版本检查、保存张量钩子、线程/stream、动态 launch hook、图重放及已完成计算图的存储释放。

## 排查

- 首次调用提示找不到 Ninja 或编译器：安装 Ninja，确认 `c++ --version` 可用，且编译器支持 C++17。
- 无法加载 CANN 或访问 NPU：检查已加载的 CANN 环境和 torch_npu 版本，并用 `npu-smi info` 确认设备状态。
- 升级依赖后出现生成启动器接口不匹配：核对版本组合并重建相应编译缓存，再运行全部测试。
- 性能比较使用相同环境、同一设备和未修改的官方基准；首次编译不计入稳态延迟。采集 profiler 时的耗时单独用于分析。
