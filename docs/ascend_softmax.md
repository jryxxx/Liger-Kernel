# Ascend Softmax

本实现面向 Liger Kernel v0.8.2，使用 Triton-Ascend 实现 Softmax 的前向和反向计算。Ascend 实现位于 `src/liger_kernel/ops/backends/_ascend/ops/softmax.py`，由仓库 backend registry 选择。

## 环境与安装

使用 CANN 9.1.0、Triton-Ascend 3.2.2、torch/torch_npu 2.9.0 系列。主机启动适配层需要 C++17 编译器及 Ninja。

```bash
source /path/to/cann/set_env.sh
python -m pip install ninja
python -m pip install --no-deps -e .
```

在任务固定依赖环境中使用 `--no-deps`，避免 v0.8.2 的较旧依赖声明覆盖已经准备的版本。首次执行会编译 Triton 内核及 C++ 主机适配层。可设置 `MAX_JOBS=2` 限制 C++ 编译并行度。

## 接口

```python
import torch
import torch_npu
from liger_kernel.transformers.softmax import LigerSoftmax
from liger_kernel.transformers.functional import liger_softmax

x = torch.randn(2, 128, 4096, dtype=torch.bfloat16, device="npu", requires_grad=True)
y = LigerSoftmax(dim=-1)(x)
z = liger_softmax(x, dim=1)
y.backward(torch.randn_like(y))
```

默认沿最后一维计算；`dim` 支持正负维度。输入和输出保持相同 shape、dtype。实现覆盖 FP16、BF16、FP32，支持标量、空张量和非连续输入。非最后一维通过维度移动及连续存储完成计算，输出再恢复原维度顺序。

## 实现

归约以 FP32 进行，先减去行最大值，再计算指数及归一化。短行使用二维行列分块；非整齐列宽的反向按有效宽度拆成两段或三段，避免过量填充。长行分段计算最大值、归一化和输出，限制片上内存用量。反向计算 `y * (dy - sum(dy * y))`。

C++ 主机适配层负责分配、autograd 节点和启动，数值计算仍在 Triton 内核中。反向保存前向输出，并遵循保存张量钩子和版本校验。编译上下文缓存有固定容量，键包括元素数、列宽、dtype 和设备；每次启动读取当前 stream 和真实 tensor 地址。

原生入口复用 Triton-Ascend 3.2.2 生成启动器中的后端提交逻辑。升级编译器或 torch_npu 时，应重新执行功能、图重放和性能测试。

## 复测

```bash
python -m pytest -q test/transformers/test_softmax.py test/transformers/test_softmax_ascend.py
LIGER_BENCH_TARGET=softmax_check python benchmark/scripts/benchmark_softmax.py --overwrite
LIGER_BENCH_TARGET=softmax_check python benchmark/scripts/benchmark_softmax.py --sweep-mode model_config --overwrite
```

原始 pytest 的断言、容差和官方 benchmark 保持不变。正式性能验收逐配置比较 Liger 与本机 PyTorch，包含 forward、backward、full；不同硬件分别记录环境、原始数据、重复测量和 profiling 结果。
