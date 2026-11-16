from functools import lru_cache
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl

from triton.compiler.compiler import CompiledKernel

from liger_kernel.ops.utils import ensure_contiguous

MAX_FUSED_COLUMNS = 8192
FORWARD_TILE_ELEMENTS = 16384
BACKWARD_TILE_ELEMENTS = 8192
MAX_TILE_ROWS = 16
LONG_TILE_COLUMNS = 2048
MIN_SPLIT_COLUMNS = 2048
SPLIT_TILE_ROWS = 2
COMPILED_CACHE_SIZE = 128


@triton.jit(do_not_specialize_on_alignment=[0, 1])
def _softmax_ascend_forward(X, Y, ROWS: tl.constexpr, COLS: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    cols = tl.arange(0, C)
    offsets = rows[:, None] * COLS + cols[None, :]
    mask = (rows[:, None] < ROWS) & (cols[None, :] < COLS)
    x = tl.load(X + offsets, mask, other=-float("inf")).to(tl.float32)
    exp = tl.exp(x - tl.max(x, 1)[:, None])
    result = exp / tl.sum(exp, 1)[:, None]
    tl.store(Y + offsets, result, mask)


@triton.jit(do_not_specialize_on_alignment=[0, 1, 2])
def _softmax_ascend_backward(DY, Y, DX, ROWS: tl.constexpr, COLS: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    cols = tl.arange(0, C)
    offsets = rows[:, None] * COLS + cols[None, :]
    mask = (rows[:, None] < ROWS) & (cols[None, :] < COLS)
    dy = tl.load(DY + offsets, mask, other=0).to(tl.float32)
    y = tl.load(Y + offsets, mask, other=0).to(tl.float32)
    dot = tl.sum(dy * y, 1)
    tl.store(DX + offsets, y * (dy - dot[:, None]), mask)


@triton.jit
def _softmax_long_forward(X, Y, COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    maximum = tl.full((), -float("inf"), tl.float32)
    for start in range(tl.cdiv(COLS, BLOCK)):
        cols = start * BLOCK + offsets
        x = tl.load(X + row * COLS + cols, cols < COLS, other=-float("inf")).to(tl.float32)
        maximum = tl.maximum(maximum, tl.max(x, 0))
    total = tl.full((), 0, tl.float32)
    for start in range(tl.cdiv(COLS, BLOCK)):
        cols = start * BLOCK + offsets
        x = tl.load(X + row * COLS + cols, cols < COLS, other=-float("inf")).to(tl.float32)
        total += tl.sum(tl.exp(x - maximum), 0)
    for start in range(tl.cdiv(COLS, BLOCK)):
        cols = start * BLOCK + offsets
        x = tl.load(X + row * COLS + cols, cols < COLS, other=-float("inf")).to(tl.float32)
        tl.store(Y + row * COLS + cols, tl.exp(x - maximum) / total, cols < COLS)


@triton.jit
def _softmax_long_backward(DY, Y, DX, COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    total = tl.full((), 0, tl.float32)
    for start in range(tl.cdiv(COLS, BLOCK)):
        cols = start * BLOCK + offsets
        dy = tl.load(DY + row * COLS + cols, cols < COLS, other=0).to(tl.float32)
        y = tl.load(Y + row * COLS + cols, cols < COLS, other=0).to(tl.float32)
        total += tl.sum(dy * y, 0)
    for start in range(tl.cdiv(COLS, BLOCK)):
        cols = start * BLOCK + offsets
        dy = tl.load(DY + row * COLS + cols, cols < COLS, other=0).to(tl.float32)
        y = tl.load(Y + row * COLS + cols, cols < COLS, other=0).to(tl.float32)
        tl.store(DX + row * COLS + cols, y * (dy - total), cols < COLS)


@triton.jit(do_not_specialize_on_alignment=[0, 1, 2])
def split_backward(
    DY, Y, DX, ROWS: tl.constexpr, COLS: tl.constexpr, R: tl.constexpr, C: tl.constexpr, T: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    a = tl.arange(0, C)
    b = C + tl.arange(0, T)
    ia = row[:, None] * COLS + a[None, :]
    ib = row[:, None] * COLS + b[None, :]
    ma = row[:, None] < ROWS
    mb = ma & (b[None, :] < COLS)
    ya = tl.load(Y + ia, ma, other=0).to(tl.float32)
    yb = tl.load(Y + ib, mb, other=0).to(tl.float32)
    da = tl.load(DY + ia, ma, other=0).to(tl.float32)
    db = tl.load(DY + ib, mb, other=0).to(tl.float32)
    dot = tl.sum(ya * da, 1) + tl.sum(yb * db, 1)
    tl.store(DX + ia, ya * (da - dot[:, None]), ma)
    tl.store(DX + ib, yb * (db - dot[:, None]), mb)


@triton.jit(do_not_specialize_on_alignment=[0, 1, 2])
def triple_backward(
    DY,
    Y,
    DX,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    U: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    a = tl.arange(0, C)
    b = C + tl.arange(0, T)
    c = C + T + tl.arange(0, U)
    ic = row[:, None] * COLS + c[None, :]
    ia = row[:, None] * COLS + a[None, :]
    ib = row[:, None] * COLS + b[None, :]
    ma = row[:, None] < ROWS
    mb = ma & (b[None, :] < COLS)
    mc = ma & (c[None, :] < COLS)
    ya = tl.load(Y + ia, ma, other=0).to(tl.float32)
    yb = tl.load(Y + ib, mb, other=0).to(tl.float32)
    da = tl.load(DY + ia, ma, other=0).to(tl.float32)
    db = tl.load(DY + ib, mb, other=0).to(tl.float32)
    yc = tl.load(Y + ic, mc, other=0).to(tl.float32)
    dc = tl.load(DY + ic, mc, other=0).to(tl.float32)
    dot = tl.sum(ya * da, 1) + tl.sum(yb * db, 1) + tl.sum(yc * dc, 1)
    tl.store(DX + ia, ya * (da - dot[:, None]), ma)
    tl.store(DX + ib, yb * (db - dot[:, None]), mb)
    tl.store(DX + ic, yc * (dc - dot[:, None]), mc)


def _ascend_tile(cols):
    columns = triton.next_power_of_2(cols)
    rows = max(1, min(MAX_TILE_ROWS, BACKWARD_TILE_ELEMENTS // columns))
    return rows, columns


@lru_cache(maxsize=COMPILED_CACHE_SIZE)
def _launcher(rows, cols, dtype, device, backward):
    row_tile, col_tile = _ascend_tile(cols)
    if not backward:
        row_tile = max(1, min(MAX_TILE_ROWS, FORWARD_TILE_ELEMENTS // col_tile))
    with torch.npu.device(device):
        if cols > MAX_FUSED_COLUMNS:
            kernel = _softmax_long_backward if backward else _softmax_long_forward
            grid = (rows, 1, 1)
            constants = (cols, LONG_TILE_COLUMNS)
        elif backward and MIN_SPLIT_COLUMNS < cols <= MAX_FUSED_COLUMNS and cols != col_tile:
            prefix = col_tile // 2
            remainder = cols - prefix
            row_tile = SPLIT_TILE_ROWS
            grid = (triton.cdiv(rows, row_tile), 1, 1)
            if remainder == triton.next_power_of_2(remainder):
                kernel = split_backward
                constants = (rows, cols, row_tile, prefix, remainder)
            else:
                middle = triton.next_power_of_2(remainder) // 2
                tail = triton.next_power_of_2(remainder - middle)
                kernel = triple_backward
                constants = (rows, cols, row_tile, prefix, middle, tail)
        else:
            kernel = _softmax_ascend_backward if backward else _softmax_ascend_forward
            grid = (triton.cdiv(rows, row_tile), 1, 1)
            constants = (rows, cols, row_tile, col_tile)
        compiled = kernel.warmup(*([dtype] * (3 if backward else 2)), *constants, grid=grid)
        compiled._init_handles()
        run = compiled.run
        function, metadata = compiled.function, compiled.packed_metadata
        stream_fn = torch_npu._C._npu_getCurrentRawStreamNoWait

        def launch(*args):
            stream = stream_fn(device)
            run(
                *grid,
                stream,
                function,
                metadata,
                compiled.launch_metadata(grid, stream, *args),
                CompiledKernel.launch_enter_hook,
                CompiledKernel.launch_exit_hook,
                *args,
            )

        from liger_kernel.ops.backends._ascend.ops._softmax_launcher import native_entry

        library, entry = (
            (None, 0) if run.compile_only or run.enable_msprof_register_tensor else native_entry(compiled, grid)
        )
        launch.native_spec = (compiled, grid, stream_fn, device, library, entry)
        return launch


def _softmax_forward(x):
    cols = x.shape[-1]
    x = x.contiguous()
    rows = x.numel() // cols
    y = torch.empty_like(x)
    _launcher(rows, cols, x.dtype, x.device.index, False)(x, y)
    return y, cols, rows, False


def _softmax_backward(dy, y, col_tile, row_tile, multi_block_launch):
    dy, y = dy.contiguous(), y.contiguous()
    cols = y.shape[-1]
    rows = y.numel() // cols
    dx = torch.empty_like(y)
    _launcher(rows, cols, y.dtype, y.device.index, True)(dy, y, dx)
    return dx


@lru_cache(maxsize=1)
def _native_module():
    from torch.utils.cpp_extension import load

    return load(
        name="liger_softmax_ascend_autograd",
        sources=[str(Path(__file__).with_name("_softmax_native.cpp"))],
        extra_cflags=["-O2", "-fvisibility=hidden"],
        verbose=False,
    )


def _native_specs(x):
    cols = x.shape[-1]
    rows = x.numel() // cols
    return (
        _launcher(rows, cols, x.dtype, x.device.index, False).native_spec,
        _launcher(rows, cols, x.dtype, x.device.index, True).native_spec,
    )


@lru_cache(maxsize=1)
def _native_apply():
    return _native_module().CachedOp(_native_specs).apply


class LigerSoftmaxFunction(torch.autograd.Function):
    @classmethod
    def apply(cls, x, dim=-1):
        if torch._C._are_functorch_transforms_active() or not x.numel():
            return super().apply(x, dim)
        rank = x.ndim
        if dim < -max(rank, 1) or dim >= max(rank, 1):
            raise IndexError(f"Dimension out of range for tensor with {rank} dimensions: {dim}")
        axis = dim % max(rank, 1)
        if not rank:
            return _native_apply()(x.reshape(1)).reshape(())
        if axis == rank - 1:
            return _native_apply()(x)
        return _native_apply()(x.movedim(axis, -1)).movedim(-1, axis)

    @staticmethod
    @ensure_contiguous
    def forward(ctx, input_: torch.Tensor, dim: int = -1):
        rank = input_.ndim
        if dim < -max(rank, 1) or dim >= max(rank, 1):
            raise IndexError(f"Dimension out of range for tensor with {rank} dimensions: {dim}")
        ctx.dim = dim % max(rank, 1)
        ctx.scalar = rank == 0
        ctx.empty = input_.numel() == 0
        if ctx.empty:
            return torch.empty_like(input_)
        if ctx.scalar:
            input_ = input_.reshape(1)
        elif ctx.dim != rank - 1:
            input_ = input_.movedim(ctx.dim, -1)
        y, BLOCK_SIZE, num_warps, multi_block_launch = _softmax_forward(input_)
        ctx.save_for_backward(y)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.multi_block_launch = multi_block_launch
        if ctx.scalar:
            return y.reshape(())
        return y.movedim(-1, ctx.dim)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        if ctx.empty:
            result = torch.empty_like(grad_output)
            return (result, None)[: len(ctx.needs_input_grad)]
        if ctx.scalar:
            grad_output = grad_output.reshape(1)
        else:
            grad_output = grad_output.movedim(ctx.dim, -1)
        (y,) = ctx.saved_tensors
        dx = _softmax_backward(
            grad_output,
            y,
            ctx.BLOCK_SIZE,
            ctx.num_warps,
            ctx.multi_block_launch,
        )
        dx = dx.reshape(()) if ctx.scalar else dx.movedim(-1, ctx.dim)
        return (dx, None)[: len(ctx.needs_input_grad)]
