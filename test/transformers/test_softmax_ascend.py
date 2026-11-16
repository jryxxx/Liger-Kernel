import pytest
import torch

from liger_kernel.ops import LigerSoftmaxFunction
from liger_kernel.transformers.functional import liger_softmax
from liger_kernel.transformers.softmax import LigerSoftmax


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((), 0),
        ((0, 3), -1),
        ((3, 0), 0),
        ((2, 1), -1),
        ((31,), -1),
        ((3, 5, 7), 0),
        ((3, 5, 7), 1),
        ((3, 5, 7), -1),
        ((3, 3584), -1),
        ((3, 5120), -1),
        ((3, 7168), -1),
        ((2, 4097), -1),
        ((1, 65537), -1),
    ],
)
def test_shapes_dims_gradients(dtype, shape, dim):
    x = torch.randn(shape, dtype=dtype, device="npu", requires_grad=True)
    ref = x.detach().clone().requires_grad_(True)
    actual = LigerSoftmax(dim)(x)
    expected = torch.softmax(ref, dim)
    tolerance = {torch.float16: 2e-3, torch.bfloat16: 2e-2, torch.float32: 1e-5}[dtype]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    grad = torch.randn_like(expected)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(x.grad, ref.grad, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dim", [-3, -2, -1, 0, 1, 2])
def test_noncontiguous(dim):
    base = torch.randn(4, 5, 7, device="npu", requires_grad=True)
    x = base.transpose(0, 1)
    ref = x.detach().clone().requires_grad_(True)
    y = liger_softmax(x, dim)
    expected = torch.softmax(ref, dim)
    torch.testing.assert_close(y, expected, atol=1e-6, rtol=1e-5)
    grad = torch.randn(4, 5, 7, device="npu").transpose(0, 1)
    y.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(base.grad.transpose(0, 1), ref.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("dim", [-4, 3])
def test_invalid_dim(dim):
    with pytest.raises(IndexError, match="Dimension out of range"):
        liger_softmax(torch.randn(2, 3, 4, device="npu"), dim)


def test_original_function_signature():
    x = torch.randn(3, 7, device="npu", requires_grad=True)
    y = LigerSoftmaxFunction.apply(x)
    y.backward(torch.randn_like(y))
    assert x.grad is not None


def test_numerical_gradient():
    x = torch.randn(2, 3, device="npu", requires_grad=True)
    assert torch.autograd.gradcheck(liger_softmax, (x,), eps=1e-3, atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize(
    "values",
    [
        [1e4, 1e4 - 1, -1e4],
        [float("nan"), 0, 1],
        [float("inf"), 0, 1],
        [-float("inf"), -float("inf"), -float("inf")],
        [-float("inf"), 1, 2],
    ],
)
def test_special_values(values):
    x = torch.tensor([values], device="npu", requires_grad=True)
    ref = x.detach().clone().requires_grad_(True)
    y, expected = liger_softmax(x), torch.softmax(ref, -1)
    torch.testing.assert_close(y, expected, equal_nan=True)
    dy = torch.randn_like(y)
    y.backward(dy)
    expected.backward(dy)
    torch.testing.assert_close(x.grad, ref.grad, equal_nan=True)


def test_same_elements_different_columns():
    for shape in [(8, 16), (4, 32), (2, 64), (8, 16)]:
        x = torch.randn(shape, device="npu", requires_grad=True)
        ref = x.detach().clone().requires_grad_(True)
        y, expected = liger_softmax(x), torch.softmax(ref, -1)
        torch.testing.assert_close(y, expected)
        dy = torch.randn_like(y)
        y.backward(dy)
        expected.backward(dy)
        torch.testing.assert_close(x.grad, ref.grad)


def test_saved_output_version_check():
    x = torch.randn(2, 7, device="npu", requires_grad=True)
    y = liger_softmax(x)
    with torch.no_grad():
        y.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.backward(torch.ones_like(y))


def test_saved_tensor_hooks_and_retained_graph():
    saved = []

    def pack(tensor):
        saved.append(tuple(tensor.shape))
        return tensor.detach().clone()

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        x = torch.randn(2, 7, device="npu", requires_grad=True)
        y = liger_softmax(x)
    grad = torch.randn_like(y)
    first = torch.autograd.grad(y, x, grad, retain_graph=True)[0]
    second = torch.autograd.grad(y, x, grad)[0]
    torch.testing.assert_close(first, second)
    assert saved


@pytest.mark.parametrize("mode", [torch.no_grad, torch.inference_mode])
def test_no_grad(mode):
    with mode():
        x = torch.randn(2, 7, device="npu")
        y = liger_softmax(x)
        torch.testing.assert_close(y, torch.softmax(x, -1))
        assert y.grad_fn is None


def test_graph_replay():
    x = torch.randn(4, 32, device="npu")
    for _ in range(3):
        liger_softmax(x)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        y = liger_softmax(x)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(y, torch.softmax(x, -1))


def test_unaligned_view_and_current_stream():
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        x = torch.randn(130, device="npu")[1:129].reshape(4, 32).requires_grad_(True)
        ref = x.detach().clone().requires_grad_(True)
        y, expected = liger_softmax(x), torch.softmax(ref, -1)
        dy = torch.randn_like(y)
        y.backward(dy)
        expected.backward(dy)
    stream.synchronize()
    torch.testing.assert_close(y, expected)
    torch.testing.assert_close(x.grad, ref.grad)
