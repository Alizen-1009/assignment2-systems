import torch


def online_softmax_v1(x: torch.Tensor, v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Computes the softmax of x along the specified dimension using an online
    algorithm that is more memory efficient than the standard softmax.

    Args:
        x: Input tensor.
        v: Input tensor for the softmax computation.
        dim: Dimension along which to compute the softmax.
    Returns:
        Softmax of x along the specified dimension.
    """
    state_shape = x.select(dim, 0).shape
    pre_max = torch.full(state_shape, -float("inf"), dtype=x.dtype, device=x.device)
    now_max = torch.full(state_shape, -float("inf"), dtype=x.dtype, device=x.device)
    ans = torch.zeros_like(v.select(dim, 0))
    pre_sum = torch.zeros(state_shape, dtype=x.dtype, device=x.device)
    now_sum = torch.zeros(state_shape, dtype=x.dtype, device=x.device)
    for i in range(x.shape[dim]):
        now_max = torch.maximum(pre_max, x.select(dim, i))

        scale = torch.exp(pre_max - now_max)
        value = torch.exp(x.select(dim, i) - now_max)
        now_sum = pre_sum * scale + value

        ans = pre_sum * scale * ans / now_sum + value * v.select(dim, i) / now_sum

        pre_sum = now_sum
        pre_max = now_max

    return ans


def online_softmax_v2(x: torch.Tensor, v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Computes the softmax of x along the specified dimension using an online
    algorithm that is more memory efficient than the standard softmax.

    Args:
        x: Input tensor.
        v: Input tensor for the softmax computation.
        dim: Dimension along which to compute the softmax.
    Returns:
        Softmax of x along the specified dimension.
    """
    state_shape = x.select(dim, 0).shape
    pre_max = torch.full(state_shape, -float("inf"), dtype=x.dtype, device=x.device)
    now_max = torch.full(state_shape, -float("inf"), dtype=x.dtype, device=x.device)
    fz = torch.zeros_like(v.select(dim, 0))
    fm = torch.zeros(state_shape, dtype=x.dtype, device=x.device)
    for i in range(x.shape[dim]):
        now_max = torch.maximum(pre_max, x.select(dim, i))
        scale = torch.exp(pre_max - now_max)
        exp_score = torch.exp(x.select(dim, i) - now_max)
        fz = fz * scale + exp_score * v.select(dim, i)
        fm = fm * scale + exp_score
        pre_max = now_max

    return fz / fm
