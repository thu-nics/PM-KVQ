import math
import fast_hadamard_transform

import torch

from pm_kvq.utils.hadamard_matrices_utils import *

# Adapted from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/utils/matmul_had.py


def hadamard_transform(X, transpose=False):
    """
    Y = X @ H_d^T if transpose==False,
    H_d = kron(H_m, H_{2^n}) if d != 2^n
    """
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    if K > 1:
        input = hadK.view(1, K, K).to(input) @ input

    return input.view(X.shape) / torch.tensor(n).sqrt()


def hadamard_transform_cuda(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    if K == 1:
        return fast_hadamard_transform.hadamard_transform(X.contiguous(), 1.0 / torch.tensor(n).sqrt())
    else:
        input = X.view(-1, K, n // K)
        input = fast_hadamard_transform.hadamard_transform(input.contiguous(), 1.0 / torch.tensor(n).sqrt())
        input = hadK.to(input.device).to(input.dtype) @ input
        return input.reshape(X.shape)


def blockwise_hadamard_transform_cuda(X, had_dim=-1, transpose=False):
    if had_dim == -1:
        return hadamard_transform_cuda(X, transpose)
    else:
        assert X.shape[-1] % had_dim == 0
        ori_shape = X.shape
        X = hadamard_transform_cuda(X.reshape(-1, had_dim), transpose).reshape(ori_shape)
        return X


def apply_hadamard_transform_to_linear(module, had_dim=-1, output=False):
    """
    if output==True: out = module(in) @ H^T
    if output==False: out = module(in @ H)
    """
    assert isinstance(module, torch.nn.Linear)
    # if had_dim != -1:
    #     assert is_pow2(had_dim), "Per-head hadamard dimension must be a power of 2!"

    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    W_ = W_.float().cuda()

    if had_dim == -1:
        if output:
            W_ = hadamard_transform_cuda(W_.t()).t()
            if module.bias is not None:
                b = hadamard_transform_cuda(module.bias.data.reshape(1, -1)).reshape(-1)
                module.bias.data = b.to(device=module.bias.device, dtype=module.bias.dtype)
        else:
            W_ = hadamard_transform_cuda(W_)
    else:
        if output:
            W_ = blockwise_hadamard_transform_cuda(W_.t(), had_dim).t()
            if module.bias is not None:
                module.bias.data = blockwise_hadamard_transform_cuda(module.bias.data, had_dim)
        else:
            W_ = blockwise_hadamard_transform_cuda(W_, had_dim)

    module.weight.data = W_.to(device=dev, dtype=dtype)


def get_random_orthogonal_matrix(size, mode, device):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def random_hadamard_matrix(size, device):
    Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return hadamard_transform(Q).to(device)


def random_orthogonal_matrix(size, device):
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_hadK(n, transpose=False):
    hadK, K = None, None
    if n % 232 == 0:
        assert is_pow2(n // 232)
        K = 232
        hadK = get_had232().T if transpose else get_had232()
    elif n % 172 == 0:  # llama-2-7b up
        assert is_pow2(n // 172)
        K = 172
        hadK = get_had172().T if transpose else get_had172()
    elif n % 156 == 0:  # llama-1-30b 3x hidden
        assert is_pow2(n // 156)
        K = 156
        hadK = get_had156().T if transpose else get_had156()
    elif n % 148 == 0:  # qwen2-7b intermediate, 4x hidden
        assert is_pow2(n // 148)
        K = 148
        hadK = get_had148().T if transpose else get_had148()
    elif n % 140 == 0:  # llama-1-30b intermediate
        assert is_pow2(n // 140)
        K = 140
        hadK = get_had140().T if transpose else get_had140()
    elif n % 108 == 0:  # llama-1-13b intermediate
        assert is_pow2(n // 108)
        K = 108
        hadK = get_had108().T if transpose else get_had108()
    elif n % 60 == 0:  # llama-1-13b 3x hidden
        assert is_pow2(n // 60)
        K = 60
        hadK = get_had60().T if transpose else get_had60()
    elif n % 52 == 0:  # llama-1-13b 1x hidden
        assert is_pow2(n // 52)
        K = 52
        hadK = get_had52().T if transpose else get_had52()
    elif n % 36 == 0:
        assert is_pow2(n // 36)
        K = 36
        hadK = get_had36().T if transpose else get_had36()
    elif n % 28 == 0:  # llama-3 up
        assert is_pow2(n // 28)
        K = 28
        hadK = get_had28().T if transpose else get_had28()
    elif n % 40 == 0:
        assert is_pow2(n // 40)
        K = 40
        hadK = get_had40().T if transpose else get_had40()
    elif n % 20 == 0:
        assert is_pow2(n // 20)
        K = 20
        hadK = get_had20().T if transpose else get_had20()
    elif n % 12 == 0:
        assert is_pow2(n // 12)
        K = 12
        hadK = get_had12().T if transpose else get_had12()
    else:
        assert is_pow2(n)
        K = 1

    return hadK, K


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)
