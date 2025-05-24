from typing import Literal

import torch
from torch import nn

from pm_kvq.quantization.quantizer.pack_funcs import pack_funcs, unpack_funcs


class BaseQuantizer(nn.Module):

    def __init__(
        self,
        n_bits: int,
        granularity: Literal["per_tensor", "per_channel", "per_token", "per_group"],
        symmetric: bool,
        group_size: int,
        pack: bool = False,
        round_zeros: bool = True,
        ste: bool = False,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.quant_dtype = torch.uint16 if not self.symmetric else torch.int16

        if not symmetric:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1
        else:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1

        self.granularity = granularity
        if self.granularity == "per_group":
            self.group_size = group_size

        self.pack = pack
        self.round_zeros = round_zeros
        self.enable = True

        if ste:
            self.round_func = round_ste
        else:
            self.round_func = torch.round

    def forward(self, x: torch.Tensor):
        if self.n_bits < 0 or not self.enable:
            return x
        x_dequant = self.fake_quant(x)
        return x_dequant

    def fake_quant(self, tensor):
        pass

    def real_quant(self, tensor):
        pass

    def dequant(self, tensor):
        pass

    def reshape_tensor(self, tensor):
        dim = tensor.shape[-1]
        if self.granularity == "per_group":
            assert dim % self.group_size == 0
            tensor = tensor.reshape(-1, self.group_size)
        elif self.granularity == "per_token" or self.granularity == "per_channel":
            tensor = tensor.reshape(-1, dim)
        elif self.granularity == "per_tensor":
            tensor = tensor.reshape(1, -1)
        else:
            raise NotImplementedError
        return tensor

    def get_quant_params(self, tensor):
        if not self.symmetric:
            max_val = tensor.amax(dim=1, keepdim=True)
            min_val = tensor.amin(dim=1, keepdim=True)
            scales = torch.clamp((max_val - min_val) / self.qmax, min=1e-5, max=1e5)
            if self.round_zeros:
                zeros = torch.clamp(-torch.round(min_val / scales), self.qmin, self.qmax)
            else:
                zeros = -min_val
        else:
            max_val = tensor.abs().amax(dim=1, keepdim=True)
            scales = torch.clamp(max_val / self.qmax, min=1e-4, max=1e4)
            zeros = 0
        return scales, zeros

    def pack_tensor(self, tensor):
        return pack_funcs[self.n_bits](tensor)

    def unpack_tensor(self, tensor):
        return unpack_funcs[self.n_bits](tensor)


class UntrainableQuantizer(BaseQuantizer):

    def __init__(
        self,
        n_bits: int,
        granularity: Literal["per_tensor", "per_channel", "per_token", "per_group"],
        symmetric: bool,
        group_size: int,
        pack: bool = False,
        round_zeros: bool = True,
        ste: bool = False,
    ):
        super().__init__(n_bits, granularity, symmetric, group_size, pack, round_zeros, ste)

    def fake_quant(self, tensor):
        org_tensor_shape = tensor.shape
        tensor = self.reshape_tensor(tensor)
        with torch.no_grad():
            scales, zeros = self.get_quant_params(tensor)
        if self.n_bits == 16:
            tensor = tensor.to(torch.float)
        if self.round_zeros:
            tensor = (torch.clamp(self.round_func(tensor / scales) + zeros, self.qmin, self.qmax) - zeros) * scales
        else:
            tensor = torch.clamp(self.round_func((tensor + zeros) / scales), self.qmin, self.qmax) * scales - zeros
        if self.n_bits == 16:
            tensor = tensor.to(torch.bfloat16)
        assert torch.isnan(tensor).sum() == 0
        tensor = tensor.reshape(org_tensor_shape)
        return tensor

    def real_quant(self, tensor, return_params=False):
        org_tensor_shape = tensor.shape
        tensor = self.reshape_tensor(tensor)
        with torch.no_grad():
            scales, zeros = self.get_quant_params(tensor)
        if self.n_bits == 16:
            tensor = tensor.to(torch.float)
        if self.round_zeros:
            tensor = torch.clamp(self.round_func(tensor / scales) + zeros, self.qmin, self.qmax)
        else:
            tensor = torch.clamp(self.round_func((tensor + zeros) / scales), self.qmin, self.qmax)
        assert torch.isnan(tensor).sum() == 0
        tensor = tensor.reshape(org_tensor_shape).to(self.quant_dtype)
        if self.pack:
            tensor = self.pack_tensor(tensor)
        if return_params:
            if self.granularity == "per_group":
                scales = scales.reshape(*org_tensor_shape[:-1], -1)
                if not self.symmetric:
                    zeros = zeros.reshape(*org_tensor_shape[:-1], -1)
            return tensor, scales, zeros
        else:
            return tensor

    def dequant(self, tensor, scales, zeros):
        if self.pack:
            tensor = self.unpack_tensor(tensor)
            if not self.symmetric:
                tensor &= 2**self.n_bits - 1
            tensor = tensor.to(self.quant_dtype)
        org_tensor_shape = tensor.shape
        tensor = self.reshape_tensor(tensor)
        scales = scales.reshape(-1, 1)
        if not self.symmetric:
            zeros = zeros.reshape(-1, 1)

        if self.round_zeros:
            tensor = (tensor - zeros) * scales
        else:
            tensor = tensor * scales - zeros
        assert torch.isnan(tensor).sum() == 0
        tensor = tensor.reshape(org_tensor_shape).to(torch.bfloat16)
        return tensor


def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x
