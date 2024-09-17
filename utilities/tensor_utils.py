import torch

from utilities.misc_utils import DEFAULT_DTYPE


def zeros(shape, dtype=None, device=None, ref_tensor=None):
    if ref_tensor is None and (dtype is None or device is None):
        raise Exception("Need to specify either ref tensor or (dtype and device)")

    if ref_tensor is not None:
        dtype = ref_tensor.dtype
        device = ref_tensor.device

    return torch.zeros(shape, dtype=dtype, device=device)


def ones(shape, dtype=None, device=None, ref_tensor=None):
    if ref_tensor is None and (dtype is None or device is None):
        raise Exception("Need to specify either ref tensor or (dtype and device)")

    if ref_tensor is not None:
        dtype = ref_tensor.dtype
        device = ref_tensor.device

    return torch.ones(shape, dtype=dtype, device=device)


def tensorify(non_tensor, dtype=None, reshape=None):
    dtype = dtype if dtype else DEFAULT_DTYPE
    ten = torch.tensor(non_tensor, dtype=dtype)
    reshape = reshape if reshape else ten.shape
    return ten.reshape(reshape)


def safe_norm(tensor, dim=1):
    return tensor / torch.clamp_min(tensor.norm(dim=dim, keepdim=True), 1e-8)
