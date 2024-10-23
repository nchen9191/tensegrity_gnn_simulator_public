from typing import Tuple

import torch

from state_objects.base_state_object import BaseStateObject


class AccumulatedNormalizer(BaseStateObject):

    def __init__(self,
                 shape: Tuple,
                 max_acc_steps: int = 2000,
                 dtype: torch.dtype = torch.float64,
                 name: str = 'unknown'):
        """
        Normalizer that accumulates during first epoch to compute mean and std of features

        @param shape: shape of feature
        @param max_acc_steps: max number of accumulation steps
        @param dtype: data type for torch tensors
        @param name: feature name
        """
        super().__init__('normalizer')
        zeros = torch.zeros(shape, dtype=dtype, device=self.device)

        self.start_accum_flag = False
        self.name = name

        self._max_acc_steps = max_acc_steps
        self._num_accumulations = 0
        self._acc_count = 0
        self._acc_sum = zeros.clone()
        self._acc_sum_squared = zeros.clone()

        self._std_epsilon = zeros + 1e-3

    def start_accum(self):
        self.start_accum_flag = True

    def stop_accum(self):
        self.start_accum_flag = False

    def __call__(self, batched_data: torch.Tensor) -> torch.Tensor:
        """
        normal function
        @param batched_data:
        @return:
        """
        if self.start_accum_flag and self._num_accumulations < self._max_acc_steps:
            self._num_accumulations += 1
            self._acc_count += batched_data.shape[0]
            self._acc_sum += batched_data.detach().sum(dim=0, keepdim=True)
            self._acc_sum_squared += (batched_data.detach() ** 2).sum(dim=0, keepdim=True)

        normalized = (batched_data - self.mean) / self.std_w_eps

        return normalized

    def to(self, device):
        super().to(device)
        self._acc_sum = self._acc_sum.to(device)
        self._acc_sum_squared = self._acc_sum_squared.to(device)
        self._std_epsilon = self._std_epsilon.to(device)

        return self

    def inverse(self, normalized_batch_data: torch.Tensor) -> torch.Tensor:
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self.std_w_eps + self.mean

    def inverse_no_mean(self, normalized_batch_data: torch.Tensor) -> torch.Tensor:
        return normalized_batch_data * self.std_w_eps

    @property
    def _safe_count(self):
        # To ensure count is at least one and avoid nan's.
        return max(self._acc_count, 1)

    @property
    def mean(self):
        return self._acc_sum / self._safe_count

    @property
    def std(self):
        var = self._acc_sum_squared / self._safe_count - self.mean ** 2
        var = torch.clamp_min(var, 0.)
        return torch.sqrt(var)

    @property
    def std_w_eps(self):
        # To use in case the std is too small.
        return torch.maximum(self.std, self._std_epsilon)
