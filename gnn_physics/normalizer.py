import torch

from state_objects.base_state_object import BaseStateObject


class AccumulatedNormalizer(BaseStateObject):

    def __init__(self,
                 shape,
                 max_acc_steps=2000,
                 dtype=torch.float64,
                 name='unknown'):
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

    @classmethod
    def static_mode(cls, shape, mean=0.0, std=1.0):
        normalizer = cls(shape)
        normalizer._num_accumulations = normalizer._max_acc_steps
        normalizer._acc_count = 1
        normalizer._acc_sum += mean
        normalizer._acc_sum_squared += std ** 2 + mean ** 2

        return normalizer

    def start_accum(self):
        self.start_accum_flag = True

    def stop_accum(self):
        self.start_accum_flag = False

    def __call__(self, batched_data):
        """Direct transformation of the normalizer."""
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

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self.std_w_eps + self.mean

    def inverse_no_mean(self, normalized_batch_data):
        return normalized_batch_data * self.std_w_eps

    def _safe_max(self, var):
        zero = torch.zeros(var.shape, device=var.device, dtype=var.dtype)
        return torch.maximum(var, zero)

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
        var = self._safe_max(var)
        return torch.sqrt(var)

    @property
    def std_w_eps(self):
        # To use in case the std is too small.
        return torch.maximum(self.std, self._std_epsilon)
