import torch

from state_objects.base_state_object import BaseStateObject
from utilities.tensor_utils import zeros


class PID(BaseStateObject):
    def __init__(self,
                 k_p=6.0,
                 k_i=0.01,
                 k_d=0.5,
                 min_length=100,
                 RANGE=100,
                 tol=0.15):
        super().__init__('pid')
        # self.last_control = np.zeros(n_motor)
        self.last_error = None
        self.cum_error = None
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.min_length = min_length / 100.
        self.RANGE = RANGE / 100.
        self.tol = tol
        self.LEFT_RANGE = None
        self.RIGHT_RANGE = None
        self.done = None

    def move_tensors(self, device):
        if self.last_error is not None:
            self.last_error = self.last_error.to(device)

        if self.cum_error is not None:
            self.cum_error = self.cum_error.to(device)

        return self

    def update_control_target_length(self, current_length, target_length):
        if self.cum_error is None:
            self.cum_error = zeros(current_length.shape, ref_tensor=current_length)

        u = zeros(current_length.shape, ref_tensor=current_length)
        RANGE = 1.0
        diff = current_length - target_length
        error = diff / RANGE

        high_error = torch.abs(error) >= 0.05
        d_error = zeros(current_length.shape, ref_tensor=current_length) \
            if self.last_error is None else error - self.last_error
        self.cum_error += error
        self.last_error = error

        u[high_error] = (self.k_p * error[high_error]
                         + self.k_i * self.cum_error[high_error]
                         + self.k_d * d_error[high_error])
        u = torch.clip(u, min=-1, max=1)

        return u

    def update_control_by_target_gait(self, current_length, target_gait, rest_length):
        if self.done is None:
            self.done = torch.tensor([False] * current_length.shape[0],
                                     device=current_length.device)

        if self.cum_error is None:
            self.cum_error = zeros(current_length.shape,
                                   ref_tensor=current_length)

        u = zeros(current_length.shape,
                  ref_tensor=current_length)

        min_length = self.min_length
        range_ = torch.clamp_min(self.RANGE, 1e-5)

        position = (current_length - min_length) / range_

        # if self.done:
        #     return u, position

        target_length = min_length + range_ * target_gait
        error = position - target_gait

        low_error_cond1 = torch.abs(error).flatten() < self.tol.flatten()
        low_error_cond2 = torch.abs(current_length - target_length).flatten() < 0.1
        low_error_cond3 = torch.logical_and(target_gait.flatten() == 0, position.flatten() < 0)

        low_error = torch.logical_or(
            torch.logical_or(self.done, low_error_cond1),
            torch.logical_or(low_error_cond2, low_error_cond3)
        )

        self.done[low_error] = True

        d_error = zeros(error.shape, ref_tensor=error) \
            if self.last_error is None else error - self.last_error
        self.cum_error += error
        self.last_error = error

        u[~low_error] = (self.k_p * error[~low_error]
                         + self.k_i * self.cum_error[~low_error]
                         + self.k_d * d_error[~low_error])

        u = torch.clip(u, min=-1, max=1)

        slack = torch.logical_and(current_length < rest_length,
                                  u < 0)
        u[slack] = 0

        return u, position

    def compute_ctrl_target_gait(self,
                                 position,
                                 min_length,
                                 range_,
                                 target_gait,
                                 ):
        u = zeros(position.shape, ref_tensor=position)

        if self.done:
            return u

        target_length = min_length + range_ * target_gait
        current_length = min_length + range_ * position
        error = position - target_gait

        low_error = (abs(error) < self.tol
                     or abs(current_length - target_length) < 0.1
                     or (target_gait == 0 and position < 0))

        if low_error.all():
            self.done = True

        if self.cum_error is None:
            self.cum_error = zeros(current_length.shape,
                                   ref_tensor=current_length)

        d_error = zeros(error.shape, ref_tensor=error) \
            if self.last_error is None else error - self.last_error
        self.cum_error += error
        self.last_error = error
        try:
            u[~low_error] = (self.k_p * error[~low_error]
                             + self.k_i * self.cum_error[~low_error]
                             + self.k_d * d_error[~low_error])
        except:
            ignore = 0

        u = torch.clip(u, min=-1, max=1)

        if u.all() == 0.0:
            self.done = True

        return u

    def reset(self):
        self.last_error = None
        self.cum_error = None
        self.done = None

    def set_range(self, RANGE):
        self.LEFT_RANGE = RANGE[0] / 100.
        self.RIGHT_RANGE = RANGE[1] / 100.
        # self.RANGE = RANGE / 100.

    def set_min_length(self, min_length):
        self.min_length = min_length / 100.
