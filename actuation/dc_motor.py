import torch

from state_objects.base_state_object import BaseStateObject
from utilities.tensor_utils import zeros


class MotorState(BaseStateObject):
    def __init__(self):
        super().__init__("motor_state")
        self.omega_t = torch.zeros(1, dtype=self.dtype)  # angular velocity

    def to(self, device):
        super().to(device)
        self.omega_t = self.omega_t.to(device)

        return self

    def reset(self):
        self.omega_t = zeros(1, ref_tensor=self.omega_t)


class DCMotor(BaseStateObject):
    def __init__(self,
                 speed):
        super().__init__("motor")
        self.max_omega = torch.tensor(220 * 2 * torch.pi / 60., dtype=self.dtype)
        self.speed = speed
        self.motor_state = MotorState()

    def to(self, device):
        super().to(device)

        self.motor_state = self.motor_state.to(device)
        self.speed = self.speed.to(device)
        self.max_omega = self.max_omega.to(device)

        return self

    def compute_cable_length_delta(self, control, winch_r, delta_t, dim_scale=1.):
        pre_omega = self.motor_state.omega_t.clone()
        self.motor_state.omega_t = self.speed * self.max_omega * control
        delta_l = (pre_omega + self.motor_state.omega_t) / 2. * winch_r * dim_scale * delta_t

        return delta_l

    def reset_omega_t(self):
        self.motor_state.reset()

