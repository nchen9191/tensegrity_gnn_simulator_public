from typing import Union

import torch

from state_objects.base_state_object import BaseStateObject
from utilities.tensor_utils import zeros


class MotorState(BaseStateObject):
    """
    Class for holding motor states
    """

    def __init__(self):
        super().__init__("motor_state")
        self.omega_t = torch.zeros(1, dtype=self.dtype)  # angular velocity

    def to(self, device: Union[str, torch.device]):
        """
        Method to move tensors between cpus and gpus
        @param device: cpu or gpu
        @return: self
        """
        super().to(device)
        self.omega_t = self.omega_t.to(device)

        return self

    def reset(self):
        """
        Resets the motor angular velocity to zero
        """
        self.omega_t = zeros(1, ref_tensor=self.omega_t)


class DCMotor(BaseStateObject):
    """
    Model of a simple DC motor
    """
    def __init__(self,
                 speed: torch.Tensor):
        super().__init__("motor")
        self.max_omega = torch.tensor(220 * 2 * torch.pi / 60., dtype=self.dtype)
        self.speed = speed
        self.motor_state = MotorState()

    def to(self, device: Union[str, torch.device]):
        """
        Method to move tensors between cpus and gpus
        @param device: cpu or gpu
        @return: self
        """
        super().to(device)

        self.motor_state = self.motor_state.to(device)
        self.speed = self.speed.to(device)
        self.max_omega = self.max_omega.to(device)

        return self

    def compute_cable_length_delta(self,
                                   control: torch.Tensor,
                                   winch_r: torch.Tensor,
                                   dt: torch.Tensor
                                   ) -> torch.Tensor:
        """
        Computes the change in the cable's rest length given control signal
        @param control: control signal between [-1, 1]
        @param winch_r: winch radius of the motor
        @param dt: timestep size

        @return: change in cable rest length
        """
        pre_omega = self.motor_state.omega_t.clone()
        self.motor_state.omega_t = self.speed * self.max_omega * control
        delta_l = (pre_omega + self.motor_state.omega_t) / 2. * winch_r * dt

        return delta_l

    def reset_omega_t(self):
        """
        Resets motor state
        """
        self.motor_state.reset()

