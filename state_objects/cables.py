from typing import Union, List, Tuple

import torch

from actuation.dc_motor import DCMotor
from state_objects.base_state_object import BaseStateObject
from utilities.tensor_utils import zeros, tensorify


class Spring(BaseStateObject):

    def __init__(self,
                 stiffness: torch.Tensor,
                 damping: torch.Tensor,
                 rest_length: torch.Tensor,
                 end_pts: Union[List, Tuple],
                 name: str):
        """
        :param stiffness: spring stiffness
        :param damping: spring damping coefficient
        :param rest_length: spring rest length
        :param end_pts: (end_pt1 site_name, end_pt2 site_name), site names should match whats in system topology
        :param name: unique name
        """
        super().__init__(name)

        self.stiffness = stiffness
        self.damping = damping
        self._rest_length = rest_length
        self.end_pts = end_pts

    @classmethod
    def init_from_cfg(cls, cfg):
        cfg_copy = {k: v for k, v in cfg.items()}

        cfg_copy['stiffness'] = tensorify(cfg['stiffness'], reshape=(1, 1, 1))
        cfg_copy['damping'] = tensorify(cfg['damping'], reshape=(1, 1, 1))
        cfg_copy['rest_length'] = tensorify(cfg['rest_length'], reshape=(1, 1, 1))

        return cls(**cfg_copy)

    def to(self, device):
        super().to(device)

        self.stiffness = self.stiffness.to(device)
        self.damping = self.damping.to(device)
        self._rest_length = self._rest_length.to(device)

        return self

    @property
    def rest_length(self):
        return self._rest_length

    def compute_curr_length(self, end_pt1, end_pt2):
        spring_pos_vec = end_pt2 - end_pt1
        spring_pos_len = spring_pos_vec.norm(dim=1, keepdim=True)

        return spring_pos_len

    def compute_force(self,
                      end_pt1: torch.Tensor,
                      end_pt2: torch.Tensor,
                      vel_1: torch.Tensor,
                      vel_2: torch.Tensor) -> torch.Tensor:
        """
        Computes a spring force with the equation F = stiffness * (curr len - rest len) - damping * relative velocity
        Force direction relative to (endpt2 - endpt1) vector

        :param end_pt1: One end point
        :param end_pt2: Other end point
        :param vel_1: Velocity of end_pt1
        :param vel_2: Velocity of end_pt2
        :return: Spring's force
        """
        # Compute spring direction
        spring_pos_vec = end_pt2 - end_pt1
        spring_pos_len = spring_pos_vec.norm(dim=1, keepdim=True)
        spring_pos_vec_unit = spring_pos_vec / spring_pos_len

        # Compute spring velocity
        rel_vel_1 = torch.linalg.vecdot(vel_1, spring_pos_vec_unit, dim=1).unsqueeze(2)
        rel_vel_2 = torch.linalg.vecdot(vel_2, spring_pos_vec_unit, dim=1).unsqueeze(2)

        # Compute spring force based on hooke's law and damping
        stiffness_mag = self.stiffness * (spring_pos_len - self.rest_length)

        damping_mag = self.damping * (rel_vel_1 - rel_vel_2)
        spring_force_mag = stiffness_mag - damping_mag

        spring_force = spring_force_mag * spring_pos_vec_unit

        return spring_force


class Cable(Spring):
    def compute_force(self,
                      end_pt1: torch.Tensor,
                      end_pt2: torch.Tensor,
                      vel_1: torch.Tensor,
                      vel_2: torch.Tensor,
                      pull_only=True) -> torch.Tensor:
        """
        Computes a spring force with the equation F = stiffness * (curr len - rest len) - damping * relative velocity
        Force direction relative to (endpt2 - endpt1) vector

        :param end_pt1: One end point
        :param end_pt2: Other end point
        :param vel_1: Velocity of end_pt1
        :param vel_2: Velocity of end_pt2
        :return: Spring's force
        """
        # Compute spring direction
        spring_pos_vec = end_pt2 - end_pt1
        spring_pos_len = spring_pos_vec.norm(dim=1, keepdim=True)
        spring_pos_vec_unit = spring_pos_vec / spring_pos_len

        # Compute spring velocity
        rel_vel_1 = torch.linalg.vecdot(vel_1, spring_pos_vec_unit, dim=1).unsqueeze(2)
        rel_vel_2 = torch.linalg.vecdot(vel_2, spring_pos_vec_unit, dim=1).unsqueeze(2)

        # Compute spring force based on hooke's law and damping
        stiffness_mag = self.stiffness * (spring_pos_len - self.rest_length)
        damping_mag = self.damping * (rel_vel_1 - rel_vel_2)

        if pull_only:
            stiffness_mag = torch.clamp_min(stiffness_mag, 0.0)
            # damping_mag = torch.clamp_max(damping_mag, 0.0)

        spring_force_mag = stiffness_mag - damping_mag

        spring_force = spring_force_mag * spring_pos_vec_unit

        return spring_force


class ActuatedCable(Cable):

    def __init__(self,
                 stiffness,
                 damping,
                 rest_length,
                 end_pts,
                 name,
                 winch_r,
                 min_winch_r=0.01,
                 max_winch_r=0.07,
                 sys_precision=torch.float64,
                 motor=None,
                 motor_speed=0.6,
                 init_act_length=0.0):
        super().__init__(stiffness,
                         damping,
                         rest_length,
                         end_pts,
                         name)
        motor_speed = torch.tensor(motor_speed, dtype=sys_precision)
        self.motor = DCMotor(motor_speed) if motor is None else motor
        self.init_act_length = torch.tensor(init_act_length, dtype=sys_precision)
        self.actuation_length = self.init_act_length.clone().reshape(1, 1, 1)
        self.min_winch_r = torch.tensor(min_winch_r, dtype=sys_precision)
        self.max_winch_r = torch.tensor(max_winch_r, dtype=sys_precision)
        self._winch_r = self._set_winch_r(winch_r)

    def _set_winch_r(self, winch_r):
        assert self.min_winch_r <= winch_r <= self.max_winch_r

        if not isinstance(winch_r, torch.Tensor):
            winch_r = torch.tensor(winch_r, dtype=self.dtype)

        delta = self.max_winch_r - self.min_winch_r
        winch_r = torch.logit((winch_r - self.min_winch_r) / delta)

        return winch_r

    def to(self, device):
        super().to(device)

        self.motor = self.motor.to(device)
        self.actuation_length = self.actuation_length.to(device)
        self.init_act_length = self.init_act_length.to(device)
        self._winch_r = self._winch_r.to(device)
        self.min_winch_r = self.min_winch_r.to(device)
        self.max_winch_r = self.max_winch_r.to(device)

        return self

    @property
    def winch_r(self):
        winch_r_range = self.max_winch_r - self.min_winch_r
        dwinch_r = torch.sigmoid(self._winch_r) * winch_r_range

        winch_r = dwinch_r + self.min_winch_r
        return winch_r

    @property
    def rest_length(self):
        if self.actuation_length is None:
            return self._rest_length

        rest_length = self._rest_length - self.actuation_length
        return rest_length

    def update_rest_length(self,
                           control,
                           cable_length,
                           dt):
        if self.actuation_length is None:
            self.actuation_length = zeros(cable_length.shape,
                                          ref_tensor=cable_length)
        dl = self.motor.compute_cable_length_delta(control,
                                                   self.winch_r,
                                                   dt)
        self.actuation_length = self.actuation_length + dl * self.rest_length / cable_length
        self.actuation_length = torch.clamp_max(self.actuation_length,
                                                self._rest_length)

    def reset_cable(self):
        self.actuation_length = self.init_act_length.clone()
        self.motor.reset_omega_t()

    def compute_force(self,
                      end_pt1: torch.Tensor,
                      end_pt2: torch.Tensor,
                      vel_1: torch.Tensor,
                      vel_2: torch.Tensor,
                      pull_only=True) -> torch.Tensor:
        spring_force = super().compute_force(end_pt1,
                                             end_pt2,
                                             vel_1,
                                             vel_2,
                                             True)
        return spring_force


def get_cable(spring_type):
    if spring_type.lower() == 'cable':
        return Cable
    elif spring_type.lower() == 'actuated_cable':
        return ActuatedCable
    else:
        return Spring
