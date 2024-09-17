from typing import Union, Dict, List

import torch

from state_objects.base_state_object import BaseStateObject
from utilities import torch_quaternion
from utilities.inertia_tensors import body_to_world


class RigidBody(BaseStateObject):

    def __init__(self,
                 name: str,
                 mass: Union[float, int, torch.Tensor],
                 I_body: torch.Tensor,
                 pos: torch.Tensor,
                 quat: torch.Tensor,
                 linear_vel: torch.Tensor,
                 ang_vel: torch.Tensor,
                 sites: List[str]):
        super().__init__(name)

        self.mass = mass
        self.I_body = I_body
        self.I_body_inv = torch.linalg.inv(I_body)

        self.pos = pos

        self.linear_vel = linear_vel
        self.ang_vel = ang_vel

        self.quat = quat

        self.sites = {s: None for s in sites}

    def to(self, device):
        super().to(device)

        self.mass = self.mass.to(device)
        self.I_body = self.I_body.to(device)
        self.I_body_inv = self.I_body_inv.to(device)

        self.pos = self.pos.to(device)
        self.quat = self.quat.to(device)
        self.linear_vel = self.linear_vel.to(device)
        self.ang_vel = self.ang_vel.to(device)

        for k, v in self.sites.items():
            if isinstance(v, torch.Tensor):
                self.sites[k] = v.to(device)

        return self

    def world_to_body_coords(self, world_coords):
        rot_mat_inv = self.rot_mat.transpose(1, 2)
        return rot_mat_inv @ (world_coords - self.pos)

    def body_to_world_coords(self, body_coords):
        return (self.rot_mat @ body_coords) + self.pos

    def update_state(self, pos, linear_vel, quat, ang_vel):
        self.pos = pos
        self.linear_vel = linear_vel
        self.ang_vel = ang_vel
        self.quat = quat

    @property
    def state(self):
        return torch.hstack([
            self.pos,
            self.quat,
            self.linear_vel,
            self.ang_vel
        ])

    @property
    def rot_mat(self):
        return torch_quaternion.quat_as_rot_mat(self.quat)

    @property
    def I_world_inv(self):
        return body_to_world(
            self.rot_mat,
            self.I_body_inv
        ).reshape(-1, 3, 3)
