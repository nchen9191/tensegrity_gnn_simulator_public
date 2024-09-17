from collections import OrderedDict

import torch

from state_objects.rigid_object import RigidBody
from utilities import torch_quaternion, tensor_utils
from utilities.inertia_tensors import parallel_axis_offset


class CompositeBody(RigidBody):

    def __init__(self,
                 name,
                 linear_vel,
                 ang_vel,
                 quat,
                 rigid_bodies,
                 sites):
        self.inner_bodies_updated = True
        self._rigid_bodies = self.init_rigid_bodies_dict(rigid_bodies)

        mass = sum([
            body.mass
            for body in self._rigid_bodies.values()
        ])

        com = torch.stack(
            [body.pos * body.mass
             for body in self._rigid_bodies.values()],
            dim=-1
        ).sum(dim=-1) / mass

        I_body = self._compute_inertia_tensor(com, quat)

        self._rigid_bodies_body_vecs = self._compute_body_vecs(com, quat)
        self.body_vecs_tensor = torch.vstack(
            list(self._rigid_bodies_body_vecs.values())
        )

        super().__init__(name,
                         mass,
                         I_body,
                         com,
                         quat,
                         linear_vel,
                         ang_vel,
                         sites)

    def update_state(self, pos, linear_vel, rot_val, ang_vel):
        super().update_state(pos, linear_vel, rot_val, ang_vel)
        self.inner_bodies_updated = False

    def init_rigid_bodies_dict(self, rigid_bodies):
        rigid_body_dict = OrderedDict()
        for body in rigid_bodies:
            rigid_body_dict[body.name] = body

        return rigid_body_dict

    def _compute_inertia_tensor(self, com, quat):
        rot_mat = torch_quaternion.quat_as_rot_mat(quat)
        rot_mat_inv = rot_mat.transpose(1, 2)
        I_body_total = torch.zeros((3, 3), dtype=com.dtype)

        for body in self.rigid_bodies.values():
            offset_world = body.pos - com
            offset_body = torch.matmul(rot_mat_inv, offset_world)
            I_body = parallel_axis_offset(body.I_body, body.mass, offset_body)
            I_body_total += I_body[0]

        I_body_total = torch.diag(torch.diag(I_body_total, 0))
        return I_body_total

    def compute_body_offset_inertia(self, body_name):
        body = self.rigid_bodies[body_name]

        offset_body = self.rigid_bodies_body_vecs[body_name]
        I_body = parallel_axis_offset(body.I_body, body.mass, offset_body)

        return I_body

    def _compute_body_vecs(self, com, quat):
        body_vecs = {}
        rot_mat = torch_quaternion.quat_as_rot_mat(quat)
        rot_mat_inv = rot_mat.transpose(1, 2)

        for body in self.rigid_bodies.values():
            offset_world = body.pos - com
            offset_body = torch.matmul(rot_mat_inv, offset_world)
            body_vecs[body.name] = offset_body

        return body_vecs

    def to(self, device):
        super().to(device)
        for k, v in self._rigid_bodies_body_vecs.items():
            self._rigid_bodies_body_vecs[k] = v.to(device)

        for body in self._rigid_bodies.values():
            body.to(device)

        return self

    @property
    def rigid_bodies(self):
        if not self.inner_bodies_updated:
            for name, body in self._rigid_bodies.items():
                body_vec = self._rigid_bodies_body_vecs[name]
                world_vec = torch.matmul(self.rot_mat, body_vec)
                world_vec += self.pos
                lin_vel = self.linear_vel + torch.cross(self.ang_vel, body_vec, dim=1)
                body.update_state(world_vec, lin_vel, self.quat, self.ang_vel)
            self.inner_bodies_updated = True

        return self._rigid_bodies

    @property
    def rigid_bodies_body_vecs(self):
        return self._rigid_bodies_body_vecs