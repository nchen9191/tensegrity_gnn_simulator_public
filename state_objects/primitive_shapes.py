from typing import List, Optional, Dict

import torch

from state_objects.rigid_object import RigidBody
from utilities.inertia_tensors import *
from utilities import torch_quaternion, mesh_utils
from utilities.misc_utils import DEFAULT_DTYPE


class Cylinder(RigidBody):

    def __init__(self,
                 name: str,
                 end_pts: Union[torch.Tensor, List],
                 linear_vel: torch.Tensor,
                 ang_vel: torch.Tensor,
                 radius: torch.Tensor,
                 mass: torch.Tensor,
                 sites: Dict):
        self.shape = 'cylinder'

        self._body_verts = None
        self._faces = None

        self.end_pts = end_pts
        if isinstance(end_pts, torch.Tensor):
            self.end_pts = end_pts.reshape(-1, 3, 2)
            self.end_pts = [self.end_pts[:, :, :1], self.end_pts[:, :, 1:2]]

        self.radius = radius
        self.length = (self.end_pts[0] - self.end_pts[1]).norm(dim=1).squeeze()  # compute length

        linear_vel = linear_vel.reshape(-1, 3, 1)
        pos = (self.end_pts[0] + self.end_pts[1]) / 2.0  # compute pos from end points
        ang_vel = ang_vel.reshape(-1, 3, 1)

        # Compute an initial quaternion and rotation matrix
        prin = self.end_pts[1] - self.end_pts[0]
        q = torch_quaternion.compute_quat_btwn_z_and_vec(prin)

        I_body = cylinder_body(mass, self.length, self.radius, DEFAULT_DTYPE)

        super().__init__(name,
                         mass,
                         I_body,
                         pos,
                         q,
                         linear_vel,
                         ang_vel,
                         sites)

    def to(self, device):
        super(Cylinder, self).to(device)
        self.radius = self.radius.to(device)
        self.length = self.length.to(device)

        self.end_pts[0] = self.end_pts[0].to(device)
        self.end_pts[1] = self.end_pts[1].to(device)

        return self

    def get_principal_axis(self):
        """
        Method to get principal axis
        :return:
        """
        # z-axis aligned with rod cylinder axis along length
        return self.rot_mat[..., :, 2:]

    def _compute_end_pts(self) -> List[torch.Tensor]:
        """
        Internal method to compute end points
        :return: End point tensors
        """
        principle_axis_vec = self.get_principal_axis()
        end_pts = self.compute_end_pts_from_state(self.pos[..., :3],
                                                  principle_axis_vec,
                                                  self.length)
        # end_pts = torch.concat(end_pts, dim=-1)

        return end_pts

    @staticmethod
    def compute_principal_axis(quat):
        """
        Computes principal axis from state input

        :param state: State (pos1, pos2, pos3, q1, q2, q3, q4, lin_v1, lin_v2, lin_v3, ang_v1, ang_v2, ang_v3)
        :return: principal axises
        """
        return torch_quaternion.quat_as_rot_mat(quat)[..., :, 2:]

    @staticmethod
    def compute_end_pts_from_state(rod_pos_state, principal_axis, rod_length):
        """
        :param rod_pos_state: (x, y, z, quat.w, quat.x, quat.y, quat.z)
        :param principal_axis: tensor of vector(s)
        :param rod_length: length of rod
        :return: ((x1, y1, z1), (x2, y2, z2))
        """
        # Get position
        pos = rod_pos_state[:, :3]

        # Compute half-length vector from principal axis
        half_length_vec = rod_length * principal_axis / 2

        # End points are +/- of half-length vector from COM
        end_pt1 = pos - half_length_vec
        end_pt2 = pos + half_length_vec

        return [end_pt1, end_pt2]

    def update_state(self, pos, linear_vel, rot_val, ang_vel):
        super().update_state(pos, linear_vel, rot_val, ang_vel)
        self.end_pts = self._compute_end_pts()


class HollowCylinder(Cylinder):
    def __init__(self,
                 name,
                 end_pts,
                 linear_vel,
                 ang_vel,
                 outer_radius,
                 inner_radius,
                 mass,
                 sites):
        self.shape = 'cylinder'

        super().__init__(name,
                         end_pts,
                         linear_vel,
                         ang_vel,
                         outer_radius,
                         mass,
                         sites)

        self.inner_radius = inner_radius

        self.I_body = hollow_cylinder_body(mass,
                                           self.length,
                                           outer_radius,
                                           inner_radius)

        self.I_body_inv = torch.linalg.inv(self.I_body)


class SphereState(RigidBody):

    def __init__(self,
                 name: str,
                 center: torch.Tensor,
                 linear_vel: Optional[torch.Tensor],
                 ang_vel: Optional[torch.Tensor],
                 radius: torch.Tensor,
                 mass: torch.Tensor,
                 principal_axis: Optional[torch.Tensor],
                 sites: Dict,
                 rot_val: Optional[torch.Tensor] = None):
        self.shape = 'sphere'

        self.radius = radius

        linear_vel = linear_vel.reshape(-1, 3, 1)
        ang_vel = ang_vel.reshape(-1, 3, 1)

        if rot_val is None:
            rot_val = torch_quaternion.compute_quat_btwn_z_and_vec(principal_axis)

        self._body_verts = None
        self._faces = None

        super().__init__(name,
                         mass,
                         solid_sphere_body(mass, self.radius, DEFAULT_DTYPE),
                         center,
                         rot_val,
                         linear_vel,
                         ang_vel,
                         sites)

    def signed_dist_fn(self, pt):
        if len(pt.shape) == 2:
            pt = pt.unsqueeze(-1)

        rel_pt = pt - self.pos
        sdf = rel_pt.norm(dim=1, keepdim=True) - self.radius

        return sdf

    def proj_surface_pt(self, pt):
        rel_pt = pt - self.pos
        dir_vec = rel_pt / torch.linalg.norm(rel_pt, dim=1)
        surface_pt = self.pos + dir_vec * self.radius

        return surface_pt

    def to(self, device):
        super(SphereState, self).to(device)
        self.radius = self.radius.to(device)

        if self._body_verts is not None:
            self._body_verts = self._body_verts.to(device)
            self._faces = self._faces.to(device)

        return self


class Ground(RigidBody):

    def __init__(self, ground_z=0.0, sys_precision=DEFAULT_DTYPE):
        self.shape = "ground"

        mass = torch.tensor(torch.inf, dtype=sys_precision)
        I_body = torch.diag(torch.tensor([torch.inf, torch.inf, torch.inf], dtype=sys_precision))
        pos = torch.tensor([0, 0, ground_z], dtype=sys_precision).reshape(1, -1, 1)
        quat = torch.tensor([1, 0, 0, 0], dtype=sys_precision).reshape(1, -1, 1)
        linear_vel = torch.tensor([0, 0, 0], dtype=sys_precision).reshape(1, -1, 1)
        ang_vel = torch.tensor([0, 0, 0], dtype=sys_precision).reshape(1, -1, 1)

        super().__init__("ground", mass, I_body, pos, quat, linear_vel, ang_vel, [])

    def repeat_state(self, batch_size):
        self.reset_batch_size()

        self.pos = self.pos.repeat(batch_size, 1, 1)
        self.quat = self.quat.repeat(batch_size, 1, 1)
        self.linear_vel = self.linear_vel.repeat(batch_size, 1, 1)
        self.ang_vel = self.ang_vel.repeat(batch_size, 1, 1)

    def reset_batch_size(self):
        self.pos = self.pos[0:1]
        self.quat = self.quat[0:1]
        self.linear_vel = self.linear_vel[0:1]
        self.ang_vel = self.ang_vel[0:1]
