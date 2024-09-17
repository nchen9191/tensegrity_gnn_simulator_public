from typing import List, Optional

import torch

from state_objects.composite_body import CompositeBody
from state_objects.primitive_shapes import Cylinder, SphereState, HollowCylinder
from utilities import torch_quaternion
from utilities.tensor_utils import tensorify


class TensegrityRod(CompositeBody):

    def __init__(self,
                 name: str,
                 end_pts: torch.Tensor,
                 radius: torch.Tensor,
                 mass: torch.Tensor,
                 sphere_radius: torch.Tensor,
                 sphere_mass: torch.Tensor,
                 motor_offset: torch.Tensor,
                 motor_length: torch.Tensor,
                 motor_radius: torch.Tensor,
                 motor_mass: torch.Tensor,
                 linear_vel: torch.Tensor,
                 ang_vel: torch.Tensor,
                 sites: List[str],
                 quat: Optional[torch.Tensor] = None,
                 split_length: Optional[float] = None):
        prin_axis = end_pts[1] - end_pts[0]
        self.length = prin_axis.norm(dim=1, keepdim=True)
        prin_axis /= self.length

        rods = self._init_inner_rods(name,
                                     mass,
                                     end_pts,
                                     linear_vel,
                                     ang_vel,
                                     radius,
                                     split_length)
        endcaps = self._init_endcaps(name,
                                     end_pts,
                                     linear_vel,
                                     ang_vel,
                                     sphere_radius,
                                     sphere_mass,
                                     prin_axis,
                                     quat)
        motors = self._init_motors(name,
                                   (end_pts[1] + end_pts[0]) / 2.,
                                   ang_vel,
                                   linear_vel,
                                   motor_length,
                                   motor_mass,
                                   motor_offset,
                                   motor_radius,
                                   prin_axis,
                                   radius)

        rigid_bodies = rods + endcaps + motors
        self.num_rods = len(rods)
        self.motor_length = motor_length
        self.motor_offset = motor_offset
        self.sphere_radius = sphere_radius
        self.end_pts = end_pts

        if quat is None:
            quat = torch_quaternion.compute_quat_btwn_z_and_vec(prin_axis)

        super().__init__(name,
                         linear_vel,
                         ang_vel,
                         quat,
                         rigid_bodies,
                         sites)

        self.body_verts, self.sphere_idx0, self.sphere_idx1 = (
            self._init_body_verts())

    def to(self, device):
        super().to(device)

        self.motor_length = self.motor_length.to(device)
        self.motor_offset = self.motor_offset.to(device)
        self.sphere_radius = self.sphere_radius.to(device)
        self.length = self.length.to(device)
        self.end_pts = [e.to(device) for e in self.end_pts]

        self.body_verts = self.body_verts.to(device)

        return self

    @classmethod
    def init_from_cfg(cls, cfg):
        cfg_copy = {k: v for k, v in cfg.items()}

        end_pts = tensorify(cfg['end_pts'], reshape=(2, 3, 1))
        cfg_copy['end_pts'] = [end_pts[:1], end_pts[1:]]

        cfg_copy['radius'] = tensorify(cfg['radius'], reshape=(1, 1, 1))
        cfg_copy['mass'] = tensorify(cfg['mass'], reshape=(1, 1, 1))
        cfg_copy['sphere_radius'] = tensorify(cfg['sphere_radius'], reshape=(1, 1, 1))
        cfg_copy['sphere_mass'] = tensorify(cfg['sphere_mass'], reshape=(1, 1, 1))
        cfg_copy['motor_offset'] = tensorify(cfg['motor_offset'], reshape=(1, 1, 1))
        cfg_copy['motor_length'] = tensorify(cfg['motor_length'], reshape=(1, 1, 1))
        cfg_copy['motor_radius'] = tensorify(cfg['motor_radius'], reshape=(1, 1, 1))
        cfg_copy['motor_mass'] = tensorify(cfg['motor_mass'], reshape=(1, 1, 1))
        cfg_copy['linear_vel'] = tensorify(cfg['linear_vel'], reshape=(1, 3, 1))
        cfg_copy['ang_vel'] = tensorify(cfg['ang_vel'], reshape=(1, 3, 1))

        return cls(**cfg_copy)

    def _init_body_verts(self):
        body_verts = []
        sphere0_idx, sphere1_idx = -1, -1
        inv_quat = torch_quaternion.inverse_unit_quat(self.quat)
        for j, body in enumerate(self.rigid_bodies.values()):
            world_vert = self.rigid_bodies[body.name].pos
            body_vert = torch_quaternion.rotate_vec_quat(
                inv_quat,
                world_vert - self.pos
            )
            body_verts.append(body_vert)

            if "sphere0" in body.name:
                sphere0_idx = j
            elif "sphere1" in body.name:
                sphere1_idx = j

        body_verts = torch.vstack(body_verts)

        return body_verts, sphere0_idx, sphere1_idx

    def _init_motors(self,
                     name,
                     pos,
                     ang_vel,
                     linear_vel,
                     motor_length,
                     motor_mass,
                     motor_offset,
                     motor_radius,
                     prin_axis,
                     radius):
        motor_e1_dist = (motor_length / 2 + motor_offset) * prin_axis
        motor_e2_dist = (-motor_length / 2 + motor_offset) * prin_axis
        ang_vel_comp = torch.cross(ang_vel, motor_offset * prin_axis)
        motor0 = HollowCylinder(f'{name}_motor0',
                                [pos - motor_e1_dist, pos - motor_e2_dist],
                                linear_vel - ang_vel_comp,
                                ang_vel.clone(),
                                motor_radius,
                                radius,
                                motor_mass,
                                {})
        motor1 = HollowCylinder(f'{name}_motor1',
                                [pos + motor_e2_dist, pos + motor_e1_dist],
                                linear_vel + ang_vel_comp,
                                ang_vel.clone(),
                                motor_radius,
                                radius,
                                motor_mass,
                                {})
        return [motor0, motor1]

    def _init_endcaps(self,
                      name,
                      end_pts,
                      linear_vel,
                      ang_vel,
                      sphere_radius,
                      sphere_mass,
                      prin_axis,
                      quat):
        endcap0 = SphereState(name + "_sphere0",
                              end_pts[0],
                              linear_vel.clone(),
                              ang_vel.clone(),
                              sphere_radius,
                              sphere_mass,
                              prin_axis,
                              {},
                              quat)
        endcap1 = SphereState(name + "_sphere1",
                              end_pts[1],
                              linear_vel.clone(),
                              ang_vel.clone(),
                              sphere_radius,
                              sphere_mass,
                              prin_axis,
                              {},
                              quat)

        return [endcap0, endcap1]

    def _init_inner_rods(self,
                         name,
                         mass,
                         end_pts,
                         lin_vel,
                         ang_vel,
                         radius,
                         split_length):

        if split_length:
            rod_prin_axis = end_pts[1] - end_pts[0]
            rod_length = rod_prin_axis.norm(keepdim=True)
            rod_prin_axis /= rod_length

            inner_length = split_length
            num_rods = int(rod_length / inner_length)
            outer_length = (rod_length - num_rods * inner_length) / 2.0 + inner_length
            offsets = torch.tensor(([0, outer_length]
                                    + [inner_length] * (num_rods - 2)
                                    + [outer_length]))
            offsets1 = torch.cumsum(offsets[:-1], dim=0)
            offsets2 = torch.cumsum(offsets[1:], dim=0)

            rods = []
            for i in range(num_rods):
                offset1, offset2 = offsets1[i], offsets2[i]
                rod_end_pts = torch.concat([
                    end_pts[0] + offset1 * rod_prin_axis,
                    end_pts[0] + offset2 * rod_prin_axis
                ], dim=-1)

                rod_mass = mass * (offset2 - offset1) / rod_length

                rod = Cylinder(name + f"_rod{i}",
                               rod_end_pts,
                               lin_vel,
                               ang_vel,
                               radius,
                               rod_mass,
                               {})
                rods.append(rod)
        else:
            rods = [
                Cylinder(name + "_rod",
                         end_pts,
                         lin_vel,
                         ang_vel,
                         radius,
                         mass,
                         {})
            ]

        return rods

    def get_template_graph(self):
        template_graph = [
            (f"{self.name}_rod0", f"{self.name}_sphere0"),
            (f"{self.name}_sphere0", f"{self.name}_rod0")
        ]

        for i in range(self.num_rods - 1):
            template_graph.append((f"{self.name}_rod{i}",
                                   f"{self.name}_rod{i + 1}"))
            template_graph.append((f"{self.name}_rod{i + 1}",
                                   f"{self.name}_rod{i}"))

        template_graph.append((f"{self.name}_rod{self.num_rods - 1}",
                               f"{self.name}_sphere1"))
        template_graph.append((f"{self.name}_sphere1",
                               f"{self.name}_rod{self.num_rods - 1}"))

        motor0 = self.rigid_bodies[f"{self.name}_motor0"]
        motor1 = self.rigid_bodies[f"{self.name}_motor1"]

        for i in range(self.num_rods):
            rod_name = f"{self.name}_rod{i}"
            rod = self.rigid_bodies[rod_name]

            for motor in [motor0, motor1]:
                if self._overlap_rods(rod, motor):
                    template_graph.append((rod.name, motor.name))
                    template_graph.append((motor.name, rod.name))

        return template_graph

    def _overlap_rods(self, rod1, rod2):
        # assuming parallel/concentric
        def rod_inside(rod_a, rod_b):
            prin_axis = rod_a.get_principal_axis()
            for end_pt in rod_b.end_pts:
                rel_vec = end_pt - rod_a.end_pts[0]
                length = torch.linalg.vecdot(prin_axis, rel_vec, dim=1)
                if 0 <= length <= rod_a.length:
                    return True
            return False

        return rod_inside(rod1, rod2) or rod_inside(rod2, rod1)

    def update_state(self, pos, linear_vel, quat, ang_vel):
        super().update_state(pos, linear_vel, quat, ang_vel)

        prin_axis = Cylinder.compute_principal_axis(quat)
        self.end_pts = Cylinder.compute_end_pts_from_state(
            self.state,
            prin_axis,
            self.length
        )

    def update_sites(self, site, pos):
        self.sites[site] = pos
