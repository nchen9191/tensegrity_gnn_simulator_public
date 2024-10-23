import torch
from torch.linalg import vecdot, matmul, cross, norm

from linear_contact.collision_detector import *
from state_objects.base_state_object import BaseStateObject
from state_objects.primitive_shapes import Ground
from utilities import torch_quaternion
from utilities.inertia_tensors import body_to_world
from utilities.tensor_utils import zeros


class ContactParameters(BaseStateObject):

    def __init__(self, restitution, baumgarte, friction, friction_damping, rolling_friction=0.0):
        super().__init__("contact_params")
        self.restitution = restitution
        self.baumgarte = baumgarte
        self.friction = friction
        self.friction_damping = friction_damping
        # self.rolling_friction = rolling_friction

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def move_tensors(self, device):
        if isinstance(self.restitution, torch.Tensor):
            self.restitution = self.restitution.to(device)
            self.baumgarte = self.baumgarte.to(device)
            self.friction = self.friction.to(device)
            self.friction_damping = self.friction_damping.to(device)

        return self


class CollisionResponseGenerator(BaseStateObject):

    def __init__(self, ground_z=0):
        super().__init__("collision_response_generator")
        self.ground = Ground(ground_z)
        self.contact_params = ContactParameters(**{
            "restitution": torch.tensor(0.7, dtype=self.dtype),
            "baumgarte": torch.tensor(0.1, dtype=self.dtype),
            "friction": torch.tensor(0.5, dtype=self.dtype),
            "friction_damping": torch.tensor(0.5, dtype=self.dtype)
        })

    def set_contact_params(self, key, contact_params):
        self.contact_params.update(**contact_params)

    def move_tensors(self, device):
        self.ground.move_tensors(device)
        self.contact_params.move_tensors(device)

    def resolve_contact(self, rigid_body1, rigid_body2, next_state1, next_state2,
                        delta_t, collision_detector):
        return self.compute_delta_vel_contact(rigid_body1,
                                              rigid_body2,
                                              next_state1,
                                              next_state2,
                                              delta_t,
                                              collision_detector)

    def resolve_contact_ground(self, rigid_body1, next_state1,
                               delta_t, collision_detector):
        if self.ground.state.shape[0] != next_state1.shape[0]:
            self.ground.repeat_state(next_state1.shape[0])

        return self.compute_delta_vel_contact(self.ground,
                                              rigid_body1,
                                              self.ground.state,
                                              next_state1,
                                              delta_t,
                                              collision_detector)

    def compute_delta_vel_contact(self, rigid_body1, rigid_body2, next_state1, next_state2,
                                  delta_t, collision_detector):
        if delta_t.shape[0] != next_state1.shape[0]:
            repeat = int(next_state1.shape[0] // delta_t.shape[0])
            delta_t = delta_t.repeat(repeat, 1, 1)

        baumgarte, friction_mu, restit_coeff, friction_damping = self.get_contact_params(rigid_body1, rigid_body2)

        toi = zeros((next_state1.shape[0], 1, 1), ref_tensor=next_state1)

        dummy_tensor = zeros((next_state1.shape[0], 3, 1), ref_tensor=next_state1)
        delta_v1, delta_w1 = dummy_tensor.clone(), dummy_tensor.clone()
        delta_v2, delta_w2 = dummy_tensor.clone(), dummy_tensor.clone()
        impulse_pos, impulse_vel, impulse_friction = dummy_tensor.clone(), dummy_tensor.clone(), dummy_tensor.clone()

        # Check current state collision
        curr_state1, curr_state2 = rigid_body1.state, rigid_body2.state
        detection_params_t = collision_detector.detect(curr_state1,
                                                       curr_state2,
                                                       rigid_body1,
                                                       rigid_body2)

        for has_collision_t, contact1_t, contact2_t, normal in detection_params_t:
            if has_collision_t.any():
                curr_state1 = torch.concat([curr_state1[:, :7, ...], next_state1[:, 7:, ...]], dim=1)
                curr_state2 = torch.concat([curr_state2[:, :7, ...], next_state2[:, 7:, ...]], dim=1)
                params = self.compute_contact_params(curr_state1,
                                                     curr_state2,
                                                     rigid_body1,
                                                     rigid_body2,
                                                     has_collision_t,
                                                     contact1_t,
                                                     contact2_t,
                                                     normal)

                mass_norm, mass_tan, rel_vel_norm, rel_vel_tan, tangent, r1, r2, inv_inertia1, inv_inertia2 = params

                # Apply spring-mass
                impulse_pos[has_collision_t] += self.baumgarte_contact_impulse(mass_norm,
                                                                               contact1_t[has_collision_t],
                                                                               contact2_t[has_collision_t],
                                                                               baumgarte,
                                                                               delta_t[has_collision_t],
                                                                               normal[has_collision_t])

                # Apply impulse
                impulse_vel[has_collision_t] = self.reaction_impulse(mass_norm,
                                                                     restit_coeff,
                                                                     rel_vel_norm,
                                                                     normal[has_collision_t])

                # Friction for current time step
                impulse_normal = impulse_vel + impulse_pos
                impulse_friction[has_collision_t] = self.friction_impulse(rel_vel_tan,
                                                                          tangent,
                                                                          impulse_normal[has_collision_t],
                                                                          friction_mu,
                                                                          friction_damping,
                                                                          mass_tan)

                # impulse_total = impulse_normal + impulse_friction
                dv1, dv2, dw1, dw2 = self.compute_delta_vels(impulse_normal[has_collision_t],
                                                             impulse_friction[has_collision_t],
                                                             r1,
                                                             r2,
                                                             rigid_body1.mass,
                                                             rigid_body2.mass,
                                                             inv_inertia1,
                                                             inv_inertia2)
                delta_v1[has_collision_t] += dv1
                delta_v2[has_collision_t] += dv2
                delta_w1[has_collision_t] += dw1
                delta_w2[has_collision_t] += dw2

        has_collision_t = torch.stack([d[0] for d in detection_params_t], dim=-1).max(dim=-1).values

        # Else, Check next state collision
        detection_params_tp = collision_detector.detect(next_state1,
                                                        next_state2,
                                                        rigid_body1,
                                                        rigid_body2)

        for has_collision, contact1_tp, contact2_tp, normal in detection_params_tp:
            has_collision_tp = torch.logical_and(has_collision, ~has_collision_t)

            if has_collision_tp.any():
                params = self.compute_contact_params(next_state1,
                                                     next_state2,
                                                     rigid_body1,
                                                     rigid_body2,
                                                     has_collision_tp,
                                                     contact1_tp,
                                                     contact2_tp,
                                                     normal)
                mass_norm, mass_tan, rel_vel_norm, rel_vel_tan, tangent, r1, r2, inv_inertia1, inv_inertia2 = params

                impulse_vel[has_collision_tp] += self.reaction_impulse(mass_norm,
                                                                       restit_coeff,
                                                                       rel_vel_norm,
                                                                       normal[has_collision_tp])

                # toi
                pen_depth = norm(contact2_tp[has_collision_tp] - contact1_tp[has_collision_tp], dim=1).unsqueeze(2)
                toi[has_collision_tp] = torch.clamp(delta_t[has_collision_tp] + pen_depth / (rel_vel_norm + 1e-12),
                                                    zeros(delta_t[has_collision_tp].shape,
                                                          ref_tensor=delta_t),
                                                    delta_t[has_collision_tp])

                # Friction
                impulse_normal = impulse_vel
                impulse_friction[has_collision_tp] += self.friction_impulse(rel_vel_tan,
                                                                            tangent,
                                                                            impulse_normal[has_collision_tp],
                                                                            friction_mu,
                                                                            friction_damping,
                                                                            mass_tan)

                dv1, dv2, dw1, dw2 = self.compute_delta_vels(impulse_normal[has_collision_tp],
                                                             impulse_friction[has_collision_tp],
                                                             r1,
                                                             r2,
                                                             rigid_body1.mass,
                                                             rigid_body2.mass,
                                                             inv_inertia1,
                                                             inv_inertia2)
                delta_v1[has_collision_tp] += dv1
                delta_v2[has_collision_tp] += dv2
                delta_w1[has_collision_tp] += dw1
                delta_w2[has_collision_tp] += dw2

        return delta_v1, delta_w1, delta_v2, delta_w2, toi

    def get_contact_params(self, rigid_body1, rigid_body2):
        # contact_key = "_".join(sorted([rigid_body1.name, rigid_body2.name]))
        # contact_key = contact_key if contact_key in self.contact_params else "default"

        # restit_coeff = self.contact_params[contact_key]['restitution']
        # baumgarte = self.contact_params[contact_key]['baumgarte']
        # friction_mu = self.contact_params[contact_key]['friction']
        # friction_damping = self.contact_params[contact_key]['friction_damping']
        #
        restit_coeff = self.contact_params.restitution
        baumgarte = self.contact_params.baumgarte
        friction_mu = self.contact_params.friction
        friction_damping = self.contact_params.friction_damping
        # rolling_friction = self.contact_params.rolling_friction

        return baumgarte, friction_mu, restit_coeff, friction_damping

    def compute_delta_vels(self,
                           impulse_normal, impulse_tangent,
                           r1, r2,
                           mass1, mass2,
                           inv_inertia1, inv_inertia2):
        impulse_total = impulse_normal + impulse_tangent

        delta_v1 = -impulse_total / mass1
        delta_v2 = impulse_total / mass2

        delta_w1 = matmul(inv_inertia1, cross(r1, -impulse_total, dim=1))
        delta_w2 = matmul(inv_inertia2, cross(r2, impulse_total, dim=1))

        return delta_v1, delta_v2, delta_w1, delta_w2

    def compute_contact_params(self,
                               state1,
                               state2,
                               rigid_body1,
                               rigid_body2,
                               has_collision,
                               contact1,
                               contact2,
                               normal):
        mass1, mass2 = rigid_body1.mass, rigid_body2.mass

        pos1, pos2 = state1[has_collision, :3, ...], state2[has_collision, :3, ...]
        quat1, quat2 = state1[has_collision, 3:7, ...], state2[has_collision, 3:7, ...]
        vel1, vel2 = state1[has_collision, 7:10, ...], state2[has_collision, 7:10, ...]
        ang_vel1, ang_vel2 = state1[has_collision, 10:, ...], state2[has_collision, 10:, ...]
        contact1, contact2 = contact1[has_collision, :, ...], contact2[has_collision, :, ...]
        normal = normal[has_collision, :, ...]

        rot_mat1 = torch_quaternion.quat_as_rot_mat(quat1)
        rot_mat2 = torch_quaternion.quat_as_rot_mat(quat2)

        r1, r2 = contact1 - pos1, contact2 - pos2
        inertia_inv1 = body_to_world(rot_mat1, rigid_body1.I_body_inv).reshape(-1, 3, 3)
        inertia_inv2 = body_to_world(rot_mat2, rigid_body2.I_body_inv).reshape(-1, 3, 3)
        mass_norm = self.compute_contact_mass(mass1, mass2, inertia_inv1, inertia_inv2, r1, r2, normal)

        rel_vel_c = self.compute_rel_vel(vel1, vel2, ang_vel1, ang_vel2, r1, r2)
        tangent = rel_vel_c - vecdot(rel_vel_c, normal, dim=1).unsqueeze(2) * normal
        tangent /= norm(tangent + 1e-6, dim=1).unsqueeze(2)

        rel_vel_c_norm = self.compute_rel_vel_normal_comp(rel_vel_c, normal)
        rel_vel_c_tan = vecdot(rel_vel_c, tangent, dim=1).unsqueeze(2)

        mass_tan = self.compute_contact_mass(mass1, mass2, inertia_inv1, inertia_inv2, r1, r2, tangent)

        return mass_norm, mass_tan, rel_vel_c_norm, rel_vel_c_tan, tangent, r1, r2, inertia_inv1, inertia_inv2

    @staticmethod
    def reaction_impulse(mass_norm, restit_coeff, rel_vel_c_normal, normal):
        impulse_vel = (-(1 + restit_coeff)
                       * rel_vel_c_normal
                       * mass_norm
                       * normal)

        return impulse_vel

    @staticmethod
    def baumgarte_contact_impulse(mass_norm, contact1, contact2, baumgarte, delta_t, normal):
        pen_depth = norm(contact2 - contact1, dim=1).unsqueeze(2)
        impulse_pos = (baumgarte
                       * pen_depth
                       * mass_norm
                       * normal
                       / delta_t)

        return impulse_pos

    @staticmethod
    def compute_rel_vel(vel1, vel2, ang_vel1, ang_vel2, r1, r2):
        vel_c1 = vel1 + cross(ang_vel1, r1, dim=1)
        vel_c2 = vel2 + cross(ang_vel2, r2, dim=1)
        rel_vel_c = vel_c2 - vel_c1

        return rel_vel_c

    @staticmethod
    def compute_rel_vel_normal_comp(rel_vel, normal):
        v_c_norm = vecdot(rel_vel, normal, dim=1)
        v_c_norm = torch.minimum(v_c_norm, torch.tensor(0, dtype=v_c_norm.dtype, device=v_c_norm.device))

        return v_c_norm.unsqueeze(2)

    @staticmethod
    def compute_contact_mass(mass1, mass2, inv_inertia1, inv_inertia2, r1, r2, dir_vec):
        mass_inv1 = vecdot(cross(matmul(inv_inertia1, cross(r1, dir_vec, dim=1)), r1, dim=1), dir_vec, dim=1)
        mass_inv2 = vecdot(cross(matmul(inv_inertia2, cross(r2, dir_vec, dim=1)), r2, dim=1), dir_vec, dim=1)
        mass_contact = torch.maximum(torch.tensor(0, dtype=inv_inertia1.dtype, device=inv_inertia1.device),
                                     (1 / mass1 + 1 / mass2 + mass_inv1 + mass_inv2) ** -1)

        return mass_contact.reshape(-1, 1, 1)

    @staticmethod
    def friction_impulse(rel_vel_tan, tangent, impulse_normal, friction_mu, friction_damping, mass_tangent):
        static_friction = mass_tangent * rel_vel_tan
        static_friction = static_friction * friction_damping
        # friction = (1 - friction_damping) * mass_tangent * rel_vel_tan
        max_friction = friction_mu * norm(impulse_normal, dim=1, keepdim=True)
        # static_friction = (static_friction
        #                    - max_friction
        #                    + max_friction)

        friction = -torch.minimum(static_friction, max_friction)

        impulse_tangent = tangent * friction

        return impulse_tangent

    @staticmethod
    def friction_impulse2(rel_vel_tan, tangent, normal, impulse_normal, friction_mu, mass, inv_invertia, r, e):
        cotangent = cross(tangent, normal, dim=1)
        # cotangent /= cotangent.norm(dim=1).unsqueeze(2)

        inv_tan = cross(matmul(inv_invertia, cross(r, tangent, dim=1)), r, dim=1)
        inv_cotan = cross(matmul(inv_invertia, cross(r, cotangent, dim=1)), r, dim=1)
        inv_norm = cross(matmul(inv_invertia, cross(r, normal, dim=1)), r, dim=1)

        A11 = 1 / mass + vecdot(inv_tan, tangent, dim=1).unsqueeze(2)
        A21 = vecdot(inv_tan, cotangent, dim=1).unsqueeze(2)
        A12 = vecdot(inv_cotan, tangent, dim=1).unsqueeze(2)
        A22 = 1 / mass + vecdot(inv_cotan, cotangent, dim=1).unsqueeze(2)

        row1 = torch.concat([A11, A12], dim=1)
        row2 = torch.concat([A21, A22], dim=1)
        mat = torch.concat([row1, row2], dim=2)

        impulse_normal_mag = norm(impulse_normal, dim=1).unsqueeze(2)

        rhs = torch.concat([rel_vel_tan + impulse_normal_mag * vecdot(inv_norm, tangent, dim=1).unsqueeze(2),
                            impulse_normal_mag * vecdot(inv_norm, cotangent, dim=1).unsqueeze(2)], dim=1)
        sol = torch.linalg.solve(mat, rhs)
        static_friction_impulse = sol[:, 0:1] * tangent + sol[:, 1:2] * cotangent

        max_friction = friction_mu * norm(impulse_normal, dim=1).unsqueeze(2)
        friction_cond = static_friction_impulse.norm(dim=1).unsqueeze(2) <= max_friction

        impulse_tangent = -static_friction_impulse * friction_cond - max_friction * (~friction_cond) * tangent

        return impulse_tangent

    @staticmethod
    def compute_impulses(rel_vel_tan, rel_vel_norm, tangent, normal, mass, inv_inertia, r, mu, e, mass_norm):
        cotangent = cross(tangent, normal, dim=1)

        func = lambda x, y: vecdot(cross(matmul(inv_inertia, cross(r, x, dim=1)), r, dim=1), y, dim=1).unsqueeze(2)

        A11, A12, A13 = 1 / mass + func(tangent, tangent), func(tangent, cotangent), func(tangent, normal)
        A21, A22, A23 = func(cotangent, tangent), 1 / mass + func(cotangent, cotangent), func(cotangent, normal)
        A31, A32, A33 = func(normal, tangent), func(normal, cotangent), 1 / mass + func(normal, normal)

        row1 = torch.concat([A11, A12, A13], dim=1)
        row2 = torch.concat([A21, A22, A23], dim=1)
        row3 = torch.concat([A31, A32, A33], dim=1)

        lhs = torch.concat([row1, row2, row3], dim=2)
        rhs = torch.concat([rel_vel_tan,
                            zeros(rel_vel_tan.shape, ref_tensor=rel_vel_tan),
                            (1 + e) * rel_vel_norm],
                           dim=1)
        sol = torch.linalg.solve(lhs, rhs)

        impulse_fric = sol[:, 0:1] * tangent + sol[:, 1:2] * cotangent
        impulse_vel = -sol[:, 2:3] * normal

        max_friction = mu * norm(impulse_vel, dim=1).unsqueeze(2)
        friction_cond = impulse_fric.norm(dim=1).unsqueeze(2) <= max_friction

        impulse_fric = -impulse_fric * friction_cond - max_friction * (~friction_cond) * tangent

        if (~friction_cond).any():
            dw = matmul(inv_inertia, cross(r, impulse_fric, dim=1))

            dv_n = vecdot(cross(dw, r, dim=1), normal, dim=1).unsqueeze(2)
            # rel_vel_norm_new = rel_vel_norm - dv_n
            # mass_norm = (1 / mass + vecdot(cross(matmul(inv_inertia, cross(r, normal, dim=1)), r, dim=1), normal, dim=1))**-1
            impulse_vel2 = -((1 + e) * rel_vel_norm + dv_n) * normal * mass_norm
            impulse_vel = impulse_vel * friction_cond + impulse_vel2 * (~friction_cond)

        return impulse_fric, impulse_vel
