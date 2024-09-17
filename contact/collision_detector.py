import torch

from state_objects.composite_body import CompositeBody
from utilities import torch_quaternion


class CollisionDetector:

    @staticmethod
    def detect(state1, state2, rigid_body1, rigid_body2):
        """
        Normal standardized to always pointing from rigid_body1 to rigid_body2
        :param state1:
        :param state2:
        :param rigid_body1:
        :param rigid_body2:
        :return:
        """
        raise NotImplementedError


class GroundSphereDetector(CollisionDetector):

    @staticmethod
    def detect(ground_state, sphere_state, ground, sphere):
        pos = sphere_state[:, :3, ...]
        ground_z = ground_state[:, 2, ...]

        normal = torch.tensor(
            [0, 0, 1],
            dtype=pos.dtype,
            device=pos.device
        ).reshape(1, 3, 1).repeat(pos.shape[0], 1, 1)
        radius = normal * sphere.radius

        min_pts = pos - radius
        ground_contacts = min_pts.clone()
        ground_contacts[:, 2, :] = ground_z

        has_collision = min_pts[:, 2] <= ground_z

        return [(has_collision.flatten(), ground_contacts, min_pts, normal)]


class SphereSphereDetector(CollisionDetector):

    @staticmethod
    def detect(state1, state2, rigid_body1, rigid_body2):
        pos_sphere1 = state1[:, :3]
        pos_sphere2 = state2[:, :3]

        diff = pos_sphere2 - pos_sphere1
        dists = torch.linalg.norm(diff, dim=1).unsqueeze(2)

        normal = diff / dists
        contact_pts1 = pos_sphere1 + rigid_body1.radius * normal
        contact_pts2 = pos_sphere2 - rigid_body2.radius * normal

        has_collision = dists <= rigid_body1.radius + rigid_body2.radius

        return [(has_collision.flatten(), contact_pts1, contact_pts2, normal)]


class CylinderSphereDetector(CollisionDetector):

    @staticmethod
    def detect(cylinder_state, sphere_state, cylinder, sphere, check_faces=False):
        rot_mat = torch_quaternion.quat_as_rot_mat(cylinder_state[:, 3:7, ...])
        principal_axis = rot_mat[:, :, 2:]
        end_pt1, end_pt2 = cylinder.compute_end_pts_from_state(cylinder_state, principal_axis, cylinder.length)

        sphere_pos = sphere_state[:, :3, ...]
        sphere_pos_rel = sphere_pos - end_pt1
        sphere_pos_proj_mag = torch.linalg.vecdot(sphere_pos_rel, principal_axis, dim=1).unsqueeze(2)
        sphere_pos_proj = sphere_pos_proj_mag * principal_axis

        check_prin_axis = (0 <= sphere_pos_proj_mag) * (sphere_pos_proj_mag <= 1)
        check_face1 = check_faces * (sphere_pos_proj_mag < 0.0)
        check_face2 = check_faces * (sphere_pos_proj_mag > 1.0)

        prin_check_params = CylinderSphereDetector.check_outer_face(sphere_pos,
                                                                    sphere_pos_rel,
                                                                    sphere_pos_proj,
                                                                    sphere,
                                                                    cylinder,
                                                                    check_prin_axis,
                                                                    end_pt1)

        # end_pt1_params = CylinderSphereDetector.check_endface(sphere_pos,
        #                                                       sphere_pos_proj_mag,
        #                                                       sphere_pos_rel,
        #                                                       sphere,
        #                                                       cylinder,
        #                                                       principal_axis,
        #                                                       end_pt1,
        #                                                       check_face1)
        #
        # end_pt2_params = CylinderSphereDetector.check_endface(sphere_pos,
        #                                                       sphere_pos_proj_mag,
        #                                                       sphere_pos_rel,
        #                                                       sphere,
        #                                                       cylinder,
        #                                                       principal_axis,
        #                                                       end_pt2,
        #                                                       check_face2)
        #
        # has_collision, contact_pts_c, contact_pts_s, normal = CylinderSphereDetector.combine_checks(prin_check_params,
        #                                                                                             end_pt1_params,
        #                                                                                             end_pt2_params)

        return [prin_check_params]
        # return [(has_collision.flatten(), contact_pts_c, contact_pts_s, normal)]

    @staticmethod
    def combine_checks(prin_params, endpt1_params, endpt2_params):
        has_collision_prin, contact_pts_cylinder_prin, contact_pts_sphere_prin, normal_prin_axis = prin_params
        has_collision1, contact_pts_cylinder1, contact_pts_sphere1, normal1 = endpt1_params
        has_collision2, contact_pts_cylinder2, contact_pts_sphere2, normal2 = endpt2_params

        normal = normal_prin_axis + normal1 + normal2
        contact_pts_cylinder = contact_pts_cylinder_prin + contact_pts_cylinder1 + contact_pts_cylinder2
        contact_pts_sphere = contact_pts_sphere_prin + contact_pts_sphere1 + contact_pts_sphere2
        has_collision = torch.logical_or(has_collision_prin, torch.logical_or(has_collision1, has_collision2))

        return has_collision, contact_pts_cylinder, contact_pts_sphere, normal

    @staticmethod
    def check_outer_face(sphere_pos, sphere_pos_rel, sphere_pos_proj, sphere, cylinder, check_prin_axis, end_pt):
        normal_prin_axis = sphere_pos_rel - sphere_pos_proj
        dist_prin_axis = torch.linalg.norm(normal_prin_axis, dim=1).unsqueeze(2)
        normal_prin_axis /= dist_prin_axis
        has_collision_prin = (dist_prin_axis <= (sphere.radius + cylinder.radius)) * check_prin_axis
        contact_pts_cylinder_prin = end_pt + sphere_pos_proj + cylinder.radius * normal_prin_axis
        contact_pts_sphere_prin = sphere_pos - sphere.radius * normal_prin_axis

        # normal_prin_axis *= check_prin_axis
        # contact_pts_sphere_prin *= check_prin_axis
        # contact_pts_cylinder_prin *= check_prin_axis

        return has_collision_prin.flatten(), contact_pts_cylinder_prin, contact_pts_sphere_prin, normal_prin_axis

    @staticmethod
    def check_endface(sphere_pos, sphere_pos_proj_mag, sphere_pos_rel, sphere, cylinder, prin_axis, end_pt, check_face):
        dist_endpt_face = torch.abs(sphere_pos_proj_mag)
        has_collision = (dist_endpt_face <= sphere.radius) * check_face
        tangent = sphere_pos_rel - torch.linalg.vecdot(sphere_pos_rel, prin_axis).unsqueeze(2) * prin_axis
        tangent_mag = torch.linalg.norm(tangent, dim=1).unsqueeze(2)
        tangent_mag = torch.clamp_max(tangent_mag, cylinder.radius) / tangent_mag
        contact_pts_cylinder = end_pt + tangent * tangent_mag
        normal = (sphere_pos - contact_pts_cylinder) / torch.linalg.norm(contact_pts_cylinder - sphere_pos,
                                                                         dim=1).unsqueeze(2)
        contact_pts_sphere = sphere_pos - sphere.radius * normal

        normal *= check_face
        contact_pts_sphere *= check_face
        contact_pts_cylinder *= check_face

        return has_collision, contact_pts_cylinder, contact_pts_sphere, normal


class CylinderGroundDetector(CollisionDetector):

    @staticmethod
    def detect(state, ground_state, rigid_body, ground):
        ground_z = ground_state[:, 2, ...]

        rot_mat1 = torch_quaternion.quat_as_rot_mat(state[:, 3:7, ...])
        principal_axis = rot_mat1[:, :, 2:]
        end_pts = rigid_body.compute_end_pts_from_state(state, principal_axis, rigid_body.length)
        end_pts = torch.concat(end_pts, dim=2)

        min_indices = torch.argmin(end_pts[:, 2:, :], dim=2).flatten()
        min_endpt = end_pts[torch.arange(0, end_pts.shape[0]), :, min_indices].unsqueeze(2)

        normal = torch.tensor(
            [0, 0, 1],
            dtype=state.dtype,
            device=state.device
        ).reshape(1, 3, 1)
        normal = normal.repeat(end_pts.shape[0], 1, 1).detach()

        out_vec = torch.linalg.cross(normal, principal_axis, dim=1)
        r = torch.linalg.cross(principal_axis, out_vec, dim=1)
        r /= torch.linalg.norm(r + 1e-6, dim=1).unsqueeze(2) / rigid_body.radius
        min_pts = min_endpt - r

        ground_contacts = min_pts.clone()
        ground_contacts[:, 2, :] = ground_z

        has_collision = min_pts[:, 2, ...] <= ground_z

        return [(has_collision.flatten(), ground_contacts, min_pts, normal)]


class CylinderCylinderDetector(CollisionDetector):

    @staticmethod
    def detect(state1, state2, rigid_body1, rigid_body2):
        rot_mat1 = torch_quaternion.quat_as_rot_mat(state1[:, 3:7, ...])
        rot_mat2 = torch_quaternion.quat_as_rot_mat(state2[:, 3:7, ...])
        end_pts1 = rigid_body1.compute_end_pts_from_state(state1, rot_mat1[:, :, 2:], rigid_body1.length)
        end_pts2 = rigid_body2.compute_end_pts_from_state(state2, rot_mat2[:, :, 2:], rigid_body2.length)

        p0, p1 = end_pts1[0], end_pts1[1]
        q0, q1 = end_pts2[0], end_pts2[1]

        u, v = p1 - p0, q1 - q0
        u_length, v_length = u.norm(dim=1).unsqueeze(2), v.norm(dim=1).unsqueeze(2)
        u /= u_length
        v /= v_length
        w = torch.linalg.cross(u, v, dim=1)
        w_length2 = torch.linalg.norm(w, dim=1).unsqueeze(2) ** 2

        # t = q0 - p0
        # det_u = torch.linalg.det(torch.concat([t, v, w], dim=2)).reshape(-1, 1, 1)
        # det_v = torch.linalg.det(torch.concat([t, u, w], dim=2)).reshape(-1, 1, 1)
        # t0, t1 = det_u / w_length2, det_v / w_length2
        lhs, rhs = torch.concat([u, -v, w], dim=2), q0 - p0
        sol = torch.linalg.solve(lhs, rhs)
        t0, t1, t2 = sol[:, 0:1, :], sol[:, 1:2, :], sol[:, 2:3, :]
        p, q = p0 + t0 * u, q0 + t1 * v
        # zero = torch.zeros(t0.shape, dtype=state1.dtype, device=state1.device)
        # p, q = p0 + torch.clamp(t0, zero.clone(), u_length) * u, q0 + torch.clamp(t1, zero.clone(), v_length) * v
        # dot_q, dot_p = torch.linalg.vecdot(v, p - q0, dim=1), torch.linalg.vecdot(u, q - p0, dim=1)
        # p, q = p0 + u * torch.clamp(dot_p, zero.clone(), u_length), q0 + v * torch.clamp(dot_q, zero.clone(), v_length)

        dists = torch.linalg.norm(p - q, dim=1).unsqueeze(2)
        has_collision = ((dists <= (rigid_body1.radius + rigid_body2.radius))
                         * (0 <= t0)
                         * (t0 <= u_length)
                         * (0 <= t1)
                         * (t1 <= v_length))

        # dists = torch.linalg.norm(t2 * w, dim=1)
        # has_collision = dists <= rigid_body1.radius + rigid_body2.radius and 0 < t0 < 1 and 0 < t1 < 1
        #
        # p = p0 + u * t0
        # q = q0 + v * t1

        normal = q - p
        normal /= torch.linalg.norm(normal, dim=1).unsqueeze(2)
        normal_no_v = normal - torch.linalg.vecdot(v, normal, dim=1).unsqueeze(2) * v
        normal_no_u = normal - torch.linalg.vecdot(u, normal, dim=1).unsqueeze(2) * u

        contact_q = q - normal * rigid_body1.radius / torch.linalg.norm(normal_no_v, dim=1).unsqueeze(2)
        contact_p = p + normal * rigid_body2.radius / torch.linalg.norm(normal_no_u, dim=1).unsqueeze(2)

        return [(has_collision.flatten(), contact_q, contact_p, normal)]

    # @staticmethod
    # def check_parallel(w_length):

    @staticmethod
    def check_axes_intersect(u, v, q0, p0):
        lhs = torch.concat([u[:, :2, ...], -v[:, :2, ...]], dim=2)
        rhs = (q0 - p0)[:, :2]
        sol = torch.linalg.solve(lhs, rhs)

        p_z = p0[:, 2, ...] + sol[:, 0, ...] * u[:, 2, ...]
        q_z = p0[:, 2, ...] + sol[:, 1, ...] * v[:, 2, ...]

        return q_z == p_z

    @staticmethod
    def _get_planar_endpts(endpt1, endpt2, radius, plane_normal, prin_axis):
        r_dir = torch.linalg.cross(plane_normal, prin_axis, dim=1)
        r_dir /= torch.linalg.norm(r_dir, dim=1).unsqueeze(2)

        pt11, pt12 = endpt1 + radius * r_dir, endpt1 - radius * r_dir
        pt21, pt22 = endpt2 + radius * r_dir, endpt2 - radius * r_dir

        return pt11, pt12, pt21, pt22, r_dir

    @staticmethod
    def _check_endpts(p0, p1, q0, q1, r1, r2, u, v, w):
        p_length = torch.linalg.norm(p1 - p0, dim=1).unsqueeze(2)
        q_length = torch.linalg.norm(q1 - q0, dim=1).unsqueeze(2)

        p011, p012, p111, p112, r_p = CylinderCylinderDetector._get_planar_endpts(p0, p1, r1, w, u)
        q011, q012, q111, q112, r_q = CylinderCylinderDetector._get_planar_endpts(q0, q1, r2, w, v)

        pt_inside, pt_dist, other_contacts = [], [], []

        for p in [p011, p012, p111, p112]:
            inside, dist, other_contact_pt = CylinderCylinderDetector._check_pt_in_cylinder(p, q0, v, q_length, r2)
            pt_inside.append(inside)
            pt_dist.append(r2 - dist)
            other_contacts.append(other_contact_pt)

        for q in [q011, q012, q111, q112]:
            inside, dist, other_contact_pt = CylinderCylinderDetector._check_pt_in_cylinder(q, p0, u, p_length, r1)
            pt_inside.append(inside)
            pt_inside.append(r1 - dist)
            other_contacts.append(other_contact_pt)

        contacts = torch.stack([p011, p012, p111, p112, q011, q012, q111, q112], dim=-1)
        normals = torch.stack([r_q, r_q, r_q, r_q, r_p, r_p, r_p, r_p], dim=-1)
        other_contacts = torch.stack(other_contacts, dim=-1)
        pt_inside = torch.stack(pt_inside, dim=-1)
        pt_dist = torch.stack(pt_dist, dim=-1)

        indices = torch.argmax(pt_dist, dim=-1)

        has_collision = pt_inside[:, indices]
        normals = normals[:, :, indices]
        contacts = contacts[:, :, indices]
        other_contacts = other_contacts[::, indices]

        return has_collision, contacts, other_contacts, normals

    @staticmethod
    def _check_pt_in_cylinder(pt, endpt1, prin_axis, length, radius):
        x = pt - endpt1
        x_prin = torch.linalg.vecdot(x, prin_axis).unsqueeze(2)
        r = x - x_prin * prin_axis
        dist = torch.linalg.norm(r).unsqueeze(2)
        cylinder_surface_pt = x + r * (radius - dist) / dist

        return (0 <= x_prin <= length) * (dist <= radius), dist, cylinder_surface_pt


class CompositeBodyRigidBodyDetector(CollisionDetector):

    @staticmethod
    def detect(rigid_body_state, composite_state, rigid_body, composite_body):
        rot_mat = torch_quaternion.quat_as_rot_mat(composite_state[:, 3:7, ...])
        body_collisions = []

        for inner_rigid_body in composite_body.rigid_bodies.values():
            detector = get_detector(inner_rigid_body, rigid_body)

            body_offset = composite_body.rigid_bodies_body_vecs[inner_rigid_body.name]
            world_offset = torch.matmul(rot_mat, body_offset)
            body_state = composite_state.clone()
            body_state[:, :3, ...] += world_offset

            order_cond = inner_rigid_body.shape <= rigid_body.shape
            state1, state2 = (body_state, rigid_body_state) if order_cond else (rigid_body_state, body_state)
            body1, body2 = (inner_rigid_body, rigid_body) if order_cond else (rigid_body, inner_rigid_body)

            detect_params = detector.detect(state1, state2, body1, body2)

            body_collisions.extend(detect_params)

        return body_collisions


class CompositeBodyGroundDetector(CollisionDetector):

    @staticmethod
    def detect(ground_state, composite_state, ground, composite_body):
        return CompositeBodyRigidBodyDetector.detect(ground_state, composite_state, ground, composite_body)


class CompositeCompositeDetector(CollisionDetector):

    @staticmethod
    def detect(composite_state1, composite_state2, composite_body1, composite_body2):
        rot_mat1 = torch_quaternion.quat_as_rot_mat(composite_state1[:, 3:7, ...])
        rot_mat2 = torch_quaternion.quat_as_rot_mat(composite_state2[:, 3:7, ...])

        body_collisions = []
        for rigid_body1 in composite_body1.rigid_bodies.values():
            body_offset1 = composite_body1.rigid_bodies_body_vecs[rigid_body1.name]
            world_offset1 = torch.matmul(rot_mat1, body_offset1)
            body_state1 = composite_state1.clone()
            body_state1[:, :3, ...] += world_offset1

            for rigid_body2 in composite_body2.rigid_bodies.values():
                body_offset2 = composite_body2.rigid_bodies_body_vecs[rigid_body2.name]
                world_offset2 = torch.matmul(rot_mat2, body_offset2)
                body_state2 = composite_state2.clone()
                body_state2[:, :3, ...] += world_offset2

                detector = get_detector(rigid_body1, rigid_body2)

                order_cond = rigid_body1.shape <= rigid_body2.shape
                state1, state2 = (body_state1, body_state2) if order_cond else (body_state2, body_state1)
                body1, body2 = (rigid_body1, rigid_body2) if order_cond else (rigid_body2, rigid_body1)

                detect_params = detector.detect(state1, state2, body1, body2)
                body_collisions.extend(detect_params)

        return body_collisions


def get_detector(body1, body2):
    detector_dict = {
        "composite_composite": CompositeCompositeDetector,
        "composite_cylinder": CompositeBodyRigidBodyDetector,
        "composite_ground": CompositeBodyGroundDetector,
        "composite_sphere": CompositeBodyRigidBodyDetector,
        "cylinder_cylinder": CylinderCylinderDetector,
        "cylinder_ground": CylinderGroundDetector,
        "cylinder_sphere": CylinderSphereDetector,
        "ground_sphere": GroundSphereDetector,
        "sphere_sphere": SphereSphereDetector,
    }

    shape1 = "composite" if isinstance(body1, CompositeBody) else body1.shape
    shape2 = "composite" if isinstance(body2, CompositeBody) else body2.shape

    shapes = sorted([shape1.lower(), shape2.lower()])
    return detector_dict["_".join(shapes)]


if __name__ == '__main__':
    # pos = torch.tensor([[1, 3, 5.0], [5, 4, 5.4]], dtype=torch.float64)
    # pos = torch.tensor([[1, 3, 5.0], [5, 4, 5.4]], dtype=torch.float64)
    rod1_end_pt1 = torch.tensor([0.0000, 2.0000, 3.5000], dtype=torch.float64)
    rod1_end_pt2 = torch.tensor([11.0000, 11.0000, 3.600], dtype=torch.float64)
    rod2_end_pt1 = torch.tensor([2.0000, 0.0000, 1.4000], dtype=torch.float64)
    rod2_end_pt2 = torch.tensor([11.0000, 13.0000, 1.900], dtype=torch.float64)

    end_pts1 = torch.zeros((1, 3, 2), dtype=torch.float64)
    end_pts1[0, :, 0] = rod1_end_pt1
    end_pts1[0, :, 1] = rod1_end_pt2

    end_pts2 = torch.zeros((1, 3, 2), dtype=torch.float64)
    end_pts2[0, :, 0] = rod2_end_pt1
    end_pts2[0, :, 1] = rod2_end_pt2

    # princ_axes = torch.tensor([[3, 4, 3], [-3, -4, 5]], dtype=torch.float64)
    # print(pos + princ_axes)
    # print(pos - princ_axes)
    # print(cylinder_cylinder(end_pts1, 1, end_pts2, 1))
