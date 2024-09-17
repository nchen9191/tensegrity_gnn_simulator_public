import copy

import torch

from utilities.tensor_utils import zeros


@DeprecationWarning
class TorchQuaternion:

    def __init__(self, w, x, y, z, dtype=torch.float):
        self.tensor = torch.tensor([w, x, y, z], dtype=dtype)

    @classmethod
    def init_from_vec(cls, w, vec, dtype=torch.float64):
        return cls(w, vec[0], vec[1], vec[2], dtype)

    @classmethod
    def as_quat(cls, v, dtype=torch.float64):
        if isinstance(v, TorchQuaternion):
            return v
        elif isinstance(v, float) or isinstance(v, int):
            return cls(v, 0, 0, 0, dtype)
        else:
            return cls.init_from_vec(0, v, dtype)

    def norm(self):
        return self.tensor.norm()

    def as_mat(self):
        w, x, y, z = self.tensor
        mat = torch.tensor([[w, -x, -y, -z],
                            [x, w, -z, y],
                            [y, z, w, -x],
                            [z, -y, x, w]], dtype=self.tensor.dtype)
        return mat

    def as_rotation_mat(self):
        w, x, y, z = self.tensor / self.tensor.norm()

        # First row of the rotation matrix
        r00 = 2 * (w * w + x * x) - 1
        r01 = 2 * (x * y - w * z)
        r02 = 2 * (x * z + w * y)

        # Second row of the rotation matrix
        r10 = 2 * (x * y + w * z)
        r11 = 2 * (w * w + y * y) - 1
        r12 = 2 * (y * z - w * x)

        # Third row of the rotation matrix
        r20 = 2 * (x * z - w * y)
        r21 = 2 * (y * z + w * x)
        r22 = 2 * (w * w + z * z) - 1

        # 3x3 rotation matrix
        rot_matrix = torch.tensor([[r00, r01, r02],
                                   [r10, r11, r12],
                                   [r20, r21, r22]],
                                  dtype=self.tensor.dtype)

        return rot_matrix

    def copy(self):
        return copy.deepcopy(self)

    def __add__(self, other):
        new_tensor = self.tensor + other.tensor
        return TorchQuaternion.init_from_vec(new_tensor[0], new_tensor[1:], self.tensor.dtype)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float) or \
                (isinstance(other, torch.Tensor) and other.squeeze().shape[0] == 1):
            new_tensor = self.tensor * other
        else:
            mat = self.as_mat()
            new_tensor = torch.matmul(mat, other.tensor.unsqueeze(1)).squeeze()

        return TorchQuaternion.init_from_vec(new_tensor[0], new_tensor[1:], self.tensor.dtype)

    def __truediv__(self, other):
        new_tensor = self.tensor / other
        return TorchQuaternion.init_from_vec(new_tensor[0], new_tensor[1:], self.tensor.dtype)


def torch_quat_exp(q: TorchQuaternion):
    if q.norm() == 0:
        return TorchQuaternion(1, 0, 0, 0, q.tensor.dtype)

    w = q.tensor[0]
    v = q.tensor[1:]

    v_norm = torch.linalg.norm(v)

    new_w = torch.exp(w) * torch.cos(v_norm)
    new_v = torch.exp(w) * torch.sin(v_norm) * v / v_norm

    exp_q = TorchQuaternion.init_from_vec(new_w, new_v, q.tensor.dtype)

    return exp_q


def quat_add(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return q1 + q2


def quat_prod(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q1_w, q1_v = q1[:, 0:1], q1[:, 1:]
    q2_w, q2_v = q2[:, 0:1], q2[:, 1:]

    q_w = q1_w * q2_w - torch.linalg.vecdot(q1_v, q2_v, dim=1).unsqueeze(1)
    q_v = q1_w * q2_v + q2_w * q1_v + torch.linalg.cross(q1_v, q2_v, dim=1)

    q = torch.concat([q_w, q_v], dim=1)

    return q


def quat_exp(q: torch.Tensor) -> torch.Tensor:
    w = q[..., 0:1, :]
    v = q[..., 1:, :]

    # v_norm = torch.clamp_min(v.norm(dim=1, keepdim=True), 1e-8)
    v_norm = v.norm(dim=1, keepdim=True)

    exp_w = torch.exp(w)
    new_w = exp_w * torch.cos(v_norm)

    new_v = torch.zeros(v.shape, dtype=v.dtype, device=v.device)
    non_zero_v = torch.where(v_norm != 0)[0]
    # v_norm[v_norm == 0] = v_norm[v_norm == 0] + 1e-8
    new_v[non_zero_v] = (v[non_zero_v]
                         * exp_w[non_zero_v]
                         * torch.sin(v_norm[non_zero_v])
                         / v_norm[non_zero_v])
    # new_v = v * exp_w * torch.sin(v_norm) / v_norm
    exp_q = torch.hstack([new_w, new_v])

    return exp_q


def inverse_unit_quat(q):
    inv_q = q.clone()
    inv_q[:, 1:, :] *= -1

    return inv_q


def compute_ang_vel_quat(q_prev, q_curr, dt):
    ang_vel = torch.zeros((q_prev.shape[0], 3, 1),
                          dtype=q_prev.dtype,
                          device=q_prev.device)

    q_diff = quat_prod(
        q_curr.reshape(-1, 4, 1),
        inverse_unit_quat(q_prev.reshape(-1, 4, 1))
    )
    angle = 2 * torch.atan2(
        q_diff[:, 1:, :].norm(dim=1),
        q_diff[:, 0, :]
    ).unsqueeze(-1)

    non_zero_angle = (angle != 0.0).flatten().detach()
    # axis = q_diff[non_zero_angle, 1:, :] / torch.sin(angle[non_zero_angle])
    axis = q_diff[non_zero_angle, 1:] / torch.sin(angle[non_zero_angle] / 2)
    ang_vel[non_zero_angle] = angle[non_zero_angle] * axis / dt

    return ang_vel.detach()


def compute_ang_vel_vecs(prev_vec, curr_vec, dt):
    axis = torch.cross(prev_vec, curr_vec, dim=1)
    axis = axis / torch.clamp_min(axis.norm(dim=1, keepdim=True), 1e-8)

    angle = torch.linalg.vecdot(prev_vec, curr_vec, dim=1).unsqueeze(1)
    angle = torch.clamp(angle, -0.9999999, 0.9999999)
    angle = torch.acos(angle)

    ang_vel = (angle / dt) * axis

    return ang_vel


def compute_ang_vel_rot_mats(prev_rot_mat, curr_rot_mat, dt):
    if prev_rot_mat.shape[1] == 6:
        prev_rot_mat = xy_to_rot_mat(prev_rot_mat[:, :3], prev_rot_mat[:, 3:6])

    if curr_rot_mat.shape[1] == 6:
        curr_rot_mat = xy_to_rot_mat(curr_rot_mat[:, :3], curr_rot_mat[:, 3:6])

    rot_diff = torch.matmul(curr_rot_mat, prev_rot_mat.transpose(1, 2))
    trace_rot_diff = rot_diff[:, 0:1, 0:1] + rot_diff[:, 1:2, 1:2] + rot_diff[:, 2:3, 2:3]
    cos = (trace_rot_diff - 1) / 2.0
    cos = torch.clamp(cos, -1, 1)
    angle = torch.acos(cos)

    eye = torch.eye(
        3,
        dtype=rot_diff.dtype,
        device=rot_diff.device
    ).repeat(rot_diff.shape[0], 1, 1)
    T = rot_diff + rot_diff.transpose(1, 2) - (trace_rot_diff - 1.0) * eye
    axis = T[:, :, 0:1] / T[:, :, 0:1].norm(dim=1, keepdim=True)

    ang_vel = (angle / dt) * axis

    return ang_vel


def compute_angle_btwn_quats(q1, q2):
    # q1 -> q2 (order matters)
    q_diff = quat_prod(inverse_unit_quat(q1.reshape(-1, 4, 1)), q2.reshape(-1, 4, 1))
    angle = 2 * torch.atan2(q_diff[:, 1:, :].norm(dim=1), q_diff[:, 0, :])

    return angle


def compute_angle_btwn_rots(rot1, rot2):
    # rot1 -> rot2 (order matters)
    # assume rot1 and rot2 are proper orthonormal matrices
    rot_diff = torch.matmul(rot2, rot1.transpose(1, 2))
    trace_rot_diff = rot_diff[:, 0, 0] + rot_diff[:, 1, 1] + rot_diff[:, 2, 2]
    cos = (trace_rot_diff - 1) / 2.0
    cos = torch.clamp(cos, -1, 1)
    angle = torch.acos(cos)

    return angle


def compute_rot_mat_axis(rot_mat):
    trace = rot_mat[:, 0, 0] + rot_mat[:, 1, 1] + rot_mat[:, 2, 2]
    eye = torch.eye(
        3,
        dtype=rot_mat.dtype,
        device=rot_mat.device
    ).repeat(rot_mat.shape[0], 1, 1)
    T = rot_mat + rot_mat.transpose(1, 2) - (trace - 1.0) * eye

    return T[:, :, 0:1]


def rotate_vec_quat(q, vec):
    if len(q.shape) == 2:
        q = q.unsqueeze(-1)

    if len(vec.shape) == 2:
        vec = vec.unsqueeze(-1)

    vec_q = torch.hstack([
        torch.zeros((vec.shape[0], 1, 1),
                    device=vec.device,
                    dtype=vec.dtype),
        vec
    ])

    if vec.shape[0] == 1 and q.shape[0] > 1:
        vec = vec.repeat(q.shape[0], 1, 1)

    if q.shape[0] == 1 and vec.shape[0] > 1:
        q = q.repeat(vec.shape[0], 1, 1)

    q_conj = inverse_unit_quat(q)

    rot_vec = quat_prod(quat_prod(q, vec_q), q_conj)
    return rot_vec[:, 1:, :]


def quat_as_rot_mat(quat):
    quat_norm = quat.norm(dim=1, keepdim=True)
    q_unit = quat / quat_norm

    w = q_unit[:, 0:1]
    x = q_unit[:, 1:2]
    y = q_unit[:, 2:3]
    z = q_unit[:, 3:4]

    # First row of the rotation matrix
    r00 = 2 * (w * w + x * x) - 1
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r0 = torch.concat([r00, r01, r02], dim=2)

    # Second row of the rotation matrix
    r10 = 2 * (x * y + w * z)
    r11 = 2 * (w * w + y * y) - 1
    r12 = 2 * (y * z - w * x)
    r1 = torch.concat([r10, r11, r12], dim=2)

    # Third row of the rotation matrix
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 2 * (w * w + z * z) - 1
    r2 = torch.concat([r20, r21, r22], dim=2)

    rot_mat_tensor = torch.concat([r0, r1, r2], dim=1)

    return rot_mat_tensor


def cross_prod_mat(vec):
    if len(vec.shape) == 2:
        vec = vec.unsqueeze(-1)

    v1, v2, v3 = vec[:, 0:1], vec[:, 1:2], vec[:, 2:3]
    z0 = zeros(v1.shape, ref_tensor=vec)

    c1 = torch.hstack([z0, v3, -v2])
    c2 = torch.hstack([-v3, z0, v1])
    c3 = torch.hstack([v2, -v1, z0])

    mat = torch.concat([c1, c2, c3], dim=-1)
    return mat


def axis_angle_to_rot_mat(axis, angle):
    axis /= axis.norm(dim=1, keepdim=True)
    identity = torch.eye(3).unsqueeze(0).repeat(axis.shape[0], 1, 1)

    axis_mat = cross_prod_mat(axis)
    axis_mat_sq = torch.matmul(axis_mat, axis_mat)

    rot_mat = (identity
               + torch.sin(angle) * axis_mat
               + (1 - torch.cos(angle)) * axis_mat_sq)
    return rot_mat


def axis_angle_to_quat(axis, angle):
    axis /= axis.norm(dim=1, keepdim=True)
    w = torch.cos(angle / 2.0)
    v = torch.sin(angle / 2.0) * axis

    quat = torch.hstack([w, v])
    return quat


def rot_mat_to_quat(rot_mat):
    trace = rot_mat[:, 0, 0] + rot_mat[:, 1, 1] + rot_mat[:, 2, 2]
    qw = 0.25 * (1.0 + trace)

    use_sym = torch.abs(qw) < 1e-7

    qx = 0.25 * (rot_mat[:, 2, 1] - rot_mat[:, 1, 2])
    qy = 0.25 * (rot_mat[:, 0, 2] - rot_mat[:, 2, 0])
    qz = 0.25 * (rot_mat[:, 1, 0] - rot_mat[:, 0, 1])

    quat = torch.stack([qw, qx, qy, qz], dim=-1).unsqueeze(-1)

    if use_sym.any():
        eye = torch.eye(3, dtype=qw.dtype, device=qw.device).repeat(rot_mat.shape[0], 1, 1)
        T = rot_mat + rot_mat.transpose(1, 2) - (trace.reshape(-1, 1, 1) - 1.0) * eye
        quat[use_sym, 1:, 0] = -T[use_sym, :, 0]

        for i in range(1, 4):
            degen_quat = (torch.abs(quat) < 1e-6).all(dim=1)
            if degen_quat.any():
                if i < 3:
                    quat[degen_quat.squeeze(-1), 1:, 0] = -T[degen_quat.squeeze(-1), :, i]
                else:
                    raise Exception("Rot Mat conversion to quat degeneracy found")
            else:
                break

    quat = quat / quat.norm(dim=1, keepdim=True)

    """
    if (m22 < 0){
         if (m00 > m11){
             t = 1 + m00 - m11 - m22;
             q = quat( t, m01+m10, m20+m02, m12-m21 );
         }
         else {
             t = 1 - m00 + m11 - m22;
             q = quat( m01+m10, t, m12+m21, m20-m02 );
         }
    } else {
         if (m00 < -m11) {
             t = 1 - m00 - m11 + m22;
             q = quat( m20+m02, m12+m21, t, m01-m10 );
         }
         else {
             t = 1 + m00 + m11 + m22;
             q = quat( m12-m21, m20-m02, m01-m10, t );
         }
     }
     q *= 0.5 / Sqrt(t);
    """
    #
    # r00, r01, r02 = rot_mat[:, 0:1, 0], rot_mat[:, 0:1, 1], rot_mat[:, 0:1, 2]
    # r10, r11, r12 = rot_mat[:, 1:2, 0], rot_mat[:, 1:2, 1], rot_mat[:, 1:2, 2]
    # r20, r21, r22 = rot_mat[:, 2:3, 0], rot_mat[:, 2:3, 1], rot_mat[:, 2:3, 2]

    # pre_cond0 = r22 < 0.0
    # pre_cond1 = r00 > r11
    # pre_cond2 = r00 < -r11
    #
    # cond1 = torch.logical_and(pre_cond0, pre_cond1)
    # cond2 = torch.logical_and(pre_cond0, ~pre_cond1)
    # cond3 = torch.logical_and(~pre_cond0, pre_cond2)
    # cond4 = torch.logical_and(~pre_cond0, ~pre_cond2)
    #
    # t1 = 1.0 + r00 - r11 - r22
    # t2 = 1.0 - r00 + r11 - r22
    # t3 = 1.0 - r00 - r11 + r22
    # t4 = 1.0 + r00 + r11 + r22
    #
    # t = cond1 * t1 + cond2 * t2 + cond3 * t3 + cond4 * t4
    #
    # q1 = torch.hstack([t, r01 + r10, r20 + r02, r12 - r21])
    # q2 = torch.hstack([r01 + r10, t, r12 + r21, r20 - r02])
    # q3 = torch.hstack([r20 + r02, r12 + r21, t, r01 - r10])
    # q4 = torch.hstack([r12 - r21, r20 - r02, r01 - r10, t])
    #
    # q = cond1 * q1 + cond2 * q2 + cond3 * q3 + cond4 * q4
    # q *= 0.5 / (t ** 0.5)

    # qx = 0.25 * (rot_mat[:, 2, 1] - rot_mat[:, 1, 2]) / (qw + 1e-8)
    # qy = 0.25 * (rot_mat[:, 0, 2] - rot_mat[:, 2, 0]) / (qw + 1e-8)
    # qz = 0.25 * (rot_mat[:, 1, 0] - rot_mat[:, 0, 1]) / (qw + 1e-8)

    # q = torch.hstack([
    #     qw,
    #     torch.sign(r21 - r12) * 0.5 * torch.abs(())
    # ])

    return quat


def update_quat(quat, ang_vel, dt):
    """
    Linear exponentiation update of quat given constant ang_vel

    :param quat:
    :param ang_vel:
    :param dt:
    :return:
    """
    zero = zeros((ang_vel.shape[0], 1, quat.shape[-1]), ref_tensor=ang_vel)
    ang_vel_quat = torch.hstack([zero, ang_vel])
    ang_vel_quat = ang_vel_quat * 0.5 * dt

    new_quat = quat_prod(quat_exp(ang_vel_quat), quat)

    return new_quat


def update_quat2(quat, ang_vel, dt):
    """
    Linear exponentiation update of quat given constant ang_vel

    :param quat:
    :param ang_vel:
    :param dt:
    :return:
    """
    zero = torch.zeros((ang_vel.shape[0], 1, 1), dtype=ang_vel.dtype, device=ang_vel.device)
    ang_vel_quat = torch.concat([zero, ang_vel], dim=1)
    quat = quat + 0.5 * dt * quat_prod(ang_vel_quat, quat)

    quat_mag = torch.linalg.norm(quat, dim=1).detach()
    quat_mag[quat_mag == 0.0] += 1e-8
    quat /= quat_mag.unsqueeze(-1)

    return quat


def update_rot_mat(rot_mat, ang_vel, dt):
    if rot_mat.shape[1] == 6 and rot_mat.shape[2] == 1:
        rot_mat = xy_to_rot_mat(rot_mat[:, :3], rot_mat[:, 3:6])

    # rodrigues formula = e^(omega * theta) = I + omega * sin(theta) + omega^2 * (1 - cos(theta))
    ang_vel_norm = ang_vel.norm(dim=1, keepdim=True)
    nonzero = (ang_vel_norm != 0.0).squeeze()
    angle = ang_vel_norm * dt
    ang_vel_hat = ang_vel[nonzero] / ang_vel_norm[nonzero]

    omega = cross_prod_mat(ang_vel_hat)
    omega_rot_mat = torch.matmul(omega, rot_mat[nonzero])
    omega2_rot_mat = torch.matmul(omega, omega_rot_mat)

    new_rot_mat = rot_mat.clone()
    new_rot_mat[nonzero] = ((1 - torch.cos(angle)) * omega2_rot_mat
                            + torch.sin(angle) * omega_rot_mat
                            + rot_mat[nonzero])

    return new_rot_mat


def xy_to_rot_mat(x, y):
    if len(x.shape) == 2:
        x = x.unsqueeze(-1)

    if len(y.shape) == 2:
        y = y.unsqueeze(-1)

    x_hat = x / x.norm(dim=1, keepdim=True)

    z = torch.cross(x_hat, y, dim=1)
    z_hat = z / z.norm(dim=1, keepdim=True)

    y = torch.cross(z_hat, x_hat, dim=1)
    y_hat = y / y.norm(dim=1, keepdim=True)

    rot_mat = torch.concat([x_hat, y_hat, z_hat], dim=2)

    return rot_mat


def compute_q_btwn_vecs(v1, v2):
    v1_mag = v1.norm(dim=1, keepdim=True)
    v2_mag = v2.norm(dim=1, keepdim=True)

    v_mag = torch.sqrt(v1_mag ** 2 * v2_mag ** 2)

    q_w_dot = torch.linalg.vecdot(v1, v2, dim=1).unsqueeze(-1)
    q_xyz_cross = torch.cross(v1, v2, dim=1)
    q = torch.hstack([v_mag + q_w_dot, q_xyz_cross])
    q = q / q.norm(dim=1, keepdim=True)

    return q


def compute_quat_btwn_z_and_vec(prin_axis):
    principal_axis = prin_axis / prin_axis.norm(dim=1, keepdim=True)
    zeros = torch.zeros(principal_axis[..., 0, :].shape,
                        dtype=principal_axis.dtype,
                        device=principal_axis.device)

    # quaternion formula for rotation between z-axis and principal axis
    q = torch.concat([1 + principal_axis[..., 2, :],
                      -principal_axis[..., 1, :],
                      principal_axis[..., 0, :],
                      zeros],
                     dim=-1).reshape(-1, 4, 1)
    q = q / q.norm(dim=1, keepdim=True)

    return q.reshape(-1, 4, 1)


if __name__ == '__main__':
    q1 = torch.tensor([1.0, -1.4, 0.156, 0.67]).reshape(1, 4, 1)
    q1 /= q1.norm(dim=1, keepdim=True)

    r1 = quat_as_rot_mat(q1)
    x = r1[:, :, 0:1]

    rot_q = torch.hstack([
        torch.zeros((1, 1, 1)),
        x
        ])
    q2 = quat_prod(rot_q, q1)
    r2 = quat_as_rot_mat(q2)
    mm=0

