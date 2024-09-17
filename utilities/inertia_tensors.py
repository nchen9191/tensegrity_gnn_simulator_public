from typing import Union

import numpy as np
import torch

from utilities.misc_utils import DEFAULT_DTYPE


def cylinder_body(mass: torch.Tensor,
                  length: torch.Tensor,
                  radius: torch.Tensor,
                  dtype: torch.dtype = DEFAULT_DTYPE) -> torch.Tensor:
    """
    Get bodyframe inertia tensor for cylinder

    :param mass: mass of cylinder
    :param length: length of cylinder
    :param radius: radius of cylinder
    :param dtype: data type for tensor precision
    :param offset: distance to reference point, assumed to be along principal axis
    :return: Inertia tensor
    """
    mom_inertia_principal = (1 / 2) * mass * radius ** 2
    mom_inertia_other = (1 / 12) * mass * length ** 2 + (1 / 4) * mass * radius ** 2

    I_body = torch.diag(torch.tensor([mom_inertia_other,
                                      mom_inertia_other,
                                      mom_inertia_principal],
                                     dtype=dtype))

    return I_body


def hollow_cylinder_body(mass: torch.Tensor,
                         length: torch.Tensor,
                         radius_out: torch.Tensor,
                         radius_in: torch.Tensor,
                         dtype: torch.dtype = DEFAULT_DTYPE) -> torch.Tensor:
    """
    Get bodyframe inertia tensor for hollow cylinder

    :param mass: mass of cylinder
    :param length: length of cylinder
    :param radius_out: outer radius of hollow cylinder
    :param radius_in: inner radius of hollow cylinder
    :param dtype: data type for tensor precision
    :param offset: distance to reference point, assumed to be along principal axis
    :return: Inertia tensor
    """
    sum_sq_r = radius_out ** 2 + radius_in ** 2
    I_body = cylinder_body(mass, sum_sq_r, length, dtype)

    return I_body


def solid_sphere_body(mass: torch.Tensor,
                      radius: torch.Tensor,
                      dtype: torch.dtype = DEFAULT_DTYPE) -> torch.Tensor:
    """
    Get bodyframe inertia tensor for solid sphere

    :param mass: mass of sphere
    :param radius: radius of sphere
    :param dtype: data type for tensor precision
    :param offset: distance to reference point, assumed to be along z-principal axis
    :return: Inertia tensor
    """
    mom_inertia = (2 / 5.0) * mass * radius ** 2
    mom_inertia_offset = mom_inertia
    I_body = torch.diag(torch.tensor([mom_inertia_offset, mom_inertia_offset, mom_inertia], dtype=dtype))

    return I_body


def rect_prism_body(mass: Union[float, torch.Tensor],
                    x_length: torch.Tensor,
                    y_length: torch.Tensor,
                    z_length: torch.Tensor,
                    dtype: torch.dtype = DEFAULT_DTYPE) -> torch.Tensor:
    I_body = (mass / 12.0) * torch.diag(torch.tensor([
        (y_length ** 2 + z_length ** 2),
        (x_length ** 2 + z_length ** 2),
        (y_length ** 2 + x_length ** 2),
    ], dtype=dtype))

    return I_body


def parallel_axis_offset(I_body: torch.Tensor, mass: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    d_mat = torch.tensor([[0, -d[0, 2], d[0, 1]], [0, 0, d[0, 0]], [0, 0, 0]], dtype=I_body.dtype)
    d_mat = d_mat - d_mat.T

    dd_mat = torch.matmul(d_mat, d_mat)
    I_body_new = I_body - mass * dd_mat

    return I_body_new


def body_to_world(rot_mat: torch.Tensor, body_inertia_tensor: torch.Tensor) -> torch.Tensor:
    """
    convert body to world frame inertia tensor with equation R*I*R^-1 = R*I*R^T
    :param rot_mat: rotation matrix/tensor
    :param body_inertia_tensor: body frame inertia tensor
    :return: world frame inertia tensor
    """
    world_inertia = torch.linalg.matmul(
        rot_mat,
        torch.matmul(
            body_inertia_tensor,
            rot_mat.transpose(-1, -2)
        )
    )

    return world_inertia


def inertia_inv(rot_mat, body_inertia_tensor):
    """
    Method to invert world inertia tensor with numpy
    I_world^-1 = R * I_body^-1 * R^T
    :param rot_mat: rotation matrix
    :param body_inertia_tensor: body frame inertia np array
    :return: Inverse world frame inertia tensor as np array
    """
    body_inertia_tensor_inv = np.linalg.inv(body_inertia_tensor)
    world_inertia_inv = body_to_world(rot_mat, body_inertia_tensor_inv)

    return world_inertia_inv
