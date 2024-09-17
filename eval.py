import torch
import tqdm

from simulators.tensegrity_simulator import Tensegrity5dRobotSimulator
from simulators.tensegrity_gnn_simulator import TensegrityHybridGNNSimulator
from utilities import torch_quaternion
from utilities.misc_utils import DEFAULT_DTYPE


def rollout_by_ctrls(simulator,
                     ctrls,
                     dt,
                     start_state):
    time = 0.0
    frames = []

    curr_state = start_state \
        if start_state is not None \
        else simulator.get_curr_state()
    pose = curr_state.reshape(-1, 13, 1)[:, :7].flatten().numpy()
    frames.append({"time": time, "pose": pose.tolist()})

    for ctrl in tqdm.tqdm(ctrls):
        with torch.no_grad():
            curr_state = simulator.step(
                curr_state,
                dt,
                control_signals=ctrl
            )

        time += dt
        pose = curr_state.reshape(-1, 13, 1)[:, :7].flatten().numpy()
        frames.append({'time': time, 'pose': pose.tolist()})

    return frames


def evaluate(simulator,
             gt_data,
             ctrls,
             init_rest_lengths,
             init_motor_speeds,
             dt):
    cables = simulator.actuated_cables.values()
    for i, c in enumerate(cables):
        c.actuation_length = c._rest_length - init_rest_lengths[i]
        c.motor.motor_state.omega_t = torch.tensor(
            init_motor_speeds[i],
            dtype=DEFAULT_DTYPE
        ).reshape(1, 1, 1)

    pos, quat = gt_data[0]['pos'], gt_data[0]['quat']
    linvel, angvel = gt_data[0]['linvel'], gt_data[0]['angvel']

    start_state = torch.tensor(
        pos + quat + linvel + angvel,
        dtype=DEFAULT_DTYPE
    ).reshape(1, 13, 1)

    rollout_poses = rollout_by_ctrls(
        simulator,
        ctrls,
        dt,
        start_state
    )

    com_errs, rot_errs, pen_errs = [], [], []
    for i in range(1, len(gt_data)):
        gt_pos = torch.tensor(
            gt_data[i]['pos'],
            dtype=DEFAULT_DTYPE
        ).reshape(1, 3, 1)

        gt_quat = torch.tensor(
            gt_data[i]['quat'],
            dtype=DEFAULT_DTYPE
        ).reshape(1, 4, 1)

        pred_pos = rollout_poses[i]['pose'][:, :3]
        pred_quat = rollout_poses[i]['pose'][:, 3:7]

        com_mse = ((gt_pos - pred_pos) ** 2).mean()
        ang_err = torch_quaternion.compute_angle_btwn_quats(gt_quat, pred_quat)

        gt_pen = torch.clamp_max(gt_pos[:, 2], 0.0)
        pred_pen = torch.clamp_max(pred_pos[:, 2], 0.0)
        pen_err = torch.clamp_min(gt_pen - pred_pen, 0.0)

        com_errs.append(com_mse.item())
        rot_errs.append(ang_err.item())
        pen_errs.append(pen_err.item())

    avg_com_err = sum(com_errs) / len(com_errs)
    avg_rot_err = sum(rot_errs) / len(rot_errs)
    avg_pen_err = sum(pen_errs) / len(pen_errs)

    return avg_com_err, avg_rot_err, avg_pen_err
