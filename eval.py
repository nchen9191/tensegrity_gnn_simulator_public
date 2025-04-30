import json
from pathlib import Path

import numpy as np
import torch
import tqdm

from mujoco_visualizer_utils.mujoco_visualizer import MuJoCoVisualizer
from utilities import torch_quaternion
from utilities.misc_utils import DEFAULT_DTYPE


def rollout_by_ctrls(simulator,
                     ctrls,
                     dt,
                     start_state):
    """
    @param simulator: Simulator object
    @param ctrls: List of controls
    @param dt: Time step size (need to match training dt)
    @param start_state: Initial state

    return list of dictionary of time and poses
    """
    time = 0.0
    frames = []

    with torch.no_grad():
        curr_state = start_state \
            if start_state is not None \
            else simulator.get_curr_state()
        pose = curr_state.reshape(-1, 13, 1)[:, :7].flatten()
        frames.append({"time": time, "pose": pose})

        for ctrl in tqdm.tqdm(ctrls):
            curr_state, _ = simulator.step(
                curr_state,
                dt,
                control_signals=ctrl
            )

            time += dt
            pose = curr_state.reshape(-1, 13, 1)[:, :7].flatten()
            frames.append({'time': time, 'pose': pose})

    return frames


def compute_mean_errors(gt_data, rollout_poses):
    com_errs, rot_errs, pen_errs = [], [], []
    for i in range(1, len(gt_data)):
        pred_pose = rollout_poses[i]['pose'].reshape(-1, 7, 1)
        pred_pos, pred_quat = pred_pose[:, :3], pred_pose[:, 3:7]

        gt_pos = torch.tensor(
            gt_data[i]['pos'],
            dtype=DEFAULT_DTYPE
        ).reshape(-1, 3, 1)

        gt_quat = torch.tensor(
            gt_data[i]['quat'],
            dtype=DEFAULT_DTYPE
        ).reshape(-1, 4, 1)

        com_mse = ((gt_pos - pred_pos) ** 2).mean()
        ang_err = torch_quaternion.compute_angle_btwn_quats(gt_quat, pred_quat)

        gt_pen = torch.clamp_max(gt_pos[:, 2], 0.0)
        pred_pen = torch.clamp_max(pred_pos[:, 2], 0.0)
        pen_err = torch.clamp_min(gt_pen - pred_pen, 0.0)

        com_errs.append(com_mse)
        rot_errs.append(ang_err)
        pen_errs.append(pen_err)

    avg_com_err = torch.vstack(com_errs).mean().item()
    avg_rot_err = torch.vstack(rot_errs).mean().item()
    avg_pen_err = torch.vstack(rot_errs).mean().item()

    return avg_com_err, avg_pen_err, avg_rot_err


def save_video(dt, gt_data, rollout_poses, vid_path):
    vis = MuJoCoVisualizer()
    vis.set_xml_path(Path('mujoco_visualizer_utils/xml/3prism_real_upscaled_vis_w_gt.xml'))
    vis.set_camera('camera')
    frames = []
    for pred, gt in zip(rollout_poses, gt_data):
        pred_pose = pred['pose'].cpu().numpy().flatten()

        gt_pos = np.array(gt['pos']).reshape(-1, 3)
        gt_quat = np.array(gt['quat']).reshape(-1, 4)
        gt_pose = np.hstack((gt_pos, gt_quat)).flatten()

        frames.append({'time': gt['time'], 'pos': np.concatenate((pred_pose, gt_pose))})
    vis.data = frames
    vis.visualize(vid_path, dt)


def evaluate(simulator,
             gt_data,
             extra_gt_data,
             dt,
             vid_path=None):
    ctrls = [e['controls'] for e in extra_gt_data]
    init_rest_lengths = extra_gt_data[0]['rest_lengths']
    init_motor_speeds = extra_gt_data[0]['motor_speeds']

    cables = simulator.robot.actuated_cables.values()
    for i, c in enumerate(cables):
        c.actuation_length = c._rest_length - init_rest_lengths[i]
        c.motor.motor_state.omega_t = torch.tensor(
            init_motor_speeds[i],
            dtype=simulator.dtype
        ).reshape(1, 1, 1)

    end_pts = torch.tensor(gt_data[0]['end_pts'], dtype=simulator.dtype, device=simulator.device)
    pos = (end_pts[1::2] + end_pts[::2]) / 2
    prin = end_pts[1::2] - end_pts[::2]
    prin = prin / prin.norm(dim=1, keepdim=True)
    quat = torch_quaternion.compute_quat_btwn_z_and_vec(prin.unsqueeze(-1))

    linvel = torch.tensor(gt_data[0]['linvel'], dtype=simulator.dtype, device=simulator.device)
    angvel = torch.tensor(gt_data[0]['angvel'], dtype=simulator.dtype, device=simulator.device)

    start_state = torch.hstack([
        pos.reshape(-1, 3, 1),
        quat.reshape(-1, 4, 1),
        linvel.reshape(-1, 3, 1),
        angvel.reshape(-1, 3, 1),
    ]).reshape(1, -1, 1)

    rollout_poses = rollout_by_ctrls(
        simulator,
        ctrls,
        dt,
        start_state
    )

    if vid_path:
        save_video(dt, gt_data, rollout_poses, vid_path)

    avg_com_err, avg_pen_err, avg_rot_err = (
        compute_mean_errors(gt_data, rollout_poses))

    return avg_com_err, avg_rot_err, avg_pen_err


if __name__ == '__main__':
    model_path = Path("sample_model.pt")
    data_dir_path = Path("../data_sets/mjc_synthetic_5d_0.01/val/R2S2Rrolling_7/")
    vid_path = Path("./vid.mp4")

    simulator = torch.load(model_path, map_location='cpu')
    simulator.eval()
    simulator.to('cpu')

    gt_data_json = json.load((data_dir_path / "processed_data.json").open('r'))
    extra_data_json = json.load((data_dir_path / "5d_extra_state_data.json").open('r'))

    avg_com_err, avg_pen_err, avg_rot_err = \
        evaluate(simulator,
                 gt_data_json,
                 extra_data_json,
                 dt=0.01,
                 vid_path=vid_path)
    print(avg_com_err, avg_pen_err, avg_rot_err)
