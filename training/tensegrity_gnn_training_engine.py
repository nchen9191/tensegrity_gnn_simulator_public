import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Union, Tuple, List

import torch
import tqdm
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Data as Graph

from simulators.tensegrity_gnn_simulator import TensegrityHybridGNNSimulator
from simulators.tensegrity_simulator import Tensegrity5dRobotSimulator
from state_objects.base_state_object import BaseStateObject
from state_objects.primitive_shapes import Cylinder
from utilities import misc_utils, torch_quaternion
from utilities.misc_utils import save_curr_code
from utilities.tensor_utils import zeros


class TensegrityGNNTrainingEngine(BaseStateObject):

    def __init__(self,
                 training_config: Dict,
                 criterion: _Loss,
                 dt: Union[float, torch.Tensor]):
        super().__init__("trainer")
        self.sim_config = training_config['sim_config']
        if isinstance(training_config['sim_config'], str):
            with open(training_config['sim_config'], "r") as j:
                self.sim_config = json.load(j)

        self.config = training_config
        self.simulator = self.get_dummy_simulator()
        self.num_steps_fwd = training_config['num_steps_fwd']
        self.num_hist = training_config['num_hist']
        self.dt = dt
        self.max_batch_size = training_config['batch_size']
        self.load_sim = training_config['load_sim'] \
            if 'load_sim' in training_config else False
        self.load_sim_path = training_config['load_sim_path'] \
            if 'load_sim_path' in training_config else None

        self.output_dir = training_config['output_path']
        Path(self.output_dir).mkdir(exist_ok=True)

        self.best_val_loss = 1e20
        self.best_rollout_loss = 1e20
        self.best_train_loss = 1e20
        self.num_no_improve = 0
        self.EVAL_STEPSIZE = 20
        self.MAX_NO_IMPROVE = 10
        self.PRINT_STEP = 100

        save_code_flag = (self.config['num_steps_fwd'] == 1
                          and not self.config['load_sim'])
        self.save_code(save_code_flag)
        self.train_data_dict, self.train_batches = (
            self.init_data(training_config['train_data_paths']))
        self.val_data_dict, self.val_batches = (
            self.init_data(training_config['val_data_paths']))

        delattr(self, "simulator")
        self.simulator = self.get_simulator()

        self.trainable_params = torch.nn.ParameterList(self.simulator.parameters())
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            **training_config['optimizer_params']
        )
        self.loss_fn = criterion

    def to(self, device):
        super().to(device)
        self.simulator.to(device)

        return self

    def save_code(self, save_code_flag=True):
        if save_code_flag:
            code_dir_name = "tensegrity_physics_engine"
            curr_code_dir = os.getcwd()
            code_output = Path(self.output_dir, code_dir_name)
            save_curr_code(curr_code_dir, code_output)

    def get_dummy_simulator(self):
        sim = Tensegrity5dRobotSimulator(self.sim_config['tensegrity_cfg'],
                                         self.sim_config['gravity'],
                                         self.sim_config['contact_params'])
        return sim

    def get_simulator(self):
        if self.load_sim and self.load_sim_path:
            sim = torch.load(self.load_sim_path, map_location="cpu")
            sim.reset_actuation()
            sim.cpu()
            print("Loaded simulator")
        else:
            sim = TensegrityHybridGNNSimulator(**self.sim_config)

        sim.data_processor.training = True
        return sim

    def init_data(self, data_paths):
        data_dict = {}
        data_dict['names'] = [p.split("/")[-2] for p in data_paths]

        data_jsons, target_gaits, extra_state_infos = (
            self.load_json_files(data_paths))
        data_dict.update({'data_jsons': data_jsons,
                          'target_gaits': target_gaits,
                          'extra_state_infos': extra_state_infos})

        data_dict['gt_end_pts'] = self._get_endpts(data_jsons)

        data_dict['times'] = [[d['time'] - data_json[0]['time'] for d in data_json]
                              for data_json in data_jsons]

        data_dict['states'], data_dict['controls'] = (
            self.data_json_to_states(data_jsons,
                                     data_dict['gt_end_pts'],
                                     data_dict['times'],
                                     extra_state_infos)
        )

        data_dict['act_lengths'], data_dict['motor_omegas'] = (
            self.get_act_lens_motor_omegas(extra_state_infos)
        )

        batches = self.build_batches(data_dict['states'],
                                     data_dict['gt_end_pts'],
                                     data_dict['times'],
                                     data_dict['controls'],
                                     data_dict['act_lengths'],
                                     data_dict['motor_omegas'],
                                     data_dict['target_gaits'],
                                     data_dict['data_jsons'])

        return data_dict, batches

    def _get_endpts(self, data_jsons):
        with torch.no_grad():
            data_end_pts = []
            for data_json in data_jsons:
                end_pts = [
                    [torch.tensor(e, dtype=self.dtype).reshape(1, 3, 1)
                     for e in d['end_pts']]
                    for d in data_json
                ]
                data_end_pts.append(end_pts)
        return data_end_pts

    def load_json_files(self, paths):
        data_jsons, target_gait_jsons, extra_state_infos = [], [], []
        for path in paths:
            with Path(path, "processed_data.json").open('r') as fp:
                data_jsons.append(json.load(fp))

            with Path(path, "target_gaits.json").open('r') as fp:
                target_gait_jsons.append(json.load(fp))

            extra_info_path = Path(path, f"extra_state_data.json")
            if extra_info_path.exists():
                with extra_info_path.open('r') as fp:
                    extra_state_infos.append(json.load(fp))
            else:
                extra_state_infos.append(None)

        return data_jsons, target_gait_jsons, extra_state_infos

    def data_to_pos_quat_ctrls(self, data_jsons, gt_end_pts, extra_state_jsons):
        data_pos, data_quats, data_controls = [], [], []
        for i, data_json in enumerate(data_jsons):
            pos, quats, controls = [], [], []
            for j, d in enumerate(data_json):
                end_pts = gt_end_pts[i][j]
                end_pts = [(end_pts[2 * k], end_pts[2 * k + 1])
                           for k in range(len(end_pts) // 2)]

                pos.append([(e[1] + e[0]) / 2 for e in end_pts])
                quats.append([
                    torch_quaternion.compute_quat_btwn_z_and_vec(
                        e[1] - e[0]
                    ) for e in end_pts
                ])

            data_pos.append(pos)
            data_quats.append(quats)
            data_controls.append(controls[:-1])

        if extra_state_jsons is not None:
            times = [[d['time'] - data_json[0]['time']
                      for d in data_json]
                     for data_json in data_jsons]
            data_controls = self.load_controls(extra_state_jsons, times)

        return data_pos, data_quats, data_controls

    def pos_quat_to_states(self, data_pos, data_quats, times, data_vels):
        num_rods = len(self.simulator.robot.rods)

        data_states = []
        for k, (pos, quats, times) in enumerate(zip(data_pos, data_quats, times)):
            states = []
            for i in range(len(pos)):
                prev_i = max(i - 1, 0)
                pos_0, pos_1 = pos[prev_i], pos[i]
                quat_0, quat_1 = quats[prev_i], quats[i]
                dt = times[i] - times[prev_i]

                if i > 0:
                    lin_vels = [(pos_1[j] - pos_0[j]) / dt for j in range(num_rods)]
                    ang_vels = [
                        torch_quaternion.compute_ang_vel_quat(
                            quat_0[j],
                            quat_1[j],
                            dt
                        ) for j in range(num_rods)
                    ]
                else:
                    lin_vels = [data_vels[k][0][:, j * 6: j * 6 + 3] for j in range(num_rods)]
                    ang_vels = [data_vels[k][0][:, j * 6 + 3: j * 6 + 6] for j in range(num_rods)]

                state = torch.hstack([
                    torch.hstack([pos_1[j], quat_1[j], lin_vels[j], ang_vels[j]])
                    for j in range(num_rods)
                ])
                states.append(state)
            data_states.append(states)

        return data_states

    def data_json_to_states(self,
                            data_jsons,
                            gt_end_pts,
                            times,
                            extra_state_jsons=None):
        data_pos, data_quats, data_controls = self.data_to_pos_quat_ctrls(
            data_jsons,
            gt_end_pts,
            extra_state_jsons
        )
        data_vels = self.get_ins_vels(data_jsons)
        data_states = self.pos_quat_to_states(
            data_pos,
            data_quats,
            times,
            data_vels
        )

        return data_states, data_controls

    def get_act_lens_motor_omegas(self, extra_state_jsons):
        act_cables = self.simulator.robot.actuated_cables.values()
        _rest_lengths = [s._rest_length for s in act_cables]

        data_act_lens, data_motor_omegas = [], []

        for extra_state_json in extra_state_jsons:
            act_lens, motor_omegas = [], []

            for e in extra_state_json:
                act_lengths = torch.tensor(
                    [_rest_lengths[i] - e['rest_lengths'][i]
                     for i in range(len(e['rest_lengths']))]
                ).reshape(1, -1, 1)
                motor_speeds = torch.tensor(
                    e['motor_speeds'],
                    dtype=self.dtype
                ).reshape(1, -1, 1)
                act_lens.append(act_lengths)
                motor_omegas.append(motor_speeds)

            data_act_lens.append(act_lens)
            data_motor_omegas.append(motor_omegas)

        return data_act_lens, data_motor_omegas

    def get_ins_vels(self, data_jsons):
        data_vels = []

        for i in range(len(data_jsons)):
            vels = []
            for j in range(len(data_jsons[i])):
                linvel = data_jsons[i][j]['linvel']
                angvel = data_jsons[i][j]['angvel']

                lin_vel = torch.tensor(linvel, dtype=self.dtype).reshape(-1, 3, 1)
                ang_vel = torch.tensor(angvel, dtype=self.dtype).reshape(-1, 3, 1)
                vel = torch.hstack([lin_vel, ang_vel]).reshape(1, -1, 1)
                vels.append(vel)

            vels.append(zeros(vels[0].shape, ref_tensor=vels[0]))
            data_vels.append(vels)

        return data_vels

    def endpts2pos(self, endpt1, endpt2):
        pos = (endpt2 + endpt1) / 2.0
        quat = torch_quaternion.compute_quat_btwn_z_and_vec(endpt2 - endpt1)

        return pos, quat

    def compute_node_loss(self, graphs, gt_end_pts, dt):
        norm_pred_dv, norm_gt_dv = [], []
        body_mask = graphs[0].body_mask.flatten()

        for i in range(len(graphs)):
            graph = graphs[i]
            norm_pred_dv.append(graph.decode_output[body_mask])

            end_pts = gt_end_pts[:, :, i: i + 1].reshape(-1, 6, 1)
            gt_pos, gt_quat = self.endpts2pos(end_pts[:, :3], end_pts[:, 3:])
            gt_nodes_pos = self.simulator.data_processor.pose2node(
                torch.hstack([gt_pos, gt_quat])
            )

            gt_p_vel = (gt_nodes_pos - graph.node_pos[body_mask]) / dt
            gt_dv = gt_p_vel - graph.node_vel[body_mask] - graph.pf_dv[body_mask]
            norm_gt_dv.append(self.simulator.data_processor.normalizers['dv'](gt_dv))

        norm_pred_dv = torch.stack(norm_pred_dv, dim=2)
        norm_gt_dv = torch.stack(norm_gt_dv, dim=2)

        loss = self.loss_fn(norm_pred_dv, norm_gt_dv)
        pos_loss = self.loss_fn(
            graphs[-1].p_node_pos[body_mask],
            gt_nodes_pos
        ).detach().item()

        return loss, pos_loss

    def load_controls(self, extra_state_jsons, traj_times):
        data_controls = []
        for i, times in enumerate(traj_times):
            traj_start_time = times[0]
            extra_state_start_time = extra_state_jsons[i][0]['time']
            curr_idx = 0
            controls = []

            for j in range(len(times) - 1):
                next_time = times[j + 1] - traj_start_time
                ctrls = []

                for k in range(curr_idx, len(extra_state_jsons[i])):
                    data_time = extra_state_jsons[i][k]['time'] - extra_state_start_time
                    if data_time == next_time:
                        curr_idx = k
                        break
                    ctrl = torch.tensor(extra_state_jsons[i][k]['controls'],
                                        dtype=self.dtype
                                        ).reshape(1, -1, 1)
                    ctrls.append(ctrl)

                control = torch.concat(ctrls, dim=2)
                controls.append(control)

            data_controls.append(controls)

        return data_controls

    def build_batch_dict(self, **kwargs):
        batch_dict = {}
        states = kwargs['states']
        times = kwargs['times']
        data_gt_end_pts = kwargs["gt_end_pts"]

        for i in range(len(states)):
            gt_end_pts = torch.vstack([torch.hstack(d) for d in data_gt_end_pts[i]])
            controls = torch.vstack(kwargs['controls'][i])
            act_lengths = torch.vstack(kwargs['act_lengths'][i])
            motor_omegas = torch.vstack(kwargs['motor_omegas'][i])

            x = states[i][:-self.num_steps_fwd]
            x = [x[0].clone() for _ in range(self.num_hist - 1)] + x
            batch_x = [torch.concat(x[j - self.num_hist + 1: j + 1], 2)
                       for j in range(self.num_hist - 1, len(x))]

            batch_y, batch_ctrls = [], []
            for j in range(1, self.num_steps_fwd + 1):
                end = -(self.num_steps_fwd - j) \
                    if j < self.num_steps_fwd else gt_end_pts.shape[0]
                batch_y.append(gt_end_pts[j:end])
                batch_ctrls.append(controls[j - 1: -(self.num_steps_fwd - j + 1)])

            batch_y = torch.concat(batch_y, dim=2)
            batch_ctrls = torch.concat(batch_ctrls, dim=2)

            batch_y = [batch_y[j: j + 1] for j in range(batch_y.shape[0])]
            batch_ctrls = [batch_ctrls[j: j + 1] for j in range(batch_ctrls.shape[0])]
            batch_act_lens = [act_lengths[j: j + 1] for j in range(act_lengths.shape[0])]
            batch_motor_speeds = [motor_omegas[j: j + 1] for j in range(motor_omegas.shape[0])]

            batch_dt = [times[i][j + self.num_steps_fwd] - times[i][j]
                        for j in range(len(times[i]) - self.num_steps_fwd)]

            batch_elms = zip(batch_x, batch_y,
                             batch_dt, batch_ctrls,
                             batch_act_lens, batch_motor_speeds)
            for x, y, dt, ctrl, act_len, motor_omega in batch_elms:
                delta_t = misc_utils.compute_num_steps(dt, self.dt) * self.dt

                batch = batch_dict.get(delta_t, [[] for _ in range(7)])
                batch[0].append(x)
                batch[1].append(y)
                batch[2].append(ctrl)
                batch[3].append(torch.tensor([[[dt]]], dtype=self.dtype))
                batch[4].append(act_len)
                batch[5].append(motor_omega)
                batch[6].append(torch.tensor([[[delta_t]]], dtype=self.dtype))
                batch_dict[delta_t] = batch

        return batch_dict

    def build_batches(self,
                      states,
                      gt_end_pts,
                      times,
                      controls,
                      act_lengths,
                      motor_omegas,
                      target_gaits,
                      data_jsons):
        batch_dict = self.build_batch_dict(states=states,
                                           gt_end_pts=gt_end_pts,
                                           times=times,
                                           controls=controls,
                                           act_lengths=act_lengths,
                                           motor_omegas=motor_omegas,
                                           target_gaits=target_gaits,
                                           data_jsons=data_jsons)

        batch_dict = self.shuffle_batches(batch_dict)
        batches = self.combine_to_batches(batch_dict)

        return batches

    def shuffle_batches(self, batch_dict):
        for k, v in batch_dict.items():
            n = len(v[0])
            idxs = [i for i in range(n)]
            random.shuffle(idxs)

            for i in range(len(v)):
                v[i] = [v[i][idx] for idx in idxs]

            batch_dict[k] = v
        return batch_dict

    def combine_to_batches(self, batch_dict):
        batches = []
        for k, v in batch_dict.items():
            n = len(v[0])
            num_batches = math.ceil(n / self.max_batch_size)

            for i in range(num_batches):
                end = min((i + 1) * self.max_batch_size, n)
                batch = tuple(
                    torch.vstack(v[j][i * self.max_batch_size: end])
                    for j in range(len(v))
                )
                batches.append((k, batch))

        return batches

    def batch_sim_ctrls(self,
                        batch_state,
                        controls,
                        delta_t,
                        act_lens,
                        motor_omegas) -> List[Graph]:
        num_steps = int(delta_t / self.dt)

        cables = self.simulator.robot.actuated_cables.values()
        for i, cable in enumerate(cables):
            cable.actuation_length = act_lens[:, i: i + 1].clone()
            cable.motor.motor_state.omega_t = motor_omegas[:, i: i + 1].clone()

        curr_state = batch_state[..., -1:].clone()
        prev_states = None if batch_state.shape[2] == 1 else batch_state[..., :-1].clone()
        graphs = []
        for i in range(num_steps):
            self.simulator.prev_states = None if prev_states is None \
                else prev_states.transpose(1, 2).reshape(curr_state.shape[0], -1, 1).clone()
            prev_states = None if prev_states is None \
                else torch.concat([prev_states, curr_state], dim=2)[..., 1:]

            curr_state, graph = self.simulator.step(
                curr_state,
                self.dt,
                control_signals=controls[:, :, i].clone()
            )
            graphs.append(graph)

        return graphs

    def _batch_compute_end_pts(self, batch_state: torch.Tensor, sim=None) -> torch.Tensor:
        """
        Compute end pts for entire batch

        :param batch_state: batch of states
        :return: batch of endpts
        """
        if sim is None:
            sim = self.simulator

        end_pts = []
        for i, rod in enumerate(sim.rigid_bodies.values()):
            state = batch_state[:, i * 13: i * 13 + 7]
            prin_axis = torch_quaternion.quat_as_rot_mat(state[:, 3:7])[..., 2:]
            end_pts.extend(Cylinder.compute_end_pts_from_state(state, prin_axis, rod.length))

        return torch.concat(end_pts, dim=1)

    def evaluate_rollouts(self, data_dict):
        loss = self.eval_rollout_fixed_ctrls(
            data_dict['states'],
            data_dict['data_jsons'],
            data_dict['act_lengths'],
            data_dict['motor_omegas'],
            data_dict['target_gaits'],
            data_dict['controls'],
            data_dict['gt_end_pts']
        )
        return loss

    def eval_rollout_fixed_ctrls(self,
                                 states,
                                 data_jsons,
                                 act_lengths,
                                 motor_omegas,
                                 target_gaits,
                                 all_controls,
                                 all_gt_endpts):
        training_device = self.device
        self.device = 'cpu'
        self.to(self.device)

        total_loss = 0.0
        for i in range(len(states)):
            self.simulator.curr_graph = None
            self.simulator.reset_actuation()

            cables = self.simulator.robot.actuated_cables.values()
            for k, cable in enumerate(cables):
                cable.actuation_length = act_lengths[i][0][:, k: k + 1].clone()
                cable.motor.motor_state.omega_t = motor_omegas[i][0][:, k: k + 1].clone()

            gt_data = data_jsons[i]
            curr_state = states[i][0].clone()
            gaits = target_gaits[i]
            pred_endpts = []
            # TODO: remove gaits here
            for j, tg in enumerate(tqdm.tqdm(gaits)):
                curr_gait_idx = tg['idx']
                gait_time_idx = (gaits[j + 1]['idx']
                                 if j < len(gaits) - 1
                                 else len(gt_data) - 1)

                for k in range(curr_gait_idx, gait_time_idx):
                    gait_dt = self.get_gait_time(gt_data, k, k + 1)
                    controls = all_controls[i][k].clone()
                    max_steps = misc_utils.compute_num_steps(gait_dt, self.dt)

                    curr_time = 0
                    for m in range(max_steps):
                        diff = gait_dt - curr_time
                        dt = diff if diff - 1e-6 < self.dt else self.dt
                        curr_state, _ = self.simulator.step(
                            curr_state,
                            dt,
                            control_signals=controls[:, :, m]
                        )
                        endpts = self._batch_compute_end_pts(curr_state)
                        pred_endpts.append(endpts)

            pred_endpts = torch.vstack(pred_endpts)
            gt_endpts = torch.vstack([
                torch.hstack(e)
                for e in all_gt_endpts[i][1:]
            ])
            loss = self.loss_fn(pred_endpts, gt_endpts)
            print(loss.detach().item())
            total_loss += loss.detach().item()

        total_loss /= len(states) if len(states) > 0 else 1
        self.device = training_device
        self.to(self.device)

        return total_loss

    def rotate_data_aug(self, batch_x, gt_end_pts):
        n = len(self.simulator.robot.rods)

        angle = 2 * torch.pi * (torch.rand((batch_x.shape[0], 1, 1), device=batch_x.device) - 0.5)
        w = torch.cos(angle / 2)
        xyz = torch.tensor([0, 0, 1],
                           dtype=self.dtype,
                           device=batch_x.device
                           ).reshape(1, 3, 1)
        xyz = (xyz.repeat(batch_x.shape[0], 1, 1) * torch.sin(angle / 2))
        q = torch.hstack([w, xyz]).repeat(1, n, 1).reshape(-1, 4, 1)

        batch_x_rots = []
        for i in range(batch_x.shape[2]):
            batch_x_i = batch_x[..., i: i + 1].reshape(-1, 13, 1)
            pos = torch_quaternion.rotate_vec_quat(q, batch_x_i[:, :3])
            quat = torch_quaternion.quat_prod(q, batch_x_i[:, 3:7])
            linvel = torch_quaternion.rotate_vec_quat(q, batch_x_i[:, 7:10])
            angvel = torch_quaternion.rotate_vec_quat(q, batch_x_i[:, 10:])

            batch_x_rots.append(torch.hstack([
                pos, quat, linvel, angvel
            ]).reshape(-1, 13 * n, 1))

        batch_x_rots = torch.concat(batch_x_rots, dim=2)

        gt_end_pts_rots = []
        for i in range(gt_end_pts.shape[2]):
            gt_end_pts_ = gt_end_pts[:, :, i: i + 1].reshape(-1, 6, 1)
            endpt_0_rot = torch_quaternion.rotate_vec_quat(q, gt_end_pts_[:, :3])
            endpt_1_rot = torch_quaternion.rotate_vec_quat(q, gt_end_pts_[:, 3:])
            gt_end_pts_rot = torch.hstack([
                endpt_0_rot, endpt_1_rot
            ]).reshape(-1, 6 * n, 1)

            gt_end_pts_rots.append(gt_end_pts_rot)

        gt_end_pts_rots = torch.concat(gt_end_pts_rots, dim=2)

        return batch_x_rots, gt_end_pts_rots

    def train_epoch(self, epoch_num) -> Tuple[float, float, float]:
        train_loss = self.run_one_epoch(self.train_batches,
                                        shuffle_data=False,
                                        rot_aug=True)

        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss
            self.num_no_improve = 0
        else:
            self.num_no_improve += 1

        if self.num_no_improve > self.MAX_NO_IMPROVE:
            print("No improvement, lowering learning rate")
            self.best_train_loss = train_loss
            self.num_no_improve = 0
            for p in self.optimizer.param_groups:
                p['lr'] /= 2.0

        with torch.no_grad():
            if epoch_num % self.EVAL_STEPSIZE == 0:
                self.simulator.data_processor.training = False
                val_loss = self.run_one_epoch(self.val_batches, grad_required=False)

                device = self.device
                try:
                    val_rollout_kf_loss = self.evaluate_rollouts(
                        self.val_data_dict
                    )
                except Exception as e:
                    print(e)
                    val_rollout_kf_loss = -99.
                    self.to(device)
                    self.device = device
                self.simulator.data_processor.training = True
            else:
                val_loss = -1.
                val_rollout_kf_loss = -1.

        if 0 < val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(
                self.simulator,
                Path(self.output_dir, "best_loss_model.pt")
            )
        if 0 < val_rollout_kf_loss < self.best_rollout_loss:
            self.best_rollout_loss = val_rollout_kf_loss
            torch.save(
                self.simulator,
                Path(self.output_dir, "best_rollout_model.pt")
            )

        losses = (
            train_loss, val_loss,
            val_rollout_kf_loss,
        )

        return losses

    def run_one_epoch(self,
                      batches,
                      grad_required=True,
                      shuffle_data=False,
                      rot_aug=False) -> float:
        if shuffle_data:
            random.shuffle(batches)

        total_loss, total_other_losses = 0.0, []
        num_train, curr_batch = 0, 0
        for delta_t, batch in tqdm.tqdm(batches):
            curr_batch += 1

            batch = (b.to(self.device) for b in batch)
            (batch_x, batch_y, batch_ctrl,
             time_gaps, act_lens, motor_omegas, d_t) = batch
            num_train += batch_x.shape[0]

            if rot_aug:
                batch_x, batch_y = self.rotate_data_aug(batch_x, batch_y)

            graphs = self.batch_sim_ctrls(batch_x,
                                          batch_ctrl,
                                          delta_t,
                                          act_lens,
                                          motor_omegas)

            losses = self.compute_node_loss(graphs, batch_y, self.dt)

            # If gradient updates required, run backward pass
            if grad_required:
                self.backward(losses[0])

            total_loss += losses[0].detach().item() * batch_x.shape[0]
            total_other_losses.append([
                l * batch_x.shape[0] for l in losses[1:]
            ])

            if curr_batch % self.PRINT_STEP == 0:
                avg_other_losses = [
                    sum(l) / num_train
                    for l in zip(*total_other_losses)
                ]
                print(total_loss / num_train, avg_other_losses)
        total_loss /= num_train

        return total_loss

    def get_gait_time(self, gt_data, start_idx, end_idx):
        t0 = gt_data[start_idx]['time']
        t1 = gt_data[end_idx]['time']
        return t1 - t0

    def log_status(self, losses: Tuple, epoch_num: int) -> None:
        """
        Method to print training status to console

        :param losses: Train loss, Val loss, Val rollout KF loss
        :param epoch_num: Current epoch
        """
        losses = [f'{l:.4}' for l in losses]

        loss_file = Path(self.output_dir, "loss.txt")
        loss_msg = (f'Epoch {epoch_num}, '
                    f'"Train/Val/Val KF Losses": {losses}')

        try:
            with loss_file.open('a') as fp:
                fp.write(loss_msg)
        except:
            with loss_file.open('w') as fp:
                fp.write(loss_msg)

        print(loss_msg)

    def compute_init_losses(self):
        if not self.load_sim:
            self.simulator.data_processor.start_normalizers()

        train_loss = self.run_one_epoch(self.train_batches,
                                        grad_required=False,
                                        rot_aug=True)
        self.simulator.data_processor.stop_normalizers()

        print(self.simulator.data_processor.normalizers['dv'].mean,
              self.simulator.data_processor.normalizers['dv'].std)

        val_loss = self.run_one_epoch(self.val_batches,
                                      grad_required=False)

        # val_rollout_kf_loss = self.evaluate_rollouts(
        #     self.val_data_dict
        # )

        losses = (
            train_loss,
            val_loss,
        )

        return losses

    def run(self, num_epochs: int):
        """
        Method to run entire training

        :param num_epochs: Number of epochs to train
        :return: List of train losses, List of val losses, dictionary of trainable params and list of losses
        """
        with torch.no_grad():
            losses = self.compute_init_losses()
            self.log_status(losses, 0)

        # Run training over num_epochs
        for n in range(num_epochs):
            # Run single epoch training and evaluation
            losses = self.train_epoch(n + 1)
            self.log_status(losses, n + 1)

    def backward(self, loss: torch.Tensor) -> None:
        """
        Run back propagation with loss tensor

        :param loss: torch.Tensor
        """
        if loss.grad_fn is not None:
            loss.backward()
            self.optimizer.step()
            torch.nn.utils.clip_grad_norm_(self.parameters(),
                                           max_norm=10)

        self.optimizer.zero_grad()
