from collections import OrderedDict

import torch
from torch_geometric.data import Data as GraphData

from gnn_physics.data_processors.abstract_tensegrity_data_processor import AbstractRobotGraphDataProcessor
from gnn_physics.normalizer import AccumulatedNormalizer
from utilities import torch_quaternion
from utilities.tensor_utils import zeros


class BatchTensegrityDataProcessor(AbstractRobotGraphDataProcessor):

    def __init__(self,
                 tensegrity,
                 edge_threshold=1e9,
                 num_steps_ahead=1,
                 num_hist=1,
                 dt=0.01,
                 max_dist=0.5):
        self.MAX_DIST_TO_GRND = max_dist
        self.CONTACT_EDGE_THRESHOLD = edge_threshold
        self.NUM_STEPS_AHEAD = num_steps_ahead
        self.NUM_HIST = num_hist

        self.training = True

        node_feat_dict = OrderedDict({
            'node_inv_mass': 1,
            'node_inv_inertia': 3,
            'node_vel': 3,
            'node_dist_to_ground': 1 * num_hist,
            'node_body_verts': 3 * num_hist
        })

        if num_hist > 1:
            node_feat_dict['node_prev_vels'] = 3 * (num_hist - 1)

        body_edge_feat_dict = OrderedDict({
            'body_dist': 3 * num_hist,
            'body_dist_norm': 1 * num_hist,
            'body_rest_dist': 3,
            'body_rest_dist_norm': 1
        })

        cable_edge_feat_dict = OrderedDict({
            'cable_dist': 3 * num_hist,
            'cable_dist_norm': 1 * num_hist,
            'cable_dir': 3 * num_hist,
            'cable_rel_vel_norm': 1 * num_hist,
            'cable_rest_length': 1,
            'cable_stiffness': 1,
            'cable_damping': 1,
            'cable_stiffness_force_mag': 1,
            'cable_damping_force_mag': 1
        })

        contact_edge_feat_dict = OrderedDict({
            'contact_dist': 1 * num_hist,
            'contact_normal': 3 * num_hist,
            'contact_tangent': 3 * num_hist,
            'contact_rel_vel_normal': 1 * num_hist,
            'contact_rel_vel_tangent': 1 * num_hist,
        })

        node_dict = {'node': node_feat_dict}
        edge_dict = {
            'body': body_edge_feat_dict,
            'cable': cable_edge_feat_dict,
            'contact': contact_edge_feat_dict
        }

        super().__init__(tensegrity,
                         node_dict,
                         edge_dict,
                         dt)

        self.normalizers['dv'] = AccumulatedNormalizer(
            (1, 3),
            name='dv',
            dtype=self.dtype
        )

    def get_body_verts(self, batch_size):
        body_verts = torch.vstack([
            bv.to(self.device)
            for bv in self.robot.rod_body_verts
        ])
        body_verts = torch.vstack([
            body_verts,
            zeros(body_verts[0:1].shape, ref_tensor=body_verts)
        ]).repeat(batch_size, 1, 1).squeeze(-1)

        return body_verts

    def body_edge_index(self):
        senders = torch.tensor([s for s, r in self.robot.template_idx],
                               dtype=torch.long,
                               device=self.device)
        receivers = torch.tensor([r for s, r in self.robot.template_idx],
                                 dtype=torch.long,
                                 device=self.device)

        edge_index = torch.vstack([senders, receivers])

        return edge_index

    def node2pose(self, curr_pos, prev_pos, num_nodes, **kwargs):
        """"""
        curr_com_pos = sum(curr_pos[i::num_nodes] for i in range(num_nodes)) / num_nodes
        prev_com_pos = sum(prev_pos[i::num_nodes] for i in range(num_nodes)) / num_nodes

        lin_vel = (curr_com_pos - prev_com_pos).unsqueeze(-1) / self.dt

        curr_sphere1, prev_sphere1 = curr_pos[3::num_nodes], prev_pos[3::num_nodes]
        curr_sphere2, prev_sphere2 = curr_pos[4::num_nodes], prev_pos[4::num_nodes]

        curr_prin = (curr_sphere2 - curr_sphere1).unsqueeze(-1)
        prev_prin = (prev_sphere2 - prev_sphere1).unsqueeze(-1)

        curr_prin = curr_prin / torch.clamp_min(curr_prin.norm(dim=1, keepdim=True), 1e-8)
        prev_prin = prev_prin / torch.clamp_min(prev_prin.norm(dim=1, keepdim=True), 1e-8)

        ang_vel = torch_quaternion.compute_ang_vel_vecs(prev_prin, curr_prin, self.dt)
        quat = torch_quaternion.compute_quat_btwn_z_and_vec(curr_prin)

        n_rods = len(self.robot.rods)
        state = torch.hstack([curr_com_pos.unsqueeze(-1), quat, lin_vel, ang_vel])
        state = state.reshape(-1, state.shape[1] * n_rods, 1)

        return state

    def _compute_node_feats(self, node_pos, prev_node_pos, batch_size, **kwargs):
        """
        node_pos: (batch_size, 3 * self.num_hist)
        """
        node_vels = (node_pos - prev_node_pos) / self.dt

        inv_mass = torch.vstack([self.robot.inv_mass.squeeze(-1),
                                 zeros((1, 1), ref_tensor=node_pos)])
        inv_inertia = torch.vstack([self.robot.inv_inertia,
                                    zeros((1, 3), ref_tensor=node_pos)])

        inv_mass = inv_mass.repeat(batch_size, 1)
        inv_inertia = inv_inertia.repeat(batch_size, 1)

        sphere_radius = list(self.robot.rods.values())[0].sphere_radius.squeeze(-1)
        dist_to_ground = node_pos[:, 2::3] - sphere_radius
        dist_to_ground = torch.clamp_max(dist_to_ground, self.MAX_DIST_TO_GRND)

        node_feats = {
            "node_pos": node_pos,
            "node_vel": node_vels,
            "node_inv_mass": inv_mass,
            "node_inv_inertia": inv_inertia,
            "node_dist_to_ground": dist_to_ground,
            "node_body_verts": self.get_body_verts(batch_size)
        }

        return node_feats

    def _insert_grnd_node(self, node_pos, batch_size):
        aug_node_pos = node_pos.reshape(batch_size, -1)
        grnd_pos = zeros((batch_size, 3), ref_tensor=node_pos)
        aug_node_pos = torch.hstack([aug_node_pos, grnd_pos]).reshape(-1, 3)

        return aug_node_pos

    def compute_body_mask(self, batch_size, num_robot_nodes):
        body_mask = torch.full(
            (batch_size, num_robot_nodes + 1),
            True,
            device=self.device
        )
        body_mask[:, -1] = False
        body_mask = body_mask.reshape(-1, 1)
        return body_mask

    def compute_node_type(self, num_nodes):
        return torch.full((num_nodes, 1), 0, device=self.device)

    def build_graph(self, node_pos, prev_node_pos, batch_size):
        node_pos = self._insert_grnd_node(node_pos, batch_size)
        prev_node_pos = self._insert_grnd_node(prev_node_pos, batch_size)
        body_mask = self.compute_body_mask(batch_size, self.robot.num_nodes)
        node_type = self.compute_node_type(node_pos.shape[0])

        node_feats = self._compute_node_feats(node_pos,
                                              prev_node_pos,
                                              batch_size)

        # edge
        body_edge_idx, contact_edge_idx, cable_edge_idx, edge_type \
            = self._compute_edge_idxs()
        edge_indices = torch.hstack([body_edge_idx,
                                     contact_edge_idx,
                                     cable_edge_idx])
        edge_indices = self.batch_edge_index(edge_indices[0:1, :],
                                             edge_indices[1:2, :],
                                             batch_size)
        edge_type = edge_type.repeat(batch_size, 1)

        edge_indices, edge_type, body_rcvrs = self.filter_contact_edge_idx(
            edge_indices,
            edge_type,
            node_pos,
            batch_size
        )

        edge_feats = self._batch_compute_edge_feats(node_pos,
                                                    node_feats['node_vel'],
                                                    edge_indices,
                                                    edge_type,
                                                    batch_size,
                                                    body_rcvrs)

        node_feats['node_prev_pos'] = node_pos[:, 3:]
        node_feats['node_pos'] = node_pos[:, :3]
        node_feats['node_prev_vels'] = node_feats['node_vel'][:, 3:]
        node_feats['node_vel'] = node_feats['node_vel'][:, :3]

        # final graph
        graph = GraphData(
            edge_index=edge_indices,
            num_nodes=node_pos.shape[0],
            edge_type=edge_type,
            node_type=node_type,
            body_mask=body_mask,
            **node_feats,
            **edge_feats
        )

        return graph

    def contact_edge_index(self,
                           contact_node_idxs,
                           grnd_idx):
        senders = torch.tensor([contact_node_idxs],
                               dtype=torch.long,
                               device=self.device,
                               requires_grad=False)
        receivers = torch.full((1, senders.shape[1]),
                               grnd_idx,
                               dtype=torch.long,
                               device=self.device,
                               requires_grad=False)
        edge_index = torch.vstack([
            torch.hstack([senders, receivers]),
            torch.hstack([receivers, senders])
        ]).detach()

        return edge_index

    def _compute_edge_idxs(self):
        body_edge_idx = self.body_edge_index()
        cable_edge_idx = self.robot.get_cable_edge_idxs().to(self.device)
        contact_edge_idx = self.contact_edge_index(
            self.robot.get_contact_nodes(),
            body_edge_idx.max() + 1
        )
        edge_type = torch.tensor([
            [0] * body_edge_idx.shape[1]
            + [1] * cable_edge_idx.shape[1]
            + [2] * contact_edge_idx.shape[1]
        ], dtype=torch.long, device=self.device).reshape(-1, 1)

        return body_edge_idx, contact_edge_idx, cable_edge_idx, edge_type

    def _batch_compute_edge_feats(self,
                                  node_pos,
                                  node_vels,
                                  edge_indices,
                                  edge_type,
                                  batch_size,
                                  body_rcvrs):
        edge_type = edge_type.flatten()
        body_edge_idx_raw = edge_indices[:, edge_type == 0]
        cable_edge_idx_raw = edge_indices[:, edge_type == 1]
        contact_edge_idx_raw = edge_indices[:, edge_type == 2]

        node_pos = node_pos.reshape(-1, 3)
        node_vels = node_vels.reshape(-1, 3)
        body_edge_idx = body_edge_idx_raw.T.repeat(1, self.NUM_HIST).T
        cable_edge_idx = cable_edge_idx_raw.T.repeat(1, self.NUM_HIST).T
        contact_edge_idx = contact_edge_idx_raw.T.repeat(1, self.NUM_HIST).T

        # body edges
        body_dists = node_pos[body_edge_idx[1]] - node_pos[body_edge_idx[0]]
        body_dists_norm = body_dists.norm(dim=1, keepdim=True)
        body_verts = self.get_body_verts(batch_size)
        body_rest_dists = body_verts[body_edge_idx_raw[1]] - body_verts[body_edge_idx_raw[0]]
        body_rest_dists_norm = body_rest_dists.norm(dim=1, keepdim=True)

        # contact edges
        sphere_radius = list(self.robot.rods.values())[0].sphere_radius

        contact_dists = (node_pos[contact_edge_idx[1]][:, 2:3]
                         - node_pos[contact_edge_idx[0]][:, 2:3])
        contact_dists = contact_dists - body_rcvrs * sphere_radius

        z = torch.tensor([[0, 0, 1]], dtype=node_vels.dtype, device=node_vels.device)
        contact_normal = (z * body_rcvrs).repeat(1, self.NUM_HIST).reshape(-1, 3)

        contact_rel_vel = node_vels[contact_edge_idx[1], :3] - node_vels[contact_edge_idx[0], :3]
        contact_rel_vel_normal = torch.linalg.vecdot(
            contact_rel_vel,
            contact_normal,
            dim=1
        ).unsqueeze(1)
        contact_tangent = contact_rel_vel - contact_rel_vel_normal * contact_normal
        contact_rel_vel_tangent = contact_tangent.norm(dim=1, keepdim=True)
        contact_rel_vel_tangent = torch.clamp_min(contact_rel_vel_tangent, 1e-8)
        contact_tangent = contact_tangent / contact_rel_vel_tangent

        # cable edges
        cable_dists = node_pos[cable_edge_idx[0]] - node_pos[cable_edge_idx[1]]
        cable_dists_norm = cable_dists.norm(dim=1, keepdim=True)
        cable_dir = cable_dists / cable_dists_norm
        cable_rel_vel = node_vels[cable_edge_idx[1], :3] - node_vels[cable_edge_idx[0], :3]
        cable_rel_vel_norm = torch.linalg.vecdot(
            cable_rel_vel,
            cable_dir,
            dim=1
        ).unsqueeze(1)
        cable_stiffness = self.robot.cable_stiffness.squeeze(-1).repeat(batch_size, 1)
        cable_damping = self.robot.cable_damping.squeeze(-1).repeat(batch_size, 1)

        act_lengths = torch.hstack([s.actuation_length
                                    for cable in self.robot.actuated_cables.values()
                                    for s in [cable, cable]])
        nonact_lengths = zeros(
            (act_lengths.shape[0], len(self.robot.non_actuated_cables) * 2, 1),
            ref_tensor=act_lengths
        )
        act_lengths = torch.hstack([act_lengths, nonact_lengths])
        act_lengths = act_lengths.reshape(-1, 1)
        cable_rest_lengths = (self.robot.cable_rest_length
                              .squeeze(-1)
                              .repeat(batch_size, 1) - act_lengths)

        cable_stiffness_force_mag = torch.clamp_min(
            cable_stiffness * (cable_dists_norm - cable_rest_lengths),
            0
        )
        cable_damping_force_mag = cable_damping * cable_rel_vel_norm

        num_body_edges = body_edge_idx_raw.shape[1]
        num_con_edges = contact_edge_idx_raw.shape[1]
        num_cable_edges = cable_edge_idx_raw.shape[1]

        if num_con_edges > 0:
            contact_dists = contact_dists.reshape(num_con_edges, -1)
            contact_normal = contact_normal.reshape(num_con_edges, -1)
            contact_tangent = contact_tangent.reshape(num_con_edges, -1)
            contact_rel_vel_normal = contact_rel_vel_normal.reshape(num_con_edges, -1)
            contact_rel_vel_tangent = contact_rel_vel_tangent.reshape(num_con_edges, -1)

        edge_feats = {
            'body_dist': body_dists.reshape(num_body_edges, -1),
            'body_dist_norm': body_dists_norm.reshape(num_body_edges, -1),
            'body_rest_dist': body_rest_dists.reshape(num_body_edges, -1),
            'body_rest_dist_norm': body_rest_dists_norm.reshape(num_body_edges, -1),
            'contact_dist': contact_dists,
            'contact_normal': contact_normal,
            'contact_tangent': contact_tangent,
            'contact_rel_vel_normal': contact_rel_vel_normal,
            'contact_rel_vel_tangent': contact_rel_vel_tangent,
            'cable_dist': cable_dists.reshape(num_cable_edges, -1),
            'cable_dist_norm': cable_dists_norm.reshape(num_cable_edges, -1),
            'cable_dir': cable_dir.reshape(num_cable_edges, -1),
            'cable_rel_vel_norm': cable_rel_vel_norm.reshape(num_cable_edges, -1),
            'cable_rest_length': cable_rest_lengths.reshape(num_cable_edges, -1),
            'cable_stiffness': cable_stiffness.reshape(num_cable_edges, -1),
            'cable_damping': cable_damping.reshape(num_cable_edges, -1),
            'cable_stiffness_force_mag': cable_stiffness_force_mag.reshape(num_cable_edges, -1),
            'cable_damping_force_mag': cable_damping_force_mag.reshape(num_cable_edges, -1),
        }

        return edge_feats

    def filter_contact_edge_idx(self, edge_indices, edge_type, node_pos, batch_size):
        n = len(self.robot.rods) * 2
        mask = torch.tensor([True] * edge_indices.shape[1], device=node_pos.device)
        contact_mask = edge_type.flatten() == 2
        contact_edges = edge_indices[:, contact_mask]

        body_rcvrs = torch.tensor(
            [-1] * n + [1] * n, device=node_pos.device
        ).repeat(batch_size).reshape(-1, 1)
        dists = node_pos[contact_edges[1], 2:] - node_pos[contact_edges[0], 2:]
        dists = dists * body_rcvrs

        sphere_radius = list(self.robot.rods.values())[0].sphere_radius.squeeze(-1)
        close_dist = dists - sphere_radius < self.CONTACT_EDGE_THRESHOLD
        mask[contact_mask] = close_dist.flatten()
        edge_indices = edge_indices[:, mask]
        edge_type = edge_type[mask]
        body_rcvrs = body_rcvrs[close_dist, None]

        return edge_indices, edge_type, body_rcvrs
