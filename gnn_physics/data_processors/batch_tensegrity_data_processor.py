from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Data as GraphData

from gnn_physics.data_processors.abstract_tensegrity_data_processor import AbstractRobotGraphDataProcessor
from gnn_physics.normalizer import AccumulatedNormalizer
from robots.tensegrity import TensegrityRobot
from utilities import torch_quaternion
from utilities.tensor_utils import zeros, safe_norm


class BatchTensegrityDataProcessor(AbstractRobotGraphDataProcessor):

    def __init__(self,
                 tensegrity: TensegrityRobot,
                 con_edge_threshold: float = 1e9,
                 num_steps_ahead: int = 1,
                 num_hist: int = 1,
                 dt: float = 0.01,
                 max_dist_to_grnd: float = 0.5):
        """

        @param tensegrity: robot object
        @param con_edge_threshold: threshold to attach edge between ground and endcap node
        @param num_steps_ahead: how many steps training traj will be
        @param num_hist: how many steps behind to attach to features
        @param dt: timestep size
        @param max_dist_to_grnd: clip value for dist to ground feature
        """
        self.MAX_DIST_TO_GRND = max_dist_to_grnd
        self.CONTACT_EDGE_THRESHOLD = con_edge_threshold
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

        # Add normalizer specifically for model output: dv
        self.normalizers['dv'] = AccumulatedNormalizer(
            (1, 3),
            name='dv',
            dtype=self.dtype
        )

    def _get_body_verts(self, batch_size: int) -> torch.Tensor:
        """
        Method to get node positions in body frame from robot
        @param batch_size: size of batch
        @return: torch tensor of body verts repeated batch_size times
        """
        body_verts = torch.vstack([
            bv.to(self.device)
            for bv in self.robot.rod_body_verts
        ])
        body_verts = torch.vstack([
            body_verts,
            zeros(body_verts[0:1].shape, ref_tensor=body_verts)
        ]).repeat(batch_size, 1, 1).squeeze(-1)

        return body_verts

    def _body_edge_index(self) -> torch.Tensor:
        """
        Get
        @return:
        """
        senders = torch.tensor([s for s, r in self.robot.template_idx],
                               dtype=torch.long,
                               device=self.device)
        receivers = torch.tensor([r for s, r in self.robot.template_idx],
                                 dtype=torch.long,
                                 device=self.device)

        edge_index = torch.vstack([senders, receivers])

        return edge_index

    def node2pose(self,
                  node_pos: torch.Tensor,
                  prev_node_pos: torch.Tensor,
                  num_nodes: int,
                  **kwargs):
        """
        Method to map node poses to SE(3) poses

        @param node_pos: (batch_size * num nodes per graph, 3 * num_hist)
        @param prev_node_pos: (batch_size * num nodes per graph, 3 * num_hist)
        @param num_nodes: num nodes per rod
        @return: torch tensor of SE(3) poses
        """
        curr_com_pos = node_pos.reshape(-1, num_nodes, 3).mean(dim=1)
        prev_com_pos = prev_node_pos.reshape(-1, num_nodes, 3).mean(dim=1)

        lin_vel = (curr_com_pos - prev_com_pos).unsqueeze(-1) / self.dt

        idx_0 = list(self.robot.rods.values())[0].sphere_idx0
        idx_1 = list(self.robot.rods.values())[0].sphere_idx1

        curr_sphere0 = node_pos[idx_0::num_nodes]
        curr_sphere1 = node_pos[idx_1::num_nodes]
        prev_sphere0 = prev_node_pos[idx_0::num_nodes]
        prev_sphere1 = prev_node_pos[idx_1::num_nodes]

        curr_prin = safe_norm(curr_sphere1 - curr_sphere0).unsqueeze(-1)
        prev_prin = safe_norm(prev_sphere1 - prev_sphere0).unsqueeze(-1)

        ang_vel = torch_quaternion.compute_ang_vel_vecs(prev_prin, curr_prin, self.dt)
        quat = torch_quaternion.compute_quat_btwn_z_and_vec(curr_prin)

        n_rods = len(self.robot.rods)
        state = torch.hstack([curr_com_pos.unsqueeze(-1), quat, lin_vel, ang_vel])
        state = state.reshape(-1, state.shape[1] * n_rods, 1)

        return state

    def _compute_node_feats(self,
                            node_pos: torch.Tensor,
                            prev_node_pos: torch.Tensor,
                            batch_size: int,
                            **kwargs
                            ) -> Dict[str, torch.Tensor]:
        """
        Method to compute all node feats based on curr and prev node poses

        @param node_pos: (batch_size * num nodes per graph, 3 * num_hist)
        @param prev_node_pos: (batch_size * num nodes per graph, 3 * num_hist)
        @param batch_size: size of batch
        @return: Dictionary of feat tensors
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
            "node_body_verts": self._get_body_verts(batch_size)
        }

        return node_feats

    def _insert_grnd_node(self,
                          node_pos: torch.Tensor,
                          batch_size: int
                          ) -> torch.Tensor:
        """
        Method to insert grnd node pos into node pos tensor
        @param node_pos: node positions
        @param batch_size: size of batch
        @return: augmented node poses with ground nodes
        """
        aug_node_pos = node_pos.reshape(batch_size, -1)
        grnd_pos = zeros((batch_size, 3), ref_tensor=node_pos)
        aug_node_pos = torch.hstack([aug_node_pos, grnd_pos]).reshape(-1, 3)

        return aug_node_pos

    def _compute_body_mask(self, batch_size: int, num_robot_nodes: int) -> torch.Tensor:
        """
        Method to compute a mask indicating which nodes are body nodes (non-ground)

        @param batch_size: size of batch
        @param num_robot_nodes: number of nodes in the robot's body graph
        @return: mask tensor
        """
        body_mask = torch.full(
            (batch_size, num_robot_nodes + 1),
            True,
            device=self.device
        )
        body_mask[:, -1] = False
        body_mask = body_mask.reshape(-1, 1)
        return body_mask

    def _compute_node_type(self, num_nodes: int) -> torch.Tensor:
        return torch.full((num_nodes, 1), 0, device=self.device)

    def build_graph(self,
                    node_pose: torch.Tensor,
                    prev_node_pose: torch.Tensor,
                    batch_size: int
                    ) -> GraphData:
        """
        Method to build graph based on node poses

        @param node_pose: (batch_size * num nodes per graph, 3 * num_hist) at timestep t
        @param prev_node_pose: (batch_size * num nodes per graph, 3 * num_hist) at time t-1
        @param batch_size: size of current batch
        @return: constructed graph
        """

        # Compute node feats
        node_pos = self._insert_grnd_node(node_pose, batch_size)
        prev_node_pos = self._insert_grnd_node(prev_node_pose, batch_size)
        body_mask = self._compute_body_mask(batch_size, self.robot.num_nodes)
        node_type = self._compute_node_type(node_pos.shape[0])

        node_feats = self._compute_node_feats(node_pos,
                                              prev_node_pos,
                                              batch_size)

        # Compute edge indices and feats
        body_edge_idx, contact_edge_idx, cable_edge_idx, edge_type \
            = self._compute_edge_idxs()
        edge_indices = torch.hstack([body_edge_idx,
                                     contact_edge_idx,
                                     cable_edge_idx])
        edge_indices = self.batch_edge_index(edge_indices[0:1, :],
                                             edge_indices[1:2, :],
                                             batch_size)
        edge_type = edge_type.repeat(batch_size, 1)

        # Filter out edges that are beyond threshold
        edge_indices, edge_type, body_rcvrs = self._filter_contact_edge_idx(
            edge_indices,
            edge_type,
            node_pos,
            batch_size
        )

        # Compute edge feats for all history and combine
        edge_feats_lis = []
        node_vel = node_feats['node_vel']
        for i in range(self.NUM_HIST):
            node_feats['node_pos'] = node_pos[:, i * 3: (i + 1) * 3]
            node_feats['node_vel'] = node_vel[:, i * 3: (i + 1) * 3]

            edge_feats = self._compute_edge_feats(
                node_feats,
                edge_indices,
                edge_type,
                batch_size,
                body_rcvrs=body_rcvrs
            )
            edge_feats_lis.append(edge_feats)

        edge_feats = {
            k: torch.hstack([e[k] for e in edge_feats_lis]) \
                # if k in self.hist_edge_feats \
                # else edge_feats_lis[0][k]
            for v in self.hier_edge_feat_dict.values()
            for k in v.keys()
        }

        # Split node pos and vels to curr values and all prev values
        node_feats['node_prev_pos'] = node_pos[:, 3:]
        node_feats['node_pos'] = node_pos[:, :3]
        node_feats['node_prev_vels'] = node_feats['node_vel'][:, 3:]
        node_feats['node_vel'] = node_feats['node_vel'][:, :3]

        # Final graph
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

    def _contact_edge_index(self,
                            contact_node_idxs: List,
                            grnd_idx: int
                            ) -> torch.Tensor:
        """
        Method to get contact edge indices

        @param contact_node_idxs: indices of nodes that are involved in contact events
        @param grnd_idx: index of ground in non-batched graph
        @return:
        """
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

    def _compute_edge_idxs(self) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Method to compute different edge type indices
        """
        body_edge_idx = self._body_edge_index()
        cable_edge_idx = self.robot.get_cable_edge_idxs().to(self.device)
        contact_edge_idx = self._contact_edge_index(
            self.robot.get_contact_nodes(),
            body_edge_idx.max() + 1
        )
        edge_type = torch.tensor([
            [0] * body_edge_idx.shape[1]
            + [1] * cable_edge_idx.shape[1]
            + [2] * contact_edge_idx.shape[1]
        ], dtype=torch.long, device=self.device).reshape(-1, 1)

        return body_edge_idx, contact_edge_idx, cable_edge_idx, edge_type

    def _compute_edge_feats(self,
                            node_feats,
                            edge_indices,
                            edge_type,
                            batch_size,
                            **kwargs):
        """
        Method to compute all edge feats

        @param node_feats: dictionary of node feats
        @param edge_indices: (2, num edges)
        @param batch_size: size of batch
        @return: Dictionary of feat tensors
        """
        node_pos = node_feats['node_pos']
        node_vels = node_feats['node_vel']
        body_rcvrs = kwargs['body_rcvrs']

        # Split edge indices
        edge_type = edge_type.flatten()
        body_edge_idx = edge_indices[:, edge_type == 0]
        cable_edge_idx = edge_indices[:, edge_type == 1]
        contact_edge_idx = edge_indices[:, edge_type == 2]

        # body edges
        body_dists = node_pos[body_edge_idx[1]] - node_pos[body_edge_idx[0]]
        body_dists_norm = body_dists.norm(dim=1, keepdim=True)
        body_verts = self._get_body_verts(batch_size)
        body_rest_dists = body_verts[body_edge_idx[1]] - body_verts[body_edge_idx[0]]
        body_rest_dists_norm = body_rest_dists.norm(dim=1, keepdim=True)

        # contact edges
        sphere_radius = list(self.robot.rods.values())[0].sphere_radius

        contact_dists = (node_pos[contact_edge_idx[1]][:, 2:3]
                         - node_pos[contact_edge_idx[0]][:, 2:3])
        contact_dists = contact_dists - body_rcvrs * sphere_radius.squeeze(-1)

        z = torch.tensor([[0, 0, 1]], dtype=node_vels.dtype, device=node_vels.device)
        contact_normal = z * body_rcvrs

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

        edge_feats = {
            'body_dist': body_dists,
            'body_dist_norm': body_dists_norm,
            'body_rest_dist': body_rest_dists,
            'body_rest_dist_norm': body_rest_dists_norm,
            'contact_dist': contact_dists,
            'contact_normal': contact_normal,
            'contact_tangent': contact_tangent,
            'contact_rel_vel_normal': contact_rel_vel_normal,
            'contact_rel_vel_tangent': contact_rel_vel_tangent,
            'cable_dist': cable_dists,
            'cable_dist_norm': cable_dists_norm,
            'cable_dir': cable_dir,
            'cable_rel_vel_norm': cable_rel_vel_norm,
            'cable_rest_length': cable_rest_lengths,
            'cable_stiffness': cable_stiffness,
            'cable_damping': cable_damping,
            'cable_stiffness_force_mag': cable_stiffness_force_mag,
            'cable_damping_force_mag': cable_damping_force_mag,
        }

        return edge_feats

    def _filter_contact_edge_idx(self,
                                 edge_indices: torch.Tensor,
                                 edge_type: torch.Tensor,
                                 node_pos: torch.Tensor,
                                 batch_size: int
                                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Method to filter out edges that are outside of threshold
        @param edge_indices: tensor of edge indices
        @param edge_type: tensor of edge types
        @param node_pos: tensor of node positions
        @param batch_size: size of batch
        @return: Filter edges, edge types, and body receiving nodes
        """
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
