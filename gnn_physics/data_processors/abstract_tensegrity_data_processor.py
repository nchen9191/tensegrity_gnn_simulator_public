from typing import Dict

import torch
from torch_geometric.data import Data as GraphData

from gnn_physics.normalizer import AccumulatedNormalizer
from state_objects.base_state_object import BaseStateObject
from utilities import torch_quaternion
from utilities.misc_utils import DEFAULT_DTYPE
from utilities.tensor_utils import zeros


class AbstractRobotGraphDataProcessor(BaseStateObject):

    def __init__(self,
                 robot,
                 hier_node_feat_dict: Dict[str, Dict[str, int]],
                 hier_edge_feat_dict: Dict[str, Dict[str, int]],
                 dt=0.01):
        super().__init__('data processor')

        self.dt = dt
        self.robot = robot

        self.hier_node_feat_dict = hier_node_feat_dict
        self.hier_edge_feat_dict = hier_edge_feat_dict
        self.node_feat_lens = {k: sum(v.values()) for k, v in hier_node_feat_dict.items()}
        self.edge_feat_lens = {k: sum(v.values()) for k, v in hier_edge_feat_dict.items()}

        flatten_node_feats = {k2: v
                              for k1, d in self.hier_node_feat_dict.items()
                              for k2, v in d.items()}
        flatten_edge_feats = {k2: v
                              for k1, d in self.hier_edge_feat_dict.items()
                              for k2, v in d.items()}

        self.normalizers = {
            k: AccumulatedNormalizer((1, v), name=k, dtype=self.dtype)
            for k, v in {**flatten_node_feats, **flatten_edge_feats}.items()
        }

    def to(self, device):
        super().to(device)
        self.robot.to(device)

        for normalizer in self.normalizers.values():
            normalizer.to(device)

        return self

    def start_normalizers(self):
        for normalizer in self.normalizers.values():
            normalizer.start_accum()

    def stop_normalizers(self):
        for normalizer in self.normalizers.values():
            normalizer.stop_accum()

    def batch_edge_index(self, senders, receivers, batch_size):
        # Assume graphs are the same size and have the same connections
        senders = senders.repeat(batch_size, 1)
        receivers = receivers.repeat(batch_size, 1)

        num_nodes = senders.max() + 1
        offsets = num_nodes * torch.arange(0, batch_size,
                                           dtype=torch.long,
                                           device=senders.device).reshape(-1, 1)
        senders = (senders + offsets).reshape(1, -1)
        receivers = (receivers + offsets).reshape(1, -1)

        edge_indices = torch.vstack([senders, receivers])
        return edge_indices

    def compute_feats(self, graph):
        for n, (node_type, feats_dict) in enumerate(self.hier_node_feat_dict.items()):
            norm_feats = torch.hstack([
                self.normalizers[k](graph[k])
                for k in feats_dict.keys()
            ])
            graph[node_type + '_x'] = norm_feats

        for edge_type, feats_dict in self.hier_edge_feat_dict.items():

            norm_feats = {
                k: self.normalizers[k](graph[k])
                for k in feats_dict.keys()
            }
            for k, nn in norm_feats.items():
                if len(nn.shape) > 2:
                    print(k, nn.shape)
            norm_feats = torch.hstack(list(norm_feats.values()))
            graph[edge_type + "_edge_attr"] = norm_feats

        return graph

    def batch_state_to_graph(self, states):
        if isinstance(states, torch.Tensor):
            states = [states]

        node_poses = []
        for state in states:
            state_ = state.reshape(-1, 13, 1)
            node_pose = self.pose2node(state_[:, :7])
            node_poses.append(node_pose)

        last_state_ = states[-1].reshape(-1, 13, 1)
        last_prev_pos = last_state_[:, :3] - self.dt * last_state_[:, 7:10]
        last_prev_quat = torch_quaternion.update_quat(
            last_state_[:, 3:7],
            -last_state_[:, 10:],
            self.dt
        )
        last_prev_pose = torch.hstack([last_prev_pos, last_prev_quat])
        last_prev_node_pose = self.pose2node(last_prev_pose)
        node_poses.append(last_prev_node_pose)

        node_pose = torch.hstack(node_poses[:-1])
        prev_node_pose = torch.hstack(node_poses[1:])

        graph = self.build_graph(
            node_pose,
            prev_node_pose,
            states[0].shape[0]
        )
        return graph

    def build_graph(self, node_pose, prev_node_pose, batch_size):
        node_feats = self._compute_node_feats(
            node_pose,
            prev_node_pose,
            batch_size
        )

        edge_indices, edge_type = self._get_edge_index(batch_size)
        edge_feats = self._compute_edge_feats(
            node_feats,
            edge_indices,
            edge_type,
            batch_size,

        )

        graph = GraphData(
            edge_index=edge_indices,
            num_nodes=node_feats['pos'].shape[0],
            **node_feats,
            **edge_feats
        )

        return graph

    def pose2node(self, pose) -> torch.Tensor:
        com_pos = pose[:, :3]
        quat = pose[:, 3:7]

        n = len(self.robot.rods)
        batch_size = com_pos.shape[0] // n
        body_verts = torch.concat([
            bv.to(com_pos.device)
            for bv in self.robot.rod_body_verts
        ], dim=2).transpose(0, 2).repeat(batch_size, 1, 1)

        rot_mat = torch_quaternion.quat_as_rot_mat(quat)
        node_pos = torch.matmul(rot_mat, body_verts)
        node_pos = node_pos + com_pos

        node_pos = node_pos.transpose(1, 2).reshape(-1, 3)

        return node_pos

    def compute_node_vels(self, prev_node_pos, curr_node_pos):
        vels = (curr_node_pos - prev_node_pos) / self.dt
        return vels

    def _get_edge_index(self, batch_size):
        template_graph = self.robot.template_idx
        senders = torch.tensor([e[0] for e in template_graph],
                               dtype=torch.long,
                               device=self.device)
        recvrs = torch.tensor([e[1] for e in template_graph],
                              dtype=torch.long,
                              device=self.device)
        edge_indices = self.batch_edge_index(
            senders,
            recvrs,
            batch_size
        )

        return edge_indices

    def node2pose(self, node_pos, prev_node_pos, num_nodes, **kwargs):
        raise NotImplementedError

    def _compute_node_feats(self, node_pos, prev_node_pos, batch_size, **kwargs):
        raise NotImplementedError

    def _compute_edge_feats(self,
                            node_feats,
                            edge_indices,
                            edge_type,
                            batch_size,
                            **kwargs):
        raise NotImplementedError
