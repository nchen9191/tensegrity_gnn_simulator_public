from typing import Dict, Union, List

import torch
from torch_geometric.data import Data as GraphData

from gnn_physics.normalizer import AccumulatedNormalizer
from robots.tensegrity import TensegrityRobot
from state_objects.base_state_object import BaseStateObject
from utilities import torch_quaternion
from utilities.misc_utils import DEFAULT_DTYPE
from utilities.tensor_utils import zeros


class AbstractRobotGraphDataProcessor(BaseStateObject):
    """
    Base class for graph data processors
    """

    def __init__(self,
                 robot: TensegrityRobot,
                 hier_node_feat_dict: Dict[str, Dict[str, int]],
                 hier_edge_feat_dict: Dict[str, Dict[str, int]],
                 dt=0.01):
        """
        @param robot: Robot class
        @param hier_node_feat_dict: {node_type: {feat_name: feat_size}}
        @param hier_edge_feat_dict: {edge_type: {feat_name: feat_size}}
        @param dt: stepsize, default=0.01
        """
        super().__init__('data processor')

        self.dt = dt
        self.robot = robot

        self.hier_node_feat_dict = hier_node_feat_dict
        self.hier_edge_feat_dict = hier_edge_feat_dict

        # Compute node and edge feat sizes, used for initializing encoders' input size
        self.node_feat_lens = {k: sum(v.values()) for k, v in hier_node_feat_dict.items()}
        self.edge_feat_lens = {k: sum(v.values()) for k, v in hier_edge_feat_dict.items()}

        # flatten node and edge feats dicts to initialize feat normalizers
        flatten_node_feats = {k2: v
                              for k1, d in self.hier_node_feat_dict.items()
                              for k2, v in d.items()}
        flatten_edge_feats = {k2: v
                              for k1, d in self.hier_edge_feat_dict.items()
                              for k2, v in d.items()}

        # Initialize normalizer dict
        self.normalizers = {
            k: AccumulatedNormalizer((1, v), name=k, dtype=self.dtype)
            for k, v in {**flatten_node_feats, **flatten_edge_feats}.items()
        }

    def to(self, device: Union[str, torch.device]):
        super().to(device)
        self.robot.to(device)

        for normalizer in self.normalizers.values():
            normalizer.to(device)

        return self

    def start_normalizers(self):
        """
        Set accumulation flag of all normalizers to true
        """
        for normalizer in self.normalizers.values():
            normalizer.start_accum()

    def stop_normalizers(self):
        """
        Set accumulation flag of all normalizers to talse
        """
        for normalizer in self.normalizers.values():
            normalizer.stop_accum()

    def batch_edge_index(self,
                         senders: torch.Tensor,
                         receivers: torch.Tensor,
                         batch_size: int
                         ) -> torch.Tensor:
        """
        Expand edge indices from one graph to a batch of graphs. Method assumes
        same size and connections

        @param senders: indices of starting nodes
        @param receivers: indices of ending nodes
        @param batch_size: int
        @return:
        """
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

    def compute_feats(self, graph: GraphData) -> GraphData:
        """
        Normalize and concat all node and edge feats to form input feat vectors

        @param graph: graph data object with raw features
        @return: graph filled with node and edge feats
        """
        # Normalize and concat all node feats with suffix _x
        for n, (node_type, feats_dict) in enumerate(self.hier_node_feat_dict.items()):
            norm_feats = torch.hstack([
                self.normalizers[k](graph[k])
                for k in feats_dict.keys()
            ])
            graph[node_type + '_x'] = norm_feats

        # Normalize and concat all node feats with suffix _edge_attr
        for edge_type, feats_dict in self.hier_edge_feat_dict.items():
            norm_feats = torch.hstack([
                self.normalizers[k](graph[k])
                for k in feats_dict.keys()
            ])
            graph[edge_type + "_edge_attr"] = norm_feats

        return graph

    def batch_state_to_graph(self, states: List[torch.Tensor]) -> GraphData:
        """
        Method to convert list of states to a graph object

        @param states: List of torch tensors
        @return: a graph with all feats and structure
        """
        if isinstance(states, torch.Tensor):
            states = [states]

        # Compute all node poses based on state pos and quats
        node_poses = []
        for state in states:
            state_ = state.reshape(-1, 13, 1)
            node_pose = self.pose2node(state_[:, :7])
            node_poses.append(node_pose)

        # Compute pos and quat prior to last state by stepping 1 time step back using vels
        last_state_ = states[-1].reshape(-1, 13, 1)
        last_prev_pos = last_state_[:, :3] - self.dt * last_state_[:, 7:10]
        last_prev_quat = torch_quaternion.update_quat(
            last_state_[:, 3:7],
            -last_state_[:, 10:],
            self.dt
        )
        last_prev_pose = torch.hstack([last_prev_pos, last_prev_quat])

        # Compute last node poses
        last_prev_node_pose = self.pose2node(last_prev_pose)
        node_poses.append(last_prev_node_pose)

        # Split node pose to curr and prev node pose
        node_pose = torch.hstack(node_poses[:-1])
        prev_node_pose = torch.hstack(node_poses[1:])

        # Build final graph
        graph = self.build_graph(
            node_pose,
            prev_node_pose,
            states[0].shape[0]
        )
        return graph

    def build_graph(self,
                    node_pose: torch.Tensor,
                    prev_node_pose: torch.Tensor,
                    batch_size: int) -> GraphData:
        """
        Method to build graph based on node poses

        @param node_pose: (batch_size * num nodes per graph, 3 * num_hist) at timestep t
        @param prev_node_pose: (batch_size * num nodes per graph, 3 * num_hist) at time t-1
        @param batch_size: size of current batch
        @return: constructed graph
        """
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
            batch_size
        )

        graph = GraphData(
            edge_index=edge_indices,
            num_nodes=node_feats['pos'].shape[0],
            edge_type=edge_type,
            **node_feats,
            **edge_feats
        )

        return graph

    def pose2node(self, pose: torch.Tensor) -> torch.Tensor:
        """
        SE(3) pose to 3D node poses
        @param pose: (batch size * num rods, 7)
        @return: tensor (batch_size * num nodes per graph, 3)
        """
        # s\Split to pos and quat
        com_pos = pose[:, :3]
        quat = pose[:, 3:7]

        #  Get positions of nodes in body frame
        n = len(self.robot.rods)
        batch_size = com_pos.shape[0] // n
        body_verts = torch.concat([
            bv.to(com_pos.device)
            for bv in self.robot.rod_body_verts
        ], dim=2).transpose(0, 2).repeat(batch_size, 1, 1)

        # Rotate and translate body verts to world frame
        rot_mat = torch_quaternion.quat_as_rot_mat(quat)
        node_pos = torch.matmul(rot_mat, body_verts)
        node_pos = node_pos + com_pos
        node_pos = node_pos.transpose(1, 2).reshape(-1, 3)

        return node_pos

    def compute_node_vels(self,
                          prev_node_pos: torch.Tensor,
                          curr_node_pos: torch.Tensor
                          ) -> torch.Tensor:
        """
        Compute finite-diff/avg velocities

        @param prev_node_pos: (batch_size * num nodes per graph, 3 * num_hist)
        @param curr_node_pos: (batch_size * num nodes per graph, 3 * num_hist)
        @return: 1st order finite-diff vels (batch_size * num nodes per graph, 3 * num_hist)
        """
        vels = (curr_node_pos - prev_node_pos) / self.dt
        return vels

    def _get_edge_index(self, batch_size):
        """
        Convert template graph stored in robot to torch tensor of edges
        @param batch_size:
        @return:
        """
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

    def node2pose(self,
                  node_pos: torch.Tensor,
                  prev_node_pos: torch.Tensor,
                  num_nodes: int,
                  **kwargs
                  ) -> torch.Tensor:
        """
        Method to map node poses to SE(3) poses

        @param node_pos: (batch_size * num nodes per graph, 3 * num_hist)
        @param prev_node_pos: (batch_size * num nodes per graph, 3 * num_hist)
        @param num_nodes: num nodes per rod
        @return: torch tensor of SE(3) poses
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def _compute_edge_feats(self,
                            node_feats: Dict[str, torch.Tensor],
                            edge_indices: torch.Tensor,
                            edge_type: torch.Tensor,
                            batch_size: int,
                            **kwargs
                            ) -> Dict[str, torch.Tensor]:
        """
        Method to compute all edge feats

        @param node_feats: dictionary of node feats
        @param edge_indices: (2, num edges)
        @param batch_size: size of batch
        @return: Dictionary of feat tensors
        """
        raise NotImplementedError
