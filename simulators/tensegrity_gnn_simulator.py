from typing import Dict, Union, Tuple, Optional

import torch
from torch_geometric.data import Data as Graph

from gnn_physics.data_processors.abstract_tensegrity_data_processor import AbstractRobotGraphDataProcessor
from gnn_physics.data_processors.batch_tensegrity_data_processor import BatchTensegrityDataProcessor
from gnn_physics.gnn import EncodeProcessDecode
from robots.tensegrity import TensegrityRobotGNN
from simulators.tensegrity_simulator import Tensegrity5dRobotSimulator
from state_objects.base_state_object import BaseStateObject
from utilities import torch_quaternion
from utilities.misc_utils import DEFAULT_DTYPE
from utilities.tensor_utils import zeros


class LearnedSimulator(BaseStateObject):
    """
    Parent simulator that includes a GNN model
    """

    def __init__(
            self,
            node_types: Dict[str, int],
            edge_types: Dict[str, int],
            n_out: int,
            latent_dim: int,
            nmessage_passing_steps: int,
            nmlp_layers: int,
            mlp_hidden_dim: int,
            data_processor: AbstractRobotGraphDataProcessor,
            processor_shared_weights=False):
        super().__init__('learned simulator')

        self.data_processor = data_processor

        # Initialize the EncodeProcessDecode GNN
        self._encode_process_decode = EncodeProcessDecode(
            node_types=node_types,
            edge_types=edge_types,
            n_out=n_out,
            latent_dim=latent_dim,
            nmessage_passing_steps=nmessage_passing_steps,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            processor_shared_weights=processor_shared_weights,
        )

    @property
    def robot(self):
        return self.data_processor.robot

    def to(self, device):
        super().to(device)
        self._encode_process_decode = self._encode_process_decode.to(device)
        self.data_processor = self.data_processor.to(device)

        return self

    def forward(self, graph):
        graph = self.data_processor.compute_feats(graph)
        graph = self._encode_process_decode(graph)
        return graph

    def update_state(self, next_state: torch.Tensor) -> None:
        """
        Update state and other internal attribute from state
        """
        if len(next_state.shape) == 1:
            next_state = next_state.unsqueeze(0)

        next_state = next_state.reshape(-1, next_state.shape[1], 1)
        pos = next_state[:, :3]
        quat = next_state[:, 3:7]
        linear_vel = next_state[:, 7:10]
        ang_vel = next_state[:, 10:]

        self.rigid_body.update_state(pos, linear_vel, quat, ang_vel)

    def step(self, graph):
        return self(graph)

    def apply_controls(self, ctrls):
        pass


class TensegrityGNNSimulator(LearnedSimulator):
    """
    Tensegrity with only GNN
    """

    def __init__(self,
                 n_out: int,
                 latent_dim: int,
                 nmessage_passing_steps: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 processor_shared_weights=False,
                 dt=0.01,
                 tensegrity_cfg=None,
                 robot=None,
                 num_prev_states=0):
        assert robot is not None or tensegrity_cfg is not None

        self.prev_states = None if num_prev_states == 0 else [None] * num_prev_states

        if robot is None:
            robot = TensegrityRobotGNN(tensegrity_cfg)

        data_processor = BatchTensegrityDataProcessor(robot, dt=dt)

        node_types = {k: sum(v.values()) for k, v in data_processor.hier_node_feat_dict.items()}
        edge_types = {k: sum(v.values()) for k, v in data_processor.hier_edge_feat_dict.items()}

        super().__init__(node_types,
                         edge_types,
                         n_out,
                         latent_dim,
                         nmessage_passing_steps,
                         nmlp_layers,
                         mlp_hidden_dim,
                         data_processor,
                         processor_shared_weights)

        if self.dtype == torch.float64:
            self._encode_process_decode = self._encode_process_decode.double()

    def process_gnn(self, state):
        """
        Method to build and process graph from a state

        @param state: SE(3) state

        @return: processed graph
        """
        states = [state]
        if self.prev_states:
            states += self.prev_states

        graph = self.data_processor.batch_state_to_graph(states)
        graph = self.forward(graph)

        normalizer = self.data_processor.normalizers['dv']
        graph['p_node_dv'] = normalizer.inverse(graph['decode_output'])
        graph['p_node_vel'] = graph.node_vel + graph.p_node_dv
        graph['p_node_pos'] = graph.node_pos + self.data_processor.dt * graph.p_node_vel

        return graph

    def step(self, state, dt, ctrls=None):
        self.update_state(state)
        self.apply_controls(ctrls)

        graph = self.process_gnn(state)

        body_mask = graph.body_mask.flatten()
        next_state = self.data_processor.node2pose(
            graph.p_pos[body_mask],
            graph.pos[body_mask],
            self.robot.num_nodes_per_rod
        )

        return next_state


class TensegrityHybridGNNSimulator(Tensegrity5dRobotSimulator):
    """
    Hybrid simulator for first-principle models for passive forces + GNN
    """

    def __init__(self,
                 tensegrity_cfg,
                 gravity,
                 contact_params,
                 n_out: int,
                 latent_dim: int,
                 nmessage_passing_steps: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 processor_shared_weights=False,
                 dt=0.01):
        super().__init__(tensegrity_cfg,
                         gravity,
                         contact_params)

        # initialize GNN simulator within
        self.gnn_sim = TensegrityGNNSimulator(
            n_out,
            latent_dim,
            nmessage_passing_steps,
            nmlp_layers,
            mlp_hidden_dim,
            processor_shared_weights,
            dt,
            robot=self.robot
        )

    def to(self, device):
        super().to(device)
        self.gnn_sim = self.gnn_sim.to(device)

        return self

    def build_robot(self, cfg):
        return TensegrityRobotGNN(cfg)

    @property
    def data_processor(self):
        return self.gnn_sim.data_processor

    def compute_contact_deltas(self,
                               pre_contact_state: torch.Tensor,
                               dt: Union[torch.Tensor, float]
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return all zeros here since we are going to use GNN to resolve contact

        @param pre_contact_state: next state if contact not considered
        @param dt: timestep size
        @return: linear vel correction, ang vel correction, time of impact
        """
        pre_next_state_ = pre_contact_state.reshape(-1, 13, 1)

        delta_v = zeros((pre_next_state_.shape[0], 3, 1),
                        ref_tensor=pre_contact_state)
        delta_w = zeros((pre_next_state_.shape[0], 3, 1),
                        ref_tensor=pre_contact_state)
        toi = zeros((pre_next_state_.shape[0], 1, 1),
                    ref_tensor=pre_contact_state)

        return delta_v, delta_w, toi

    def resolve_contacts(self,
                         pre_contact_state: torch.Tensor,
                         dt: Union[torch.Tensor, float],
                         delta_v: torch.Tensor,
                         delta_w: torch.Tensor,
                         toi) -> Tuple[torch.Tensor, Graph]:
        """
        Method where GNN is used to predict velocity corrections using GNN

        @param pre_contact_state: next state if contact not considered
        @param dt: timestep size
        @param delta_v: does not use
        @param delta_w: does not use
        @param toi: does not use
        @return: next state
        """
        # Create and run forward pass with GNN
        curr_state = self.get_curr_state()
        graph = self.gnn_sim.process_gnn(curr_state)

        # Compute next poses using (dv, dw) from analytical passive force modules
        curr_state_ = curr_state.reshape(-1, 13, 1)
        pre_next_state_ = pre_contact_state.reshape(-1, 13, 1)
        dv = pre_next_state_[:, 7:10] - curr_state_[:, 7:10]
        dw = pre_next_state_[:, 10:] - curr_state_[:, 10:]
        pos = curr_state_[:, :3] + dt * dv
        quat = torch_quaternion.update_quat(curr_state_[:, 3:7], dw, dt)

        # Compute new node poses and node dv from passive forces
        pf_node_pos = self.data_processor.pose2node(
            torch.hstack([pos, quat])
        )
        body_mask = graph.body_mask.flatten()
        pf_dv = zeros(graph.node_pos.shape, ref_tensor=graph.node_pos)
        pf_dv[body_mask] = (pf_node_pos - graph.node_pos[body_mask]
                            ) / dt.squeeze(-1)

        # Update node state
        graph['pf_dv'] = pf_dv
        graph['p_node_vel'] = graph.p_node_vel + pf_dv
        graph['p_node_pos'] = graph.p_node_pos + pf_dv * dt.squeeze(-1)

        # Compute next SE(3) state from curr and next node positions
        next_state = self.data_processor.node2pose(
            graph.p_node_pos[body_mask],
            graph.node_pos[body_mask],
            self.robot.num_nodes_per_rod
        )

        return next_state, graph
