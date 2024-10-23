from typing import Any, List, Dict, Tuple

import torch
from torch import nn
from torch.nn.modules.module import T
from torch_geometric.data import Data as GraphData
from torch_geometric.nn import MessagePassing


def build_mlp(
        input_size: int,
        hidden_layer_sizes: List[int],
        output_size: int = None,
        output_activation: nn.Module = nn.Identity,
        activation: nn.Module = nn.ReLU) -> nn.Module:
    """
    Method to build MLP

    @param input_size: size of input
    @param hidden_layer_sizes: list of hidden layer sizes
    @param output_size: size of output
    @param output_activation: activation function specific for output
    @param activation: activation for all other layers
    @return: mlp object
    """
    # Size of each layer
    layer_sizes = [input_size] + hidden_layer_sizes
    if output_size:
        layer_sizes.append(output_size)

    # Number of layers
    nlayers = len(layer_sizes) - 1

    # Create a list of activation functions and
    # set the last element to output activation function
    act = [activation for i in range(nlayers)]
    act[-1] = output_activation

    # Create a torch sequential container
    mlp = nn.Sequential()
    for i in range(nlayers):
        mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i],
                                                 layer_sizes[i + 1]))
        mlp.add_module("Act-" + str(i), act[i]())

    return mlp


class Encoder(nn.Module):

    def __init__(self,
                 n_out: int,
                 nmlp_layers: int,
                 mlp_hidden_dim: int,
                 node_types: Dict[str, int],
                 edge_types: Dict[str, int]):
        """
        Encoder to take features to a higher dimensional space

        @param n_out: output size of encoder
        @param nmlp_layers: number of mlp layers
        @param mlp_hidden_dim: dim of hidden layers in mlp
        @param node_types: dictionary of node types and their input sizes
        @param edge_types: dictionary of edge types and their input sizes
        """

        def mlp(in_feats):
            """
            method to quickly augment mlp with LayerNorm as last layer
            @param in_feats:
            @return:
            """
            return nn.Sequential(
                *[build_mlp(in_feats,
                            [mlp_hidden_dim
                             for _ in range(nmlp_layers)],
                            n_out),
                  nn.LayerNorm(n_out)]
            )

        super().__init__()
        self.edge_encoding_len = n_out
        self.enum_node_types = {n: i for i, n in enumerate(node_types.keys())}
        self.enum_edge_types = {n: i for i, n in enumerate(edge_types.keys())}

        self.node_encoders = nn.ParameterDict({
            name: mlp(in_dim) for name, in_dim in node_types.items()
        })

        self.edge_encoders = nn.ParameterDict({
            name: mlp(in_dim) for name, in_dim in edge_types.items()
        })

    def double(self: T) -> T:
        super().double()
        for k, v in self.node_encoders.items():
            self.node_encoders[k] = v.double()

        for k, v in self.edge_encoders.items():
            self.edge_encoders[k] = v.double()

        return self

    def forward(self, graph: GraphData) -> GraphData:
        edge_type = graph.edge_type.flatten()

        # Encode node features
        for node_name, n in self.enum_node_types.items():
            encoded = self.node_encoders[node_name](graph[node_name + '_x'])
            graph[node_name + '_x'] = encoded

        # Encode edge features
        for edge_name, n in self.enum_edge_types.items():
            encoded = self.edge_encoders[edge_name](graph[edge_name + "_edge_attr"])
            graph[edge_name + "_edge_attr"] = encoded
            graph[edge_name + "_edge_index"] = (
                graph.edge_index[:, edge_type == n])

        return graph


class BaseInteractionNetwork(MessagePassing):

    def __init__(
            self,
            nnode_in: int,
            nedge_in: int,
            n_out: int,
            nmlp_layers: int,
            mlp_hidden_dim: int,
    ):
        """
        Base interaction network that does message passing on the lower level

        @param nnode_in: input node feat size
        @param nedge_in: input edge feat size
        @param n_out: output size
        @param nmlp_layers: number of mlp layers
        @param mlp_hidden_dim: size of hidden layers
        """
        # Aggregate features from neighbors
        super().__init__(aggr='add')
        # Edge MLP
        self.msg_fn = nn.Sequential(
            *[build_mlp(nnode_in + nnode_in + nedge_in,
                        [mlp_hidden_dim
                         for _ in range(nmlp_layers)],
                        n_out),
              nn.LayerNorm(n_out)]
        )

        self.update_fn = nn.Sequential(
            *[build_mlp(nnode_in + nnode_in + nedge_in,
                        [mlp_hidden_dim
                         for _ in range(nmlp_layers)],
                        n_out),
              nn.LayerNorm(n_out)]
        )

    def double(self: T) -> T:
        super().double()
        self.msg_fn = self.msg_fn.double()
        self.update_fn = self.update_fn.double()

        return self

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass of model

        @param x: node feats
        @param edge_index: edge indices
        @param edge_attr: edge feats
        @return: updated node feats
        """
        x_prop, edges_prop = self.propagate(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        return x_prop

    def message(self,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor
                ) -> torch.Tensor:
        """
        Message passing step
        @param x_i: node feats of one end of edges
        @param x_j: node feats of other end of edges
        @param edge_attr: edge feats
        @return: updated edges
        """
        concat_vec = torch.concat([x_i, x_j, edge_attr], dim=1)
        msg = self.msg_fn(concat_vec)

        edge_attr[:] += msg

        return edge_attr

    def update(self,
               x_updated: torch.Tensor,
               x: torch.Tensor,
               edge_attr: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Do nothing
        """
        return x_updated, edge_attr


class InteractionNetwork(nn.Module):

    def __init__(
            self,
            nnode_in: int,
            nedge_in: int,
            n_out: int,
            nmlp_layers: int,
            mlp_hidden_dim: int,
            edge_types: List[str]
    ):
        """
        Interaction network that does message passing on the higher level

        @param nnode_in: node input feature vector size
        @param nedge_in: edge input feature vector size
        @param n_out: output size
        @param nmlp_layers: number of mlp layers
        @param mlp_hidden_dim: size of hidden layers
        @param edge_types: edge types in graph
        """
        # Aggregate features from neighbors
        super().__init__()

        # Node MLP
        self.update_fn = nn.Sequential(*[
            build_mlp(nnode_in + len(edge_types) * n_out,
                      [mlp_hidden_dim for _ in range(nmlp_layers)],
                      n_out),
            nn.LayerNorm(n_out)
        ])

        # Edge message passing networks
        kwargs = {
            "nnode_in": nnode_in,
            "nedge_in": nedge_in,
            "n_out": n_out,
            "nmlp_layers": nmlp_layers,
            "mlp_hidden_dim": mlp_hidden_dim
        }
        self.mp_dict = nn.ParameterDict({
            name: BaseInteractionNetwork(**kwargs)
            for name in edge_types
        })

    def double(self: T) -> T:
        super().double()
        for k, v in self.mp_dict.items():
            self.mp_dict[k] = v.double()

        self.update_fn = self.update_fn.double()

        return self

    def forward(self, graph: GraphData) -> GraphData:
        """
        forward pass

        @param graph: batch graph data
        @return: updated graph
        """
        agg_xs = []
        for name, mp in self.mp_dict.items():
            edge_attr = graph[name + "_edge_attr"]
            edge_idx = graph[name + "_edge_index"]
            agg_x = mp(graph.node_x, edge_idx, edge_attr)
            agg_xs.append(agg_x)

        agg_xs = torch.stack(agg_xs, dim=2).sum(dim=2)
        concat_vec = torch.hstack([graph.node_x, agg_xs])
        graph['node_x'] = graph.node_x + self.update_fn(concat_vec)

        return graph


class Processor(MessagePassing):

    def __init__(
            self,
            nnode_in: int,
            nedge_in: int,
            n_out: int,
            nmessage_passing_steps: int,
            nmlp_layers: int,
            mlp_hidden_dim: int,
            edge_types: List[str],
            processor_shared_weights: bool = False,
    ):
        """
        Processor containing multiple interaction networks

        @param nnode_in: node input feature vector size
        @param nedge_in: edge input feature vector size
        @param n_out: output size
        @param nmlp_layers: number of mlp layers
        @param mlp_hidden_dim: size of hidden layers
        @param edge_types: edge types in graph
        @param processor_shared_weights: flag to have shared or sep weights per interaction network
        """
        def interaction_network():
            return InteractionNetwork(
                nnode_in=nnode_in,
                nedge_in=nedge_in,
                n_out=n_out,
                nmlp_layers=nmlp_layers,
                mlp_hidden_dim=mlp_hidden_dim,
                edge_types=edge_types
            )

        super().__init__(aggr='add')
        self.num_msg_passes = nmessage_passing_steps
        self.shared_weights = processor_shared_weights
        # Create a stack of L Graph Networks GNs.
        if self.shared_weights:
            self.gnn_stacks = interaction_network()
        else:
            self.gnn_stacks = nn.ModuleList([
                interaction_network()
                for _ in range(nmessage_passing_steps)
            ])

    def double(self: T) -> T:
        super().double()
        if isinstance(self.gnn_stacks, list):
            for i in range(len(self.gnn_stacks)):
                self.gnn_stacks[i] = self.gnn_stacks[i].double()
        else:
            self.gnn_stacks = self.gnn_stacks.double()

        return self

    def forward(self, graph_batch: GraphData) -> GraphData:
        """
        Forward pass

        @param graph_batch: graph data
        @return: message passed graph
        """
        for i in range(self.num_msg_passes):
            graph_batch = self.gnn_stacks(graph_batch) \
                if self.shared_weights \
                else self.gnn_stacks[i](graph_batch)

        return graph_batch


class Decoder(nn.Module):
    def __init__(
            self,
            nnode_in: int,
            nnode_out: int,
            nmlp_layers: int,
            mlp_hidden_dim: int):
        """
        Decoder that maps node latent vectors to normalized dv

        @param nnode_in: node input feature vector size
        @param nedge_in: edge input feature vector size
        @param nmlp_layers: number of mlp layers
        @param mlp_hidden_dim: size of hidden layers
        """
        super().__init__()
        self.node_decode_fn = build_mlp(
            nnode_in,
            [mlp_hidden_dim for _ in range(nmlp_layers)],
            nnode_out
        )

    def double(self: T) -> T:
        super().double()
        self.node_decode_fn = self.node_decode_fn.double()
        return self

    def forward(self, graph_batch: GraphData) -> GraphData:
        """
        forward pass

        @param graph_batch: graph data
        @return: decoded graph data
        """
        graph_batch['decode_output'] = self.node_decode_fn(graph_batch.node_x)

        return graph_batch


class EncodeProcessDecode(nn.Module):

    def __init__(
            self,
            node_types: Dict[str, int],
            edge_types: Dict[str, int],
            n_out: int,
            latent_dim: int,
            nmessage_passing_steps: int,
            nmlp_layers: int,
            mlp_hidden_dim: int,
            processor_shared_weights: bool = False
    ):
        """
        GNN class

        @param node_types: dictionary of node types to input vec size
        @param edge_types: dictionary of edge types to input vec size
        @param n_out: output size
        @param latent_dim: latent dim size
        @param nmessage_passing_steps: number of msg passing steps
        @param nmlp_layers: number of mlp layers per mlp
        @param mlp_hidden_dim: hidden dim size
        @param processor_shared_weights: flag for shared or sep weights for interaction networks
        """
        super().__init__()
        self._encoder = Encoder(
            n_out=latent_dim,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            node_types=node_types,
            edge_types=edge_types
        )
        self._processor = Processor(
            nnode_in=latent_dim,
            nedge_in=latent_dim,
            n_out=latent_dim,
            nmessage_passing_steps=nmessage_passing_steps,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            processor_shared_weights=processor_shared_weights,
            edge_types=list(edge_types.keys())
        )
        self._decoder = Decoder(
            nnode_in=latent_dim,
            nnode_out=n_out,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )

    def double(self: T) -> T:
        super().double()
        self._encoder = self._encoder.double()
        self._processor = self._processor.double()
        self._decoder = self._decoder.double()

        return self

    def forward(self,
                graph_batch: GraphData):
        """
        forward pass

        @param graph_batch: graph data
        @return: GNN processed graph data
        """
        graph_batch = self._encoder(graph_batch)
        graph_batch = self._processor(graph_batch)
        graph_batch = self._decoder(graph_batch)

        return graph_batch
