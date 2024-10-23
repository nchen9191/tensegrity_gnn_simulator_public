from collections import OrderedDict
from typing import Dict, Union, Tuple

import torch

from state_objects.base_state_object import BaseStateObject
from state_objects.cables import get_cable, ActuatedCable, Cable
from state_objects.system_topology import SystemTopology
from state_objects.tensegrity_rods import TensegrityRod
from utilities.tensor_utils import zeros


class TensegrityRobot(BaseStateObject):

    def __init__(self, cfg: Dict):
        """
        Tensegrity robot class

        @param cfg: config dict
        """
        super().__init__(cfg['name'])
        topology_dict = cfg['system_topology']
        self.system_topology = SystemTopology.init_to_torch(
            topology_dict['sites'],
            topology_dict['topology'],
        )
        self.rods = self._init_rods(cfg)
        self.cables = self._init_cables(cfg)

        # Concat of state vars
        self.pos = torch.hstack([rod.pos for rod in self.rods.values()])
        self.linear_vel = torch.hstack([rod.linear_vel for rod in self.rods.values()])
        self.quat = torch.hstack([rod.quat for rod in self.rods.values()])
        self.ang_vel = torch.hstack([rod.ang_vel for rod in self.rods.values()])

        # Split cables to actuated and non-actuated
        self.actuated_cables, self.non_actuated_cables = {}, {}
        for k, cable in self.cables.items():
            if isinstance(cable, ActuatedCable):
                self.actuated_cables[k] = cable
            else:
                self.non_actuated_cables[k] = cable

        # Pre-computed cable consts used to compute forces
        self.k_mat, self.c_mat, self.cable2rod_idxs, self.rod_end_pts \
            = self.build_cable_consts()

        self._init_sites()
        self.cable_map = cfg['act_cable_mapping'] \
            if 'act_cable_mapping' in cfg \
            else {s.name: s.name for s in self.actuated_cables.values()}

    def _init_sites(self):
        """
        Initialize rod sites from system topology
        """
        for rod in self.rods.values():
            for site in rod.sites:
                world_frame_pos = self.system_topology.sites_dict[site].reshape(-1, 3, 1)
                body_frame_pos = rod.world_to_body_coords(world_frame_pos)
                rod.update_sites(site, body_frame_pos)

    def to(self, device: Union[str, torch.device]):
        self.system_topology.to(device)
        for k, rod in self.rods.items():
            rod.to(device)

        for k, cable in self.cables.items():
            cable.to(device)

        self.k_mat = self.k_mat.to(device)
        self.c_mat = self.c_mat.to(device)

        return self

    def update_state(self, pos, lin_vel, quat, ang_vel):
        """
        Method to update state and compute any other kinematic

        @param pos: center of mass positions of all rods
        @param lin_vel: linear vel of com of all rods
        @param quat: quaternion of all rods
        @param ang_vel: angular vel of all rods
        """
        self.pos = pos
        self.linear_vel = lin_vel
        self.quat = quat
        self.ang_vel = ang_vel

        # Up each rod
        for i, rod in enumerate(self.rods.values()):
            rod.update_state(
                pos[:, i * 3: (i + 1) * 3],
                lin_vel[:, i * 3: (i + 1) * 3],
                quat[:, i * 4: (i + 1) * 4],
                ang_vel[:, i * 3: (i + 1) * 3],
            )

        self.update_system_topology()

    def update_system_topology(self):
        """
        Update site positions in sys topology object after updating rod states
        """
        for rod in self.rods.values():
            for site, rel_pos in rod.sites.items():
                world_pos = rod.body_to_world_coords(rel_pos)
                self.system_topology.update_site(site, world_pos)

    def _init_rods(self, config: dict) -> Dict[str, TensegrityRod]:
        """
        Instantiate rod objects
        @param config: config containing rod configs
        @return: dictionary of rod name to rod object
        """
        rods = OrderedDict()
        for rod_config in config['rods']:
            rod_state = TensegrityRod.init_from_cfg(rod_config)
            rods[rod_state.name] = rod_state

        return rods

    def _init_cables(self, config: dict) -> Dict[str, Cable]:
        """
        Instantiate cable objects
        @param config: config containing cable configs
        @return: dictionary of cable name to cable object
        """
        cables = OrderedDict()
        for cable_config in config['cables']:
            cable_cls = get_cable(cable_config['type'])
            config = {k: v for k, v in cable_config.items() if k != 'type'}

            cable = cable_cls.init_from_cfg(
                config
            )
            cables[cable.name] = cable

        return cables

    def _find_rod_idxs(self, cable: Cable) -> Tuple[int, int]:
        end_pt0, end_pt1 = cable.end_pts
        rod_idx0, rod_idx1 = None, None

        for i, rod in enumerate(self.rods.values()):
            if end_pt0 in rod.sites:
                rod_idx0 = i
            elif end_pt1 in rod.sites:
                rod_idx1 = i

        return rod_idx0, rod_idx1

    def build_cable_consts(self) \
            -> Tuple[torch.Tensor, torch.Tensor, Tuple[list, list], dict]:
        """
        Precompute constants used to compute cable forces and torques analytically
        @return:
        """
        k_mat = torch.zeros((1, len(self.rods), len(self.cables)),
                            dtype=self.dtype)
        c_mat = torch.zeros((1, len(self.rods), len(self.cables)),
                            dtype=self.dtype)

        end_pt0_idxs, end_pt1_idxs = [], []
        rod_end_pts = [[None] * len(self.cables) for _ in range(len(self.rods))]
        for j, cable in enumerate(self.cables.values()):
            k = cable.stiffness
            c = cable.damping

            rod_idx0, rod_idx1 = self._find_rod_idxs(cable)
            end_pt0_idxs.append(rod_idx0)
            end_pt1_idxs.append(rod_idx1)
            rod_end_pts[rod_idx0][j] = cable.end_pts[0]
            rod_end_pts[rod_idx1][j] = cable.end_pts[1]

            m, n = (rod_idx0, rod_idx1) if rod_idx1 > rod_idx0 \
                else (rod_idx1, rod_idx0)

            k_mat[0, m, j] = k
            k_mat[0, n, j] = -k

            c_mat[0, m, j] = c
            c_mat[0, n, j] = -c

        return k_mat, c_mat, (end_pt0_idxs, end_pt1_idxs), rod_end_pts

    def get_rest_lengths(self):
        batch_size = self.pos.shape[0]
        rest_lengths = []
        for s in self.cables.values():
            rest_length = s.rest_length
            if rest_length.shape[0] == 1:
                rest_length = rest_length.repeat(batch_size, 1, 1)
            rest_lengths.append(rest_length)
        return torch.concat(rest_lengths, dim=2)

    def get_cable_acting_pts(self):
        ref_tensor = None
        for s in self.rod_end_pts[0]:
            if s:
                ref_tensor = self.system_topology.sites_dict[s]
                break

        act_pts = torch.hstack([
            torch.concat(
                [self.system_topology.sites_dict[s]
                 if s else zeros(ref_tensor.shape, ref_tensor=ref_tensor)
                 for s in cable],
                dim=2
            )
            for cable in self.rod_end_pts
        ])

        return act_pts

    def compute_cable_forces(self):
        endpt_idxs0, endpt_idxs1 = self.cable2rod_idxs
        rods = list(self.rods.values())
        cables = list(self.cables.values())

        rod_pos = torch.concat([rod.pos for rod in rods], dim=2)
        rod_linvel = torch.concat([rod.linear_vel for rod in rods], dim=2)
        rod_angvel = torch.concat([rod.ang_vel for rod in rods], dim=2)

        cable_end_pts0 = torch.concat([self.system_topology.sites_dict[s.end_pts[0]]
                                        for s in cables], dim=2)
        cable_end_pts1 = torch.concat([self.system_topology.sites_dict[s.end_pts[1]]
                                        for s in cables], dim=2)
        cable_vecs = cable_end_pts1 - cable_end_pts0
        cable_lengths = cable_vecs.norm(dim=1, keepdim=True)
        cable_unit_vecs = cable_vecs / cable_lengths

        length_diffs = cable_lengths - self.get_rest_lengths()
        length_diffs = (torch.clamp_min(length_diffs, 0.0)
                        .repeat(1, len(rods), 1))

        vels0 = (
                rod_linvel[..., endpt_idxs0]
                + torch.cross(rod_angvel[..., endpt_idxs0],
                              cable_end_pts0 - rod_pos[..., endpt_idxs0],
                              dim=1)
        )
        vels1 = (
                rod_linvel[..., endpt_idxs1]
                + torch.cross(rod_angvel[..., endpt_idxs1],
                              cable_end_pts1 - rod_pos[..., endpt_idxs1],
                              dim=1)
        )
        rel_vels = torch.linalg.vecdot(
            vels1 - vels0,
            cable_unit_vecs,
            dim=1
        ).unsqueeze(1).repeat(1, len(rods), 1)

        stiffness_force_mags = self.k_mat * length_diffs
        damping_force_mags = self.c_mat * rel_vels
        force_mags = stiffness_force_mags + damping_force_mags

        rod_forces = torch.hstack([
            force_mags[:, i: i + 1] * cable_unit_vecs
            for i in range(len(rods))
        ])
        act_pts = self.get_cable_acting_pts()

        net_rod_forces = rod_forces.sum(dim=2, keepdim=True)

        return net_rod_forces, rod_forces, act_pts

    def compute_cable_length(self, cable: Cablee):
        end_pt0 = self.system_topology.sites_dict[cable.end_pts[0]]
        end_pt1 = self.system_topology.sites_dict[cable.end_pts[1]]

        x_dir = end_pt1 - end_pt0
        length = x_dir.norm(dim=1, keepdim=True)
        x_dir = x_dir / length

        return length, x_dir


class TensegrityRobotGNN(TensegrityRobot):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._inv_mass = None
        self._inv_inertia = None

        self._cable_damping = None
        self._cable_stiffness = None
        self._cable_rest_length = None

        self._cable_template = None

        self.node_mapping = {
            body.name: i * len(rod.rigid_bodies) + j
            for i, rod in enumerate(self.rods.values())
            for j, body in enumerate(rod.rigid_bodies.values())
        }

        self.template = self.get_template_graph()
        self.template_idx = [
            (self.node_mapping[s0], self.node_mapping[s1])
            for s0, s1 in self.template
        ]

        self.num_nodes = len(self.node_mapping)
        self.rod_body_verts = [rod.body_verts for rod in self.rods.values()]
        self.body_verts = torch.vstack(self.rod_body_verts)
        self.num_nodes_per_rod = self.body_verts.shape[0] // len(self.rods)
        self.num_bodies = max(self.node_mapping.values()) + 1

    def to(self, device):
        super().to(device)
        if self._cable_damping is not None:
            self._cable_damping = self._cable_damping.to(device)
        if self._cable_stiffness is not None:
            self._cable_stiffness = self._cable_stiffness.to(device)
        if self._cable_rest_length is not None:
            self._cable_rest_length = self._cable_rest_length.to(device)
        if self._inv_mass is not None:
            self._inv_mass = self._inv_mass.to(device)
        if self._inv_inertia is not None:
            self._inv_inertia = self._inv_inertia.to(device)

        return self

    def get_contact_nodes(self):
        contact_nodes = [
            v for k, v in self.node_mapping.items()
            if 'sphere' in k
        ]

        return contact_nodes

    def get_template_graph(self):
        template = [
            edge
            for rod in self.rods.values()
            for edge in rod.get_template_graph()
        ]

        return template

    def get_cable_edge_idxs(self):
        if self._cable_template is None:
            mapping = {}
            i, j = 0, 0
            for k in self.node_mapping.keys():
                if "sphere" in k:
                    mapping[str(i)] = self.node_mapping[k]
                    i += 1
                if "motor" in k:
                    mapping["b" + str(i)] = self.node_mapping[k]
                    j += 1

            endpt_fn = lambda x, idx: mapping[x.split("_")[idx]]
            if len(list(self.cables.values())[0].end_pts[0].split("_")) > 2:
                self._cable_template = torch.tensor([
                    [endpt_fn(e, 1), endpt_fn(e, 2)]
                    for cable in self.cables.values()
                    for e in cable.end_pts
                ]).T
            else:
                self._cable_template = torch.tensor([
                    [endpt_fn(cable.end_pts[0], 1), endpt_fn(cable.end_pts[1], 1)] if i % 2 == 0 else
                    [endpt_fn(cable.end_pts[1], 1), endpt_fn(cable.end_pts[0], 1)]
                    for cable in self.cables.values()
                    for i in range(2)
                ]).T
        return self._cable_template

    @property
    def cable_damping(self):
        if self._cable_damping is None:
            self._cable_damping = torch.vstack([
                s.damping
                for cable in self.cables.values()
                for s in [cable, cable]
            ])
        return self._cable_damping

    @property
    def cable_stiffness(self):
        if self._cable_stiffness is None:
            self._cable_stiffness = torch.vstack([
                s.stiffness
                for cable in self.cables.values()
                for s in [cable, cable]
            ])
        return self._cable_stiffness

    @property
    def cable_rest_length(self):
        if self._cable_rest_length is None:
            self._cable_rest_length = torch.vstack([
                s._rest_length
                for cable in self.cables.values()
                for s in [cable, cable]
            ])
        return self._cable_rest_length

    @property
    def inv_mass(self):
        if self._inv_mass is None:
            self._inv_mass = torch.vstack([
                1 / n.mass
                for r in self.rods.values()
                for n in r.rigid_bodies.values()
            ])
        return self._inv_mass

    @property
    def inv_inertia(self):
        if self._inv_inertia is None:
            self._inv_inertia = torch.vstack([
                torch.diagonal(n.I_body_inv)
                for r in self.rods.values()
                for n in r.rigid_bodies.values()
            ])
        return self._inv_inertia
