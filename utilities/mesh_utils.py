import torch


def tri_mesh_to_graph(faces):
    senders = torch.hstack([
        faces[:, 0],
        faces[:, 1],
        faces[:, 0],
        faces[:, 2],
        faces[:, 1],
        faces[:, 2],
    ])

    receivers = torch.hstack([
        faces[:, 1],
        faces[:, 0],
        faces[:, 2],
        faces[:, 0],
        faces[:, 2],
        faces[:, 1],
    ])

    edge_index = torch.vstack([senders, receivers])

    return edge_index


def shape_matching(x, x0):
    """
    Assume x, x0: (batch, 3, num_verts)
    """
    t0 = x0.mean(dim=-1, keepdim=True)
    tx = x.mean(dim=-1, keepdim=True)

    # get nodes centered at zero
    q = x0 - t0
    p = x - tx

    # solve the system to find best transformation that matches the rest shape
    mat_pq = torch.matmul(p, q.transpose(1, 2))
    w, s, vh = torch.linalg.svd(mat_pq, full_matrices=False)
    rx = torch.matmul(w, vh)

    trans = tx - t0
    return trans, rx


def remove_intersect(verts1,
                     verts2,
                     faces1,
                     faces2,
                     obj1_not_in_obj2,
                     obj2_not_in_obj1):
    verts1 = verts1[obj1_not_in_obj2]
    verts2 = verts2[obj2_not_in_obj1]

    filter1 = torch.isin(faces1, torch.where(obj1_not_in_obj2)[0]).all(dim=1)
    filter2 = torch.isin(faces2, torch.where(obj2_not_in_obj1)[0]).all(dim=1)

    faces1 = re_enum_faces(faces1[filter1], obj1_not_in_obj2)
    faces2 = re_enum_faces(faces2[filter2], obj2_not_in_obj1)

    return verts1, verts2, faces1, faces2


def re_enum_faces(faces: torch.Tensor, keep_bool_tensor: torch.Tensor):
    d_tensor = torch.cumsum(keep_bool_tensor, dim=0) - 1
    faces = faces.apply_(lambda x: d_tensor[x])

    return faces


def connect_2_rings(ring1: torch.Tensor, ring2: torch.Tensor):
    assert len(ring1) == len(ring2)

    ring1 = ring1.reshape(-1, 1)
    ring2 = ring2.reshape(-1, 1)

    next_idx = [i for i in range(1, ring1.shape[0])] + [0]
    faces1 = torch.hstack([ring1, ring1[next_idx], ring2])
    faces2 = torch.hstack([ring1[next_idx], ring2[next_idx], ring2])
    faces = torch.vstack([faces1, faces2])

    return faces


def remap_verts_to_shape(rigid_body, world_verts):
    body_verts = rigid_body.body_verts

    com_pos = rigid_body.mesh_to_com(world_verts)
    rot_mats = rigid_body.mesh_to_rot_mat(world_verts)

    body_vecs = (torch.matmul(rot_mats, body_verts.T)
                 .transpose(1, 2)
                 .reshape(-1, 3))

    com_pos_expanded = com_pos.repeat(1, body_verts.shape[0])
    com_pos_expanded = com_pos_expanded.view(-1, com_pos.shape[1])

    world_verts = body_vecs.clone()
    world_verts += com_pos_expanded

    return world_verts


def batch_2d_to_3d(batch, num_verts):
    batch3d = torch.stack(torch.split(batch.T, num_verts, dim=1), dim=0)
    return batch3d


def batch_3d_to_2d(batch):
    split = torch.split(batch.transpose(1, 2),
                        split_size_or_sections=1,
                        dim=0)
    batch_2d = torch.hstack(split).squeeze(0)

    return batch_2d


def plot_graph(verts, all_edges):
    if not isinstance(all_edges, list):
        all_edges = [all_edges]

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use('macosx')

    edge_colors = ['black', 'red', 'blue', 'blue']

    if isinstance(verts, torch.Tensor):
        verts = verts.numpy()

    for i in range(len(all_edges)):
        if isinstance(all_edges[i], torch.Tensor):
            all_edges[i] = all_edges[i].numpy()
        all_edges[i] = all_edges[i].T.tolist()

    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]

    ax = plt.figure().add_subplot(projection='3d')

    # ax.plot_trisurf(x, z, y, triangles=faces.numpy(), linewidth=0.2, antialiased=True)
    ax.scatter(x, z, y, color='orange', s=1)
    for i, edges in enumerate(all_edges):
        for edge in edges:
            ax.plot(
                [x[e] for e in edge],
                [z[e] for e in edge],
                [y[e] for e in edge],
                color=edge_colors[i]
            )

    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_zlim(-2.0, 2.0)
    plt.show()
