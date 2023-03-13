import torch


def naive_index_max(y_shape, indices, t):
    y = torch.ones(y_shape) * -torch.inf
    for i, v in zip(indices, t):
        y[i] = torch.max(y[i], v)
    return y


def index_max(y_shape, i, t):
    assert y_shape[1:] == t.shape[1:]
    assert i.ndim == 1
    assert i.size(0) == t.size(0)

    num_nodes = y_shape[0]
    eq_tbl = torch.eq(torch.arange(num_nodes), i[:, None])
    cs = eq_tbl.cumsum(0)
    max_edge_per_node = eq_tbl.sum(0).max()

    y = torch.ones((max_edge_per_node,) + y_shape) * -torch.inf
    y[cs[torch.arange(cs.size(0)), i] - 1, i] = t
    return y.max(0).values


num_nodes = 100
num_edges = 1000

y_shape = (num_nodes, 3)
indices = torch.randint(high=num_nodes, size=(num_edges,))
t = torch.rand(num_edges, 3)

expected = naive_index_max(y_shape, indices, t)

actual = index_max(y_shape, indices, t)

torch.testing.assert_close(expected, actual)
