import networkx as nx


def filter_boundary_edges(G):
    H = nx.Graph()
    for u in G.nodes:
        H.add_node(u)
    for (u, v, data) in G.edges(data=True):
        if data['boundary_interval'] is not None:
            continue
        H.add_edge(u, v, **data)
    return H

