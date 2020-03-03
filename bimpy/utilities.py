import networkx as nx


def filter_boundary_edges(G):

    H = nx.MultiGraph()

    for u in G.nodes:
        H.add_node(u)

    for (u, v, data) in G.edges(data=True):

        boundary = data['boundary_interval']
        if boundary is None:
            H.add_edge(u, v, boundary_interval=None)
        else:
            H.add_edge(u, u, boundary_interval=boundary)
            H.add_edge(v, v, boundary_interval=boundary)

    return H

