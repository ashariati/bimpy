import networkx as nx


def filter_boundary_edges(G):

    H = nx.MultiGraph()

    for u in G.nodes:
        H.add_node(u)

    for (u, v, data) in G.edges(data=True):

        boundary = data['boundary_interval']
        if boundary is None:
            H.add_edge(u, v, **data)
        else:
            data['shared_edge'] = None
            H.add_edge(u, u, **data)
            H.add_edge(v, v, **data)

    return H

