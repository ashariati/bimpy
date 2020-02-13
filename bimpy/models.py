import numpy as np
import networkx as nx
import collections
import itertools


class CuttingPlane(object):

    def __init__(self, coefficients):
        assert isinstance(coefficients, np.ndarray), "Expected type ndarray, got %s instead" % type(coefficients)
        self.coefficients = coefficients


class Cell2D(object):

    def __init__(self, edges=None):
        if edges is None:
            edges = set()
        assert isinstance(edges, set), "Expected type set, got %s instead" % type(edges)
        self.edges = edges
        self.evidence = []


class CellComplex2D(object):

    def __init__(self, z_ref, width, length):

        self._z_ref = z_ref
        self._vertices = [np.array([width / 2, length / 2, z_ref]),
                          np.array([-width / 2, length / 2, z_ref]),
                          np.array([-width / 2, -length / 2, z_ref]),
                          np.array([width / 2, -length / 2, z_ref])]

        self._edges = {(0, 1), (1, 2), (2, 3), (3, 0)}

        cell_init = Cell2D({e for e in self._edges})
        self._cells = {cell_init}

        self._edge_cells = {e: {cell_init} for e in self._edges}

        cp_right = CuttingPlane(np.array([1, 0, 0, -width / 2]))
        cp_left = CuttingPlane(np.array([1, 0, 0, width / 2]))
        cp_top = CuttingPlane(np.array([0, 1, 0, -length / 2]))
        cp_bottom = CuttingPlane(np.array([0, 1, 0, length / 2]))
        self._edge_plane = {(0, 1): cp_top, (1, 2): cp_left, (2, 3): cp_bottom, (3, 0): cp_right}

    def insert_partition(self, plane):

        assert isinstance(plane, CuttingPlane), "Requires instance of CuttingPlane"

        # skip z planes if they end up here
        if np.isclose(np.abs(plane.coefficients[2]), 1):
            return

        # determine which elements are intersected by the new cutting plane
        vertices = np.array(self._vertices)
        side = np.sign(np.dot(vertices, plane.coefficients[:3]) + plane.coefficients[3])
        hit_edges = [(i, j) for (i, j) in self._edges if side[i] != side[j]]
        hit_cells = set(itertools.chain(*[self._edge_cells[e] for e in hit_edges]))

        edge_splits = {}
        for e in hit_edges:

            # add new vertex
            hit_plane = self._edge_plane[e]
            A = np.array([plane.coefficients[:3], hit_plane.coefficients[:3], [0, 0, 1]])
            b = -np.array([plane.coefficients[3], hit_plane.coefficients[3], self._z_ref])
            new_vertex = np.linalg.solve(A, b)
            vertex_id = len(self._vertices)
            self._vertices.append(new_vertex)

            # new edges
            i, j = e
            e1 = (i, vertex_id)
            e2 = (vertex_id, j)

            # save to splits
            edge_splits[e] = (e1, e2)

            # update books
            self._edges.discard(e)
            self._edges.add(e1)
            self._edges.add(e2)
            del self._edge_plane[e]
            self._edge_plane[e1] = hit_plane
            self._edge_plane[e2] = hit_plane
            del self._edge_cells[e]

        for c in hit_cells:

            # split edges on c
            split_edges = [e for e in c.edges if e in edge_splits]
            assert (len(split_edges) == 2), "Non-convex cell encountered. This shouldn't happen"

            # new edges
            splits = [edge_splits[split_edges[0]], edge_splits[split_edges[1]]]
            new_edge = (splits[0][0][1], splits[1][0][1])

            # remove split edges from the current cell
            c_edges = c.edges.difference(set(split_edges))

            # insert new sub-segments
            c_edges.add(splits[0][0])
            c_edges.add(splits[0][1])
            c_edges.add(splits[1][0])
            c_edges.add(splits[1][1])

            # partition into sub cells
            c1 = Cell2D({new_edge})
            c2 = Cell2D({new_edge})
            for e in c_edges:

                if e in self._edge_cells:
                    self._edge_cells[e].discard(c)

                i, j = e
                i_side = side[i] if i < len(side) else 0
                j_side = side[j] if j < len(side) else 0
                e_side = np.sign(i_side + j_side)

                if e_side > 0:
                    c1.edges.add(e)
                    self._edge_cells.setdefault(e, set()).add(c1)
                else:
                    c2.edges.add(e)
                    self._edge_cells.setdefault(e, set()).add(c2)

                # TODO: Partition evidence as well

            # update books
            self._cells.discard(c)
            self._cells.add(c1)
            self._cells.add(c2)
            self._edges.add(new_edge)
            self._edge_cells[new_edge] = {c1, c2}
            self._edge_plane[new_edge] = plane

    def delete_partition(self, plane):

        plane_edge = {}
        for (k, v) in self._edge_plane.items():
            plane_edge.setdefault(v, []).append(k)

        for e in plane_edge[plane]:

            i, j = e
            c1, c2 = self._edge_cells[e]

            # merged cell
            c = Cell2D(c1.edges.union(c2.edges))

            # remove all references to c1 and c2 within {edge:cells}
            for f in c.edges:
                self._edge_cells[f].discard(c1)
                self._edge_cells[f].discard(c2)

            # remove e from c
            c.edges.discard(e)

            # compute new edges which bypass vertices i and j now that e will be eliminated
            i_incident = [None, None]
            j_incident = [None, None]
            for f in c.edges:
                k, l = f
                if i == l:
                    i_incident[0] = f
                elif i == k:
                    i_incident[1] = f
                elif j == l:
                    j_incident[0] = f
                elif j == k:
                    j_incident[1] = f
                else:
                    continue
            i_bridged = (i_incident[0][0], i_incident[1][1])
            j_bridged = (j_incident[0][0], j_incident[1][1])

            # remove edges incident to i and j
            c.edges.discard(i_incident[0])
            c.edges.discard(i_incident[1])
            c.edges.discard(j_incident[0])
            c.edges.discard(j_incident[1])

            # add new edges that bypass i and j
            c.edges.add(i_bridged)
            c.edges.add(j_bridged)

            # insert new edge references to c
            for f in c.edges:
                if f not in self._edge_cells:
                    self._edge_cells[f] = set()
                self._edge_cells[f].add(c)

            # update books
            self._edges.discard(e)
            self._edges.discard(i_incident[0])
            self._edges.discard(i_incident[1])
            self._edges.discard(j_incident[0])
            self._edges.discard(j_incident[1])
            self._edges.add(i_bridged)
            self._edges.add(j_bridged)
            self._cells.discard(c1)
            self._cells.discard(c2)
            self._cells.add(c)
            del self._edge_plane[e]
            del self._edge_cells[e]

        # remove discarded edges from {edge:cells}
        for e in list(self._edge_cells.keys()):
            if len(self._edge_cells[e]) == 0:
                del self._edge_cells[e]

    def update_partitions(self, planes):
        pass

    def insert_boundary(self, boundary):
        pass

    def insert_evidence(self, evidence):
        pass

    def cell_graph(self):
        G = nx.Graph()
        for e in self._edge_cells:
            cells = self._edge_cells[e]
            assert (len(cells) == 1 or len(cells) == 2), "{edge:cells} map corrupted"
            if len(cells) == 2:
                c1, c2 = cells
                G.add_edge(c1, c2)
        return G

    @property
    def vertices(self):
        return self._vertices.copy()

    @property
    def edges(self):
        return self._edges.copy()

    @property
    def cells(self):
        return self._cells.copy()





