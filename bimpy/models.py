import numpy as np
import networkx as nx
import collections
import itertools

import matplotlib.pyplot as plt


class CuttingPlane(object):

    def __init__(self, coefficients):
        assert isinstance(coefficients, np.ndarray), "Expected type ndarray, got %s instead" % type(coefficients)
        assert (coefficients.size == 4), "Coefficients array must be of length 4"
        self.coefficients = coefficients


class ConvexPolygon(object):

    def __init__(self, vertices, edges):

        self._z_ref = vertices[0][2]
        assert (np.allclose(np.array(vertices)[:, 2], self._z_ref)), \
            "Vertices must lie on a plane parallel to the x-y plane"

        self.vertices = vertices
        self.edges = edges

    def _edge_to_plane(self, e):

        x1 = self.vertices[e[0]]
        x2 = self.vertices[e[1]]

        n_hat = x2 - x1
        p_hat = np.array([n_hat[1], -n_hat[0], n_hat[2]])
        d = np.dot(p_hat, x1)

        plane = np.array([n_hat[1], -n_hat[0], n_hat[2], -d])
        return plane

    def partition(self, plane):

        # skip z planes if they end up here
        if np.isclose(np.abs(plane.coefficients[2]), 1):
            return

        # determine which elements are intersected by the new cutting plane
        vertices = np.array(self.vertices)
        side = np.sign(np.dot(vertices, plane.coefficients[:3]) + plane.coefficients[3])
        hit_edges = [(i, j) for (i, j) in self.edges if side[i] != side[j]]

        # no edges were hit
        if len(hit_edges) == 0:
            if np.all(side == 1):
                return self, []
            else:
                return [], self

        assert (len(hit_edges) == 2), "This shouldn't happen, vertices must form non-convex shape"

        splits = []
        for e in hit_edges:

            # add new vertex
            hit_plane = self._edge_to_plane(e)

            A = np.array([plane.coefficients[:3], hit_plane[:3], [0, 0, 1]])
            b = -np.array([plane.coefficients[3], hit_plane[3], -self._z_ref])
            new_vertex = np.linalg.solve(A, b)
            vertex_id = len(self.vertices)
            self.vertices.append(new_vertex)

            # new edges
            i, j = e
            e1 = (i, vertex_id)
            e2 = (vertex_id, j)

            # save to splits
            splits.append((e1, e2))

            # update books
            self.edges.discard(e)
            self.edges.add(e1)
            self.edges.add(e2)

        # partition edges
        new_edge = (splits[0][0][1], splits[1][0][1])
        pos_edges = {new_edge}
        neg_edges = {new_edge}
        for e in self.edges:

            i, j = e
            i_side = side[i] if i < len(side) else 0
            j_side = side[j] if j < len(side) else 0
            e_side = np.sign(i_side + j_side)

            if e_side > 0:
                pos_edges.add(e)
            else:
                neg_edges.add(e)

        cp_pos = ConvexPolygon.fromsubset(pos_edges, self.vertices)
        cp_neg = ConvexPolygon.fromsubset(neg_edges, self.vertices)

        return cp_pos, cp_neg

    def area(self):
        self.sortccw()
        vertices = np.array(self.vertices)
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def sortccw(self):

        vertices = np.array(self.vertices)
        center = np.mean(vertices, axis=0)
        vertices = vertices - center

        angles, order = zip(*sorted([(np.arctan2(v[1], v[0]), i) for i, v in enumerate(vertices)], key=lambda x: x[0]))

        self.vertices = [self.vertices[j] for j in order]
        edges = list(zip(range(len(self.vertices)), range(1, len(self.vertices))))
        edges.append((len(self.vertices)-1, 0))
        self.edges = set(edges)
        return self

    def draw(self):

        for i, j in self.edges:
            vertices = np.array([self.vertices[i], self.vertices[j]])
            plt.plot(vertices[:, 0], vertices[:, 1], 'bo-')

        for i, u in enumerate(self.vertices):
            plt.annotate(str(i), (u[0], u[1]))
        plt.show()

    def totriangles(self):
        pass

    @staticmethod
    def fromsubset(edges, vertices):

        nodes = set()
        for e in edges:
            nodes.add(e[0])
            nodes.add(e[1])
        nodes = list(nodes)

        reindex = {i: k for k, i in enumerate(nodes)}

        return ConvexPolygon(vertices=[vertices[i] for i in nodes], edges={(reindex[i], reindex[j]) for i, j in edges})


class SceneNode2D(ConvexPolygon):

    def __init__(self, vertices, edges, evidence=None):
        super(SceneNode2D, self).__init__(vertices, edges)
        if evidence is None:
            evidence = []
        self.evidence = evidence

    def freespace_ratio(self):
        return 0


class CellComplex2D(object):

    class Cell(object):

        def __init__(self, edges=None, evidence=None):
            if evidence is None:
                evidence = []
            if edges is None:
                edges = set()
            assert isinstance(edges, set), "Expected type set, got %s instead" % type(edges)
            self.edges = edges
            self.evidence = evidence

        def to_scene_node(self, cell_complex):
            cp = ConvexPolygon.fromsubset(self.edges, cell_complex.vertices)
            return SceneNode2D(vertices=cp.vertices, edges=cp.edges, evidence=self.evidence)

    def __init__(self, z_ref, width, length, evidence=None):

        if evidence is None:
            evidence = []

        self._z_ref = z_ref
        self._vertices = [np.array([width / 2, length / 2, z_ref]),
                          np.array([-width / 2, length / 2, z_ref]),
                          np.array([-width / 2, -length / 2, z_ref]),
                          np.array([width / 2, -length / 2, z_ref])]

        self._edges = {(0, 1), (1, 2), (2, 3), (3, 0)}

        cell_init = CellComplex2D.Cell(edges={e for e in self._edges}, evidence=evidence)
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
            b = -np.array([plane.coefficients[3], hit_plane.coefficients[3], -self._z_ref])
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

            # complete edge set
            c_edges = c.edges.difference(set(split_edges))
            c_edges.add(splits[0][0])
            c_edges.add(splits[0][1])
            c_edges.add(splits[1][0])
            c_edges.add(splits[1][1])

            # partition into sub cells
            c1 = CellComplex2D.Cell({new_edge})
            c2 = CellComplex2D.Cell({new_edge})
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

            # partition evidence
            for cp in c.evidence:
                cp_pos, cp_neg = cp.partition(plane)
                c1.evidence.append(cp_pos)
                c2.evidence.append(cp_neg)

            # update books
            self._cells.discard(c)
            self._cells.add(c1)
            self._cells.add(c2)
            self._edges.add(new_edge)
            self._edge_cells[new_edge] = {c1, c2}
            self._edge_plane[new_edge] = plane

    def cell_graph(self):

        scene_nodes = {c: c.to_scene_node(self) for c in self._cells}

        G = nx.Graph()
        for e in self._edge_cells:
            cells = self._edge_cells[e]
            assert (len(cells) == 1 or len(cells) == 2), "{edge:cells} map corrupted"
            if len(cells) == 2:
                c1, c2 = cells
                G.add_edge(scene_nodes[c1], scene_nodes[c2])
        return G

    def draw(self):

        for i, j in self._edges:
            vertices = np.array([self._vertices[i], self._vertices[j]])
            plt.plot(vertices[:, 0], vertices[:, 1], 'bo-')

        for i, u in enumerate(self._vertices):
            plt.annotate(str(i), (u[0], u[1]))

        G = self.cell_graph()
        for u, v in G.edges:

            u_center = np.mean(np.array(u.vertices), axis=0)
            v_center = np.mean(np.array(v.vertices), axis=0)
            centers = np.array([u_center, v_center])

            plt.plot(centers[:, 0], centers[:, 1], 'go-')

        plt.show()

    @property
    def vertices(self):
        return self._vertices.copy()

    @property
    def edges(self):
        return self._edges.copy()

    @property
    def cells(self):
        return self._cells.copy()





