import numpy as np
import networkx as nx
import collections
import itertools

import matplotlib.pyplot as plt


class Plane(object):

    def __init__(self, coefficients):
        assert isinstance(coefficients, np.ndarray), "Expected type ndarray, got %s instead" % type(coefficients)
        assert (coefficients.size == 4), "Coefficients array must be of length 4"
        n_length = np.linalg.norm(coefficients[:3])
        self.coefficients = coefficients / n_length

    def dot(self, v):
        assert isinstance(v, np.ndarray), "Expected type ndarray, got %s instead" % type(v)
        assert (len(v.shape) == 1 and v.shape[0] == 3) or (len(v.shape) > 1 and v.shape[1] == 3), \
            "Vector(s) must be of length 3"
        return np.dot(v, self.coefficients[:3]) + self.coefficients[3]

    @staticmethod
    def from_axis_distance(axis, distance):
        axis = axis / np.linalg.norm(axis)
        coefficients = np.zeros(4)
        coefficients[:3] = axis
        coefficients[3] = -distance
        return Plane(coefficients)


class Polygon3D(object):

    def __init__(self, vertices, edges, plane):

        assert (len(vertices.shape) == 1 and vertices.shape[0] == 3) or \
               (len(vertices.shape) > 1 and vertices.shape[1] == 3), "vertices must be of size nx3"
        assert isinstance(plane, Plane), "Expected type Plane, got %s instead" % type(plane)
        assert np.allclose(plane.dot(vertices), 0), "All vertices must reside on plane"

        self.vertices = vertices
        self.edges = edges
        self.plane = plane

    def xy_interval(self, z_ref):

        """
        Projects the polygons vertices onto the line arising from the intersection of the Polygon's plane and a plane
            residing at z_ref, and returns the resulting line segment

        :param z_ref: the height of the plane intersecting the polygons plane
        :return:
        """

        line = np.array([self.plane.coefficients[0],
                         self.plane.coefficients[1],
                         self.plane.coefficients[2] * z_ref + self.plane.coefficients[3]])

        magnitude = -line[2]
        p_0 = np.array([magnitude * line[0], magnitude * line[1], z_ref])
        n_hat = np.array([-line[1], line[0], 0])

        distances = np.dot((self.vertices - n_hat), n_hat)

        interval = np.array([[np.min(distances)], [np.max(distances)]]) * n_hat + p_0
        return interval

    def centroid(self):
        return np.mean(self.vertices, axis=0)


class ConvexPolygon2D(object):

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

        assert isinstance(plane, Plane), "Expected type Plane, got %s instead" % type(plane)

        # skip z planes if they end up here
        if np.isclose(np.abs(plane.coefficients[2]), 1):
            return

        # determine which elements are intersected by the new cutting plane
        vertices = np.array(self.vertices)
        side = np.sign(plane.dot(vertices))
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

        cp_pos = ConvexPolygon2D.fromsubset(pos_edges, self.vertices)
        cp_neg = ConvexPolygon2D.fromsubset(neg_edges, self.vertices)

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

    def rigid(self, R, t):

        vertices = np.array(self.vertices)[:, :2]
        vertices = np.dot(R, vertices.T).T + t
        self.vertices = [np.array([v[0], v[1], self._z_ref]) for v in vertices]


    def draw(self):

        for i, j in self.edges:
            vertices = np.array([self.vertices[i], self.vertices[j]])
            plt.plot(vertices[:, 0], vertices[:, 1], 'bo-')

        # for i, u in enumerate(self.vertices):
        #     plt.annotate(str(i), (u[0], u[1]))

    @staticmethod
    def fromsubset(edges, vertices):

        nodes = set()
        for e in edges:
            nodes.add(e[0])
            nodes.add(e[1])
        nodes = list(nodes)

        reindex = {i: k for k, i in enumerate(nodes)}

        return ConvexPolygon2D(vertices=[vertices[i] for i in nodes], edges={(reindex[i], reindex[j]) for i, j in edges})


class SceneNode2D(ConvexPolygon2D):

    def __init__(self, vertices, edges, evidence=None):
        super(SceneNode2D, self).__init__(vertices, edges)
        if evidence is None:
            evidence = []
        self.evidence = evidence

    def free_ratio(self):
        # this should be union / area, but lets just use max for now
        return max([p.area() for p in self.evidence]) / self.area()

    def draw(self):

        for i, j in self.edges:
            vertices = np.array([self.vertices[i], self.vertices[j]])
            plt.plot(vertices[:, 0], vertices[:, 1], 'go-')

        # for i, u in enumerate(self.vertices):
        #     plt.annotate(str(i), (u[0], u[1]))


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
            cp = ConvexPolygon2D.fromsubset(self.edges, cell_complex.vertices)
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

        self._planes = set()
        self._boundaries = set()

        self._edge_cells = {e: {cell_init} for e in self._edges}

        cp_right = Plane(np.array([1, 0, 0, -width / 2]))
        cp_left = Plane(np.array([1, 0, 0, width / 2]))
        cp_top = Plane(np.array([0, 1, 0, -length / 2]))
        cp_bottom = Plane(np.array([0, 1, 0, length / 2]))
        self._edge_plane = {(0, 1): cp_top, (1, 2): cp_left, (2, 3): cp_bottom, (3, 0): cp_right}

    def insert_partition(self, plane):

        assert isinstance(plane, Plane), "Expected type Plane, got %s instead" % type(plane)

        if plane in self._planes:
            return

        # skip z planes if they end up here
        if np.isclose(np.abs(plane.coefficients[2]), 1):
            return

        # determine which elements are intersected by the new cutting plane
        vertices = np.array(self._vertices)
        side = np.sign(plane.dot(vertices))
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

        self._planes.add(plane)

    def cell_graph(self):

        scene_nodes = {c: c.to_scene_node(self) for c in self._cells}

        G = nx.Graph()
        for c in self._cells:
            G.add_node(scene_nodes[c])

        for e in self._edge_cells:
            cells = self._edge_cells[e]
            assert (len(cells) == 1 or len(cells) == 2), "{edge:cells} map corrupted"
            if len(cells) == 2:
                c1, c2 = cells
                G.add_edge(scene_nodes[c1], scene_nodes[c2])
        return G

    def draw(self):

        G = self.cell_graph()

        for node in G.nodes:
            node.draw()
            for evidence in node.evidence:
                evidence.draw()

        for u, v in G.edges:

            u_center = np.mean(np.array(u.vertices), axis=0)
            v_center = np.mean(np.array(v.vertices), axis=0)
            centers = np.array([u_center, v_center])

            plt.plot(centers[:, 0], centers[:, 1], 'ko-')

        for b in self._boundaries:
            interval = b.xy_interval(self._z_ref)
            plt.plot(interval[:, 0], interval[:, 1], 'ro-')


        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def insert_boundary(self, boundary, height_threshold=np.inf, coverage_threshold=0.3):
        """
        Severs cell adjacency connections contained in edge_cells based on whether or not an adjoining edge is covered
            by the given boundary. This test is performed by projecting the 3D boundary polygon to the line intersecting
            the reference plane and the plane on which the boundary resides, which yields a line segment colinear to
            all edges in the cell complex which are induced by the same plane. If the edge overlaps the boundary by a
            a significant amount (determined by coverage_threshold), we remove the edge from edge_cells,
            which eliminates the notion of adjacency.

        :param boundary: Polygon3D representing a boundary that divides areas of free space
        :param height_threshold: boundary shapes whose centroid resides above height will be ignored
        :param coverage_threshold: a number between [0, 1] denoting the ratio of coverage required to severe cell connections
        :return:
        """

        assert isinstance(boundary, Polygon3D), "Expected type Polygon3D, got %s instead" % type(boundary)

        boundary_height = boundary.centroid()[2]
        if boundary_height > height_threshold or boundary_height < self._z_ref:
            return

        self._boundaries.add(boundary)

        interval = boundary.xy_interval(self._z_ref)
        x1 = interval[0]
        x2 = interval[1]
        n_hat = x2 - x1

        plane_edge = collections.defaultdict(list)
        for e, p in self._edge_plane.items():
            plane_edge[p].append(e)

        for e in plane_edge[boundary.plane]:
            v1 = self.vertices[e[0]]
            v2 = self.vertices[e[1]]

            t1 = (v1 - x1)[:2] / n_hat[:2]
            assert (np.isclose(t1[0], t1[1])), "intervals and edges not colinear"
            t1 = t1[0]

            t2 = (v2 - x1)[:2] / n_hat[:2]
            assert (np.isclose(t2[0], t2[1])), "intervals and edges not colinear"
            t2 = t2[0]

            t1, t2 = (t2, t1) if t1 > t2 else (t1, t2)

            # if overlap exceeds threshold, discard from edge_cells which tracks cell adjacency
            r = min(t2 - max(t1, 0), 1)
            if r > coverage_threshold:
                del self._edge_cells[e]

    @property
    def vertices(self):
        return self._vertices.copy()

    @property
    def edges(self):
        return self._edges.copy()

    @property
    def cells(self):
        return self._cells.copy()





