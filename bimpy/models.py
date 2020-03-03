import numpy as np
import networkx as nx
import collections
import itertools
from itertools import chain
import copy

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

    def project(self, v):
        assert isinstance(v, np.ndarray), "Expected type ndarray, got %s instead" % type(v)
        assert (len(v.shape) == 1 and v.shape[0] == 3) or (len(v.shape) > 1 and v.shape[1] == 3), \
            "Vector(s) must be of length 3"
        dist = self.dot(v)
        return v - (dist * self.coefficients[:3])

    @staticmethod
    def from_axis_distance(axis, distance):
        axis = axis / np.linalg.norm(axis)
        coefficients = np.zeros(4)
        coefficients[:3] = axis
        coefficients[3] = -distance
        return Plane(coefficients)

    @staticmethod
    def intersection(plane1, plane2, plane3):

        A = np.concatenate([plane1.coefficients[:3], plane2.coefficients[:3], plane3.coefficients[:3]]).reshape(3, 3)
        b = np.array([-plane1.coefficients[3], -plane2.coefficients[3], -plane3.coefficients[3]])
        return np.linalg.solve(A, b)


class Polygon3D(object):

    def __init__(self, vertices, edges, plane):

        assert isinstance(vertices, np.ndarray), "Expected type ndarray, got %s instead" % type(plane)
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
        line = line / np.linalg.norm(line[:2])

        magnitude = -line[2]
        p_0 = np.array([magnitude * line[0], magnitude * line[1], z_ref])
        n_hat = np.array([-line[1], line[0], 0])

        distances = np.dot((self.vertices - p_0), n_hat)

        interval = np.array([[np.min(distances)], [np.max(distances)]]) * n_hat + p_0
        return interval

    def centroid(self):
        return np.mean(self.vertices, axis=0)


def _partition_edges(edges, condition):

    pos_edges = set()
    neg_edges = set()
    for e in edges:

        i, j = e
        i_side = condition[i] if i < len(condition) else 0
        j_side = condition[j] if j < len(condition) else 0
        e_side = np.sign(i_side + j_side)

        if e_side > 0:
            pos_edges.add(e)
        else:
            neg_edges.add(e)

    return pos_edges, neg_edges


class ConvexPolygon2D(object):

    def __init__(self, vertices, edges, edge_plane=None):

        assert isinstance(vertices, np.ndarray), "Expected type ndarray for vertices, got %s instead" % type(vertices)

        if len(vertices) > 0:
            self._z_ref = vertices[0, 2]
        else:
            self._z_ref = 0

        assert (len(vertices) == len(edges)), "Invalid vertex or edge set"

        self.vertices = vertices
        self.edges = edges

        if edge_plane is None:
            edge_plane = {e: self._edge_to_plane(e) for e in self.edges}
        self._edge_plane = edge_plane

    def _edge_to_plane(self, e):

        x1 = self.vertices[e[0]]
        x2 = self.vertices[e[1]]

        n_hat = x2 - x1
        axis = np.array([-n_hat[1], n_hat[0], n_hat[2]])
        axis = axis / np.linalg.norm(axis)

        d = np.dot(axis, x1)

        return Plane(np.array([axis[0], axis[1], axis[2], -d]))

    def partition(self, plane):

        assert isinstance(plane, Plane), "Expected type Plane, got %s instead" % type(plane)

        # determine which elements are intersected by the new cutting plane
        side = np.sign(plane.dot(self.vertices))
        hit_vertices = np.where(side == 0)[0]
        hit_edges = [(i, j) for (i, j) in self.edges if np.abs(side[i] - side[j]) == 2]

        if len(hit_vertices) == 0 and len(hit_edges) == 0:

            # no intersection
            #

            if np.all(side == 1):
                return self, None
            else:
                return None, self

        elif len(hit_vertices) == 1 and len(hit_edges) == 0:

            # tangent plane
            #

            if np.any(side == 1):
                return self, None
            else:
                return None, self

        elif len(hit_vertices) == 1 and len(hit_edges) == 1:

            # one vertex and one edge hit
            #

            # add new vertex
            hit_plane = self._edge_plane[hit_edges[0]]
            new_vertex = Plane.intersection(plane, hit_plane, Plane(np.array([0, 0, 1, -self._z_ref])))
            vertex_id = self.vertices.shape[0]
            self.vertices = np.append(self.vertices, new_vertex[None, :], axis=0)

            # new edges
            i, j = hit_edges[0]
            e1 = (i, vertex_id)
            e2 = (vertex_id, j)
            e_div = (vertex_id, hit_vertices[0])

            # update books
            self.edges.discard(hit_edges[0])
            self.edges.add(e1)
            self.edges.add(e2)
            self._edge_plane[e1] = hit_plane
            self._edge_plane[e2] = hit_plane
            self._edge_plane[e_div] = plane

            # partition
            pos_edges, neg_edges = _partition_edges(self.edges, side)
            pos_edges.add(e_div)
            neg_edges.add(e_div)
            cp_pos = ConvexPolygon2D.fromsubset(pos_edges, self.vertices, self._edge_plane)
            cp_neg = ConvexPolygon2D.fromsubset(neg_edges, self.vertices, self._edge_plane)

            return cp_pos, cp_neg

        elif len(hit_vertices) == 2 and len(hit_edges) == 0:

            # two vertices hit
            #

            # new dividing edge
            e_div = (hit_vertices[0], hit_vertices[1])

            # partition edges
            pos_edges, neg_edges = _partition_edges(self.edges, side)
            pos_edges.add(e_div)
            neg_edges.add(e_div)
            self._edge_plane[e_div] = plane

            cp_pos = ConvexPolygon2D.fromsubset(pos_edges, self.vertices, self._edge_plane)
            cp_neg = ConvexPolygon2D.fromsubset(neg_edges, self.vertices, self._edge_plane)

            return cp_pos, cp_neg


        elif len(hit_vertices) == 0 and len(hit_edges) == 2:

            # two edges hit
            #

            splits = []
            for e in hit_edges:

                # add new vertex
                hit_plane = self._edge_plane[e]
                new_vertex = Plane.intersection(plane, hit_plane, Plane(np.array([0, 0, 1, -self._z_ref])))
                vertex_id = self.vertices.shape[0]
                self.vertices = np.append(self.vertices, new_vertex[None, :], axis=0)

                # new edges from split
                i, j = e
                e1 = (i, vertex_id)
                e2 = (vertex_id, j)
                splits.append((e1, e2))

                # update books
                self.edges.discard(e)
                self.edges.add(e1)
                self.edges.add(e2)
                self._edge_plane[e1] = hit_plane
                self._edge_plane[e2] = hit_plane

            # new dividing edge
            e_div = (splits[0][0][1], splits[1][0][1])

            pos_edges, neg_edges = _partition_edges(self.edges, side)
            pos_edges.add(e_div)
            neg_edges.add(e_div)
            self._edge_plane[e_div] = plane

            cp_pos = ConvexPolygon2D.fromsubset(pos_edges, self.vertices, self._edge_plane)
            cp_neg = ConvexPolygon2D.fromsubset(neg_edges, self.vertices, self._edge_plane)

            return cp_pos, cp_neg

        else:
            raise RuntimeError("Non-convex shape encountered")

    def area(self):

        valid_ids = self.valid_vertices()
        vertices = self.vertices[valid_ids, :]
        center = np.mean(vertices, axis=0)
        vertices = vertices - center

        angles, order = zip(*sorted([(np.arctan2(v[1], v[0]), i) for i, v in enumerate(vertices)], key=lambda x: x[0]))
        order = np.array(order)

        vertices = vertices[order, :]
        x = vertices[:, 0]
        y = vertices[:, 1]

        y_left = np.zeros_like(y)
        y_left[:-1] = y[1:]
        y_left[-1] = y[0]

        x_left = np.zeros_like(x)
        x_left[:-1] = x[1:]
        x_left[-1] = x[0]

        a = 0.5 * (np.dot(x, y_left) - np.dot(y, x_left))

        return a

    def valid_vertices(self):
        return list(set(chain.from_iterable(self.edges)))

    def rigid(self, R, t):
        self.vertices[:, :2] = np.dot(R, self.vertices[:, :2].T).T + t
        for e in self._edge_plane:
            self._edge_plane[e] = self._edge_to_plane(e)
        return self

    def draw(self):

        for i, j in self.edges:
            vertices = np.array([self.vertices[i], self.vertices[j]])
            plt.plot(vertices[:, 0], vertices[:, 1], 'bo-')

        # for i, u in enumerate(self.vertices):
        #     plt.annotate(str(i), (u[0], u[1]))

    @staticmethod
    def fromsubset(edges, vertices, edge_plane=None):

        present_nodes = list(set(chain.from_iterable(edges)))
        reindex = {i: k for k, i in enumerate(present_nodes)}

        sub_vertices = vertices[present_nodes, :]
        sub_edges = {(reindex[i], reindex[j]) for i, j in edges}

        if edge_plane is not None:
            sub_edge_plane = {(reindex[i], reindex[j]): edge_plane[(i, j)] for i, j in edges}
        else:
            sub_edge_plane = {}

        return ConvexPolygon2D(vertices=sub_vertices,
                               edges=sub_edges,
                               edge_plane=sub_edge_plane)


class SceneNode2D(ConvexPolygon2D):

    def __init__(self, vertices, edges, evidence=None):
        super(SceneNode2D, self).__init__(vertices, edges)
        if evidence is None or len(evidence) == 0:
            self.evidence = []
            self.free_ratio = 0
        else:
            self.evidence = evidence
            # TODO: should be union.area instead of max(area)
            # NOTE: This is way to slow
            self.free_ratio = max([p.area() for p in self.evidence]) / self.area()
            # self.free_ratio = len(self.evidence)

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
            cp = ConvexPolygon2D.fromsubset(self.edges, cell_complex.vertices, cell_complex._edge_plane)
            return SceneNode2D(vertices=cp.vertices, edges=cp.edges, evidence=self.evidence)

    def __init__(self, z_ref, width, length, evidence=None):

        if evidence is None:
            evidence = []
        _evidence = copy.deepcopy(evidence)

        self._z_ref = z_ref
        self._vertices = np.array([[width / 2, length / 2, z_ref],
                                   [-width / 2, length / 2, z_ref],
                                   [-width / 2, -length / 2, z_ref],
                                   [width / 2, -length / 2, z_ref]])

        self._edges = {(0, 1), (1, 2), (2, 3), (3, 0)}

        cell_init = CellComplex2D.Cell(edges={e for e in self._edges}, evidence=_evidence)
        self._cells = {cell_init}

        self._edge_cells = {e: {cell_init} for e in self._edges}

        cp_right = Plane(np.array([1, 0, 0, -width / 2]))
        cp_left = Plane(np.array([1, 0, 0, width / 2]))
        cp_top = Plane(np.array([0, 1, 0, -length / 2]))
        cp_bottom = Plane(np.array([0, 1, 0, length / 2]))
        self._edge_plane = {(0, 1): cp_top, (1, 2): cp_left, (2, 3): cp_bottom, (3, 0): cp_right}

        self._plane_edge = collections.defaultdict(set)
        self._plane_edge[cp_top].add((0, 1))
        self._plane_edge[cp_left].add((1, 2))
        self._plane_edge[cp_bottom].add((2, 3))
        self._plane_edge[cp_right].add((3, 0))

        self._edge_coverage = {}

    def insert_partition(self, plane):

        assert isinstance(plane, Plane), "Expected type Plane, got %s instead" % type(plane)

        # determine which elements are intersected by the new cutting plane
        side = np.sign(plane.dot(self._vertices))
        hit_edges = [(i, j) for (i, j) in self._edges if side[i] != side[j]]
        hit_cells = set(itertools.chain(*[self._edge_cells[e] for e in hit_edges]))

        edge_splits = {}
        for e in hit_edges:

            # add new vertex
            hit_plane = self._edge_plane[e]
            new_vertex = Plane.intersection(plane, hit_plane, Plane(np.array([0, 0, 1, -self._z_ref])))
            vertex_id = self._vertices.shape[0]
            self._vertices = np.append(self._vertices, new_vertex[None, :], axis=0)

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
            del self._edge_cells[e]
            del self._edge_plane[e]
            self._edge_plane[e1] = hit_plane
            self._edge_plane[e2] = hit_plane
            self._plane_edge[hit_plane].discard(e)
            self._plane_edge[hit_plane].add(e1)
            self._plane_edge[hit_plane].add(e2)

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
                if cp_pos is not None:
                    c1.evidence.append(cp_pos)
                if cp_neg is not None:
                    c2.evidence.append(cp_neg)

            # update books
            self._cells.discard(c)
            self._cells.add(c1)
            self._cells.add(c2)
            self._edges.add(new_edge)
            self._edge_cells[new_edge] = {c1, c2}
            self._edge_plane[new_edge] = plane
            self._plane_edge[plane].add(new_edge)

    def insert_boundary(self, boundary, height_threshold=np.inf):

        assert isinstance(boundary, Polygon3D), "Expected type Polygon3D, got %s instead" % type(boundary)

        boundary_height = boundary.centroid()[2]
        if boundary_height > height_threshold or boundary_height < self._z_ref:
            return

        interval = boundary.xy_interval(self._z_ref)
        x1 = interval[0]
        x2 = interval[1]

        for e in self._plane_edge[boundary.plane]:
            v1 = self._vertices[e[0]]
            v2 = self._vertices[e[1]]
            n_hat = v2 - v1

            t1 = (x1 - v1) / n_hat
            t1 = t1[np.logical_not(np.isnan(t1))]
            assert (np.allclose(t1, t1[0])), "intervals and edges not colinear"
            t1 = t1[0]

            t2 = (x2 - v1) / n_hat
            t2 = t2[np.logical_not(np.isnan(t2))]
            assert (np.allclose(t2, t2[0])), "intervals and edges not colinear"
            t2 = t2[0]

            t1, t2 = (t2, t1) if t1 > t2 else (t1, t2)

            t1 = max(t1, 0)
            t2 = min(t2, 1)

            # no intersection
            if t1 > t2:
                continue

            if e not in self._edge_coverage:
                self._edge_coverage[e] = [t1, t2]
            else:
                self._edge_coverage[e][0] = min(t1, self._edge_coverage[e][0])
                self._edge_coverage[e][1] = max(t2, self._edge_coverage[e][1])

    def cell_graph(self, coverage_threshold=1.):

        # if span of the overlap equals or exceeds the threshold, then
        #   discard from edge_cells which tracks cell adjacency
        # if the length of the edge is less than the coverage threshold, use the edge length as the threshold
        def edge_coverage(e, coverage_threshold):
            if e not in self._edge_coverage:
                return
            t1, t2 = self._edge_coverage[e]
            n_hat = self.vertices[e[1]] - self.vertices[e[0]]
            span = np.linalg.norm(t2 * n_hat - t1 * n_hat)
            coverage_threshold = min(coverage_threshold, np.linalg.norm(n_hat) * 0.5)
            if span > coverage_threshold or np.isclose(coverage_threshold, span):
                # interval = [t1 * n_hat + self.vertices[e[0]], t2 * n_hat + self.vertices[e[0]]]
                interval = [self.vertices[e[0]], self.vertices[e[1]]]
                return interval
            else:
                return None

        G = nx.Graph()

        scene_nodes = {c: c.to_scene_node(self) for c in self._cells}
        for c in self._cells:
            G.add_node(scene_nodes[c])

        for e in self._edge_cells:

            cells = self._edge_cells[e]
            assert (len(cells) == 1 or len(cells) == 2), "{edge:cells} map corrupted"

            # skip border edges
            if len(cells) == 1:
                continue

            # if a significant portion of a boundary covers a shared edge, then
            #   the neighboring cells are no longer considered free-adjacent
            c1, c2 = cells
            interval = edge_coverage(e, coverage_threshold)
            G.add_edge(scene_nodes[c1], scene_nodes[c2], boundary_interval=interval)

        return G

    @property
    def vertices(self):
        return copy.copy(self._vertices)

    @property
    def edges(self):
        return copy.copy(self._edges)

    @property
    def cells(self):
        return copy.copy(self._cells)

    def draw(self, scene_graph=None):

        # draw all cell edges first
        for e in self._edges:
            i, j = e
            vertices = np.array([self.vertices[i], self.vertices[j]])
            plt.plot(vertices[:, 0], vertices[:, 1], 'yo-')

        if scene_graph is not None:

            for scene in scene_graph.nodes:
                scene.draw()
                for evidence in scene.evidence:
                    evidence.draw()

            for u, v in scene_graph.edges(keys=False):
                u_center = np.mean(np.array(u.vertices), axis=0)
                v_center = np.mean(np.array(v.vertices), axis=0)
                centers = np.array([u_center, v_center])
                plt.plot(centers[:, 0], centers[:, 1], 'ko-')

            for u, v, data in scene_graph.edges(data=True, keys=False):
                boundary_interval = data['boundary_interval']
                if boundary_interval is not None:
                    boundary_interval = np.array(boundary_interval)
                    plt.plot(boundary_interval[:, 0], boundary_interval[:, 1], 'ro-')

        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()






