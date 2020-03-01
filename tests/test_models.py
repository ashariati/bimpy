import unittest
import numpy as np
import networkx as nx
import models
import simulator
import utilities

import matplotlib.pyplot as plt


class TestCellComplex2D(unittest.TestCase):

    def test_add1(self):

        cell_complex = models.CellComplex2D(0, 20, 10)
        cutting_plane = models.Plane.from_axis_distance(np.array([1, 1, 0]), 1)

        cell_complex.insert_partition(cutting_plane)

        G = nx.to_networkx_graph({0: [1], 1: [0]})
        H = cell_complex.cell_graph()

        self.assertTrue(nx.is_isomorphic(G, H))
        self.assertEqual(6, len(cell_complex.vertices))

    def test_add2(self):

        cell_complex = models.CellComplex2D(0, 20, 10)
        cp1 = models.Plane.from_axis_distance(np.array([1, 1, 0]), 1)
        cp2 = models.Plane.from_axis_distance(np.array([1, 1, 0]), 2)

        cell_complex.insert_partition(cp1)
        cell_complex.insert_partition(cp2)

        G = nx.to_networkx_graph({0: [1], 1: [0, 2], 2: [1]})
        H = cell_complex.cell_graph()

        self.assertTrue(nx.is_isomorphic(G, H))
        self.assertEqual(8, len(cell_complex.vertices))

    def test_add3(self):

        cell_complex = models.CellComplex2D(0, 20, 10)
        cp1 = models.Plane.from_axis_distance(np.array([1, 1, 0]), 1)
        cp2 = models.Plane.from_axis_distance(np.array([1, 1, 0]), 2)
        cp3 = models.Plane.from_axis_distance(np.array([-1, 1, 0]), 1)

        cell_complex.insert_partition(cp1)
        cell_complex.insert_partition(cp2)
        cell_complex.insert_partition(cp3)

        G = nx.to_networkx_graph({0: [1, 2], 1: [0, 3], 2: [0, 3, 4], 3: [1, 2, 5], 4: [2, 5], 5: [3, 4]})
        H = cell_complex.cell_graph()

        self.assertTrue(nx.is_isomorphic(G, H))
        self.assertEqual(12, len(cell_complex.vertices))
        self.assertEqual(17, len(cell_complex.edges))

    def test_add4_random(self):

        cell_complex = models.CellComplex2D(0, 10, 10)

        np.random.seed(3)

        planes = simulator.random_planes(num_planes=4, max_distance=9, c=0)
        for p in planes:
            cell_complex.insert_partition(p)

        H = cell_complex.cell_graph()
        G = nx.to_networkx_graph({0: [1, 2, 4], 1: [0, 6], 2: [0, 3, 6], 3: [2, 4, 7, 8], 4: [0, 3, 5],
                                  5: [4, 8], 6: [1, 2, 7], 7: [6, 3], 8: [3, 5]})
        self.assertTrue(nx.is_isomorphic(G, H))

        cell_complex.draw(scene_graph=H)

    def test_multi_add_big(self):

        cell_complex = models.CellComplex2D(1, 20, 10)

        np.random.seed(12)

        planes = simulator.random_planes(num_planes=60, max_distance=9, c=0)
        for p in planes:
            cell_complex.insert_partition(p)

    def test_multi_add_small(self):

        cell_complex = models.CellComplex2D(0, 10, 10)

        np.random.seed(1)

        planes = simulator.random_planes(num_planes=5, max_distance=9, c=0)
        for p in planes:
            cell_complex.insert_partition(p)

        G = nx.to_networkx_graph({0: [1], 1: [0, 2], 2: [1, 3, 5, 7], 3: [2, 4], 4: [3, 5],
                                  5: [2, 4, 6], 6: [5, 7], 7: [6, 2]})
        H = cell_complex.cell_graph()
        self.assertTrue(nx.is_isomorphic(G, H))
        self.assertEqual(16, len(cell_complex.vertices))
        self.assertEqual(23, len(cell_complex.edges))

    def test_partition_with_evidence_simple(self):

        evidence = [simulator.ellipse(5, 3, 4, -3,)]
        cell_complex = models.CellComplex2D(-3, 20, 10, evidence=evidence)

        cutting_plane = models.Plane.from_axis_distance(np.array([1, 1, 0]), 1)

        cell_complex.insert_partition(cutting_plane)

        G = nx.to_networkx_graph({0: [1], 1: [0]})
        H = cell_complex.cell_graph()

        self.assertTrue(nx.is_isomorphic(G, H))
        self.assertEqual(6, len(cell_complex.vertices))

        node_evidence_vertices = 0
        for node in H.nodes:
            self.assertTrue(len(node.evidence) == 1)
            node_evidence_vertices += len(node.evidence[0].vertices)
        self.assertEqual(len(evidence[0].vertices), node_evidence_vertices-2)

        cell_complex.draw(scene_graph=H)

    def test_partition_with_boundary1(self):

        cell_complex = models.CellComplex2D(0, 20, 10)
        cutting_plane = models.Plane.from_axis_distance(np.array([1, 1, 0]), 1)
        boundary = simulator.rectangle(cutting_plane, 2, 1, 4, 2)

        cell_complex.insert_partition(cutting_plane)
        cell_complex.insert_boundary(boundary)

        G = nx.to_networkx_graph({0: [], 1: []})
        H = cell_complex.cell_graph(coverage_threshold=1)

        self.assertTrue(nx.is_isomorphic(G, utilities.filter_boundary_edges(H)))
        self.assertEqual(6, len(cell_complex.vertices))

        cell_complex.draw(scene_graph=H)

    def test_partition_with_boundary2(self):

        cell_complex = models.CellComplex2D(0, 20, 10)
        cutting_plane = models.Plane.from_axis_distance(np.array([1, 1, 0]), 1)
        boundary = simulator.rectangle(cutting_plane, 2, 1, 4, 2)

        cell_complex.insert_partition(cutting_plane)
        cell_complex.insert_boundary(boundary)

        G = nx.to_networkx_graph({0: [1], 1: [0]})
        H = cell_complex.cell_graph(coverage_threshold=4.1)

        self.assertTrue(nx.is_isomorphic(G, utilities.filter_boundary_edges(H)))
        self.assertEqual(6, len(cell_complex.vertices))

        cell_complex.draw(scene_graph=H)

    def test_partition_with_boundary3(self):

        cell_complex = models.CellComplex2D(0, 20, 10)
        cutting_plane = models.Plane.from_axis_distance(np.array([1, 1, -0.5]), 1)
        boundary = simulator.rectangle(cutting_plane, 2, 1, 4, 2)

        cell_complex.insert_partition(cutting_plane)
        cell_complex.insert_boundary(boundary)

        G = nx.to_networkx_graph({0: [], 1: []})
        H = cell_complex.cell_graph(coverage_threshold=1)

        self.assertTrue(nx.is_isomorphic(G, utilities.filter_boundary_edges(H)))
        self.assertEqual(6, len(cell_complex.vertices))

        cell_complex.draw(scene_graph=H)

    def test_partition_with_boundary4(self):

        cell_complex = models.CellComplex2D(0, 20, 10)
        cutting_plane = models.Plane.from_axis_distance(np.array([1, 1, -0.5]), 1)
        boundary = simulator.rectangle(cutting_plane, 2, 1, 100, 2)

        cell_complex.insert_partition(cutting_plane)
        cell_complex.insert_boundary(boundary)

        G = nx.to_networkx_graph({0: [], 1: []})
        H = cell_complex.cell_graph(coverage_threshold=15)

        self.assertTrue(nx.is_isomorphic(G, utilities.filter_boundary_edges(H)))
        self.assertEqual(6, len(cell_complex.vertices))

        cell_complex.draw(scene_graph=H)

    def test_partition_with_boundary5(self):

        cell_complex = models.CellComplex2D(2, 20, 10)
        cutting_plane = models.Plane.from_axis_distance(np.array([1, 1, -0.5]), 1)
        boundary = simulator.rectangle(cutting_plane, 7.5, 1, 4, 2)

        cell_complex.insert_partition(cutting_plane)
        cell_complex.insert_boundary(boundary)

        G = nx.to_networkx_graph({0: [1], 1: [0]})
        H = cell_complex.cell_graph(coverage_threshold=1)

        self.assertTrue(nx.is_isomorphic(G, utilities.filter_boundary_edges(H)))
        self.assertEqual(6, len(cell_complex.vertices))

        cell_complex.draw(scene_graph=H)

    def test_partition_with_boundary6(self):

        cell_complex = models.CellComplex2D(-4, 20, 10)
        cutting_plane = models.Plane.from_axis_distance(np.array([1, 1, -0.5]), 1)
        boundary = simulator.rectangle(cutting_plane, 7.5, 1, 4, 2)

        cell_complex.insert_partition(cutting_plane)
        cell_complex.insert_boundary(boundary, height_threshold=2)

        G = nx.to_networkx_graph({0: [], 1: []})
        H = cell_complex.cell_graph(coverage_threshold=1)

        self.assertTrue(nx.is_isomorphic(G, utilities.filter_boundary_edges(H)))
        self.assertEqual(6, len(cell_complex.vertices))

        cell_complex.draw(scene_graph=H)

    def test_partition_with_boundary7(self):

        cell_complex = models.CellComplex2D(-4, 20, 10)
        cutting_plane = models.Plane.from_axis_distance(np.array([1, 1, -0.5]), 1)
        boundary = simulator.rectangle(cutting_plane, 9.5, 1, 4, 2)

        cell_complex.insert_partition(cutting_plane)
        cell_complex.insert_boundary(boundary, height_threshold=2)

        G = nx.to_networkx_graph({0: [1], 1: [0]})
        H = cell_complex.cell_graph(coverage_threshold=1)

        self.assertTrue(nx.is_isomorphic(G, utilities.filter_boundary_edges(H)))
        self.assertEqual(6, len(cell_complex.vertices))

        cell_complex.draw(scene_graph=H)


class TestConvexPolygon(unittest.TestCase):

    def test_simple(self):

        vertices = [np.array([2, 1, 3]), np.array([-2, -1, 3]), np.array([-2, 1, 3]), np.array([2, -1, 3])]
        edges = {(0, 2), (2, 1), (1, 3), (3, 0)}
        cp = models.ConvexPolygon2D(vertices=vertices, edges=edges)

        cutting_plane = models.Plane(np.array([0.7071, 0.7071, 0, -1]))

        cp1, cp2 = cp.partition(cutting_plane)

        self.assertEqual(3, len(cp1.vertices))
        self.assertEqual(5, len(cp2.vertices))
        self.assertTrue(np.isclose(8, cp.area()))
        self.assertTrue(np.isclose(cp.area(), cp1.area() + cp2.area()))

    def test_ellipsoid(self):

        np.random.seed(1)

        evidence = simulator.random_evidence(3, -1, 1, [-10, -5], [10, 5])
        for e in evidence:
            e.draw()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()




if __name__ == '__main__':
    unittest.main()
