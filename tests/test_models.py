import unittest
import numpy as np
import networkx as nx
import models


class TestCellComplex2D(unittest.TestCase):

    def test_simple_add(self):

        cell_complex = models.CellComplex2D(0, 20, 10)
        cutting_plane = models.CuttingPlane(np.array([0.7071, 0.7071, 0, -1]))

        cell_complex.insert_partition(cutting_plane)

        G = nx.to_networkx_graph({0: [1], 1: [0]})
        H = cell_complex.cell_graph()

        self.assertTrue(nx.is_isomorphic(G, H))
        self.assertEqual(6, len(cell_complex.vertices))

    def test_double_add(self):

        cell_complex = models.CellComplex2D(0, 20, 10)
        cp1 = models.CuttingPlane(np.array([0.7071, 0.7071, 0, -1]))
        cp2 = models.CuttingPlane(np.array([0.7071, 0.7071, 0, -2]))

        cell_complex.insert_partition(cp1)
        cell_complex.insert_partition(cp2)

        G = nx.to_networkx_graph({0: [1], 1: [0, 2], 2: [1]})
        H = cell_complex.cell_graph()

        self.assertTrue(nx.is_isomorphic(G, H))
        self.assertEqual(8, len(cell_complex.vertices))

    def test_triple_add(self):

        cell_complex = models.CellComplex2D(0, 20, 10)
        cp1 = models.CuttingPlane(np.array([0.7071, 0.7071, 0, -1]))
        cp2 = models.CuttingPlane(np.array([0.7071, 0.7071, 0, -2]))
        cp3 = models.CuttingPlane(np.array([-0.7071, 0.7071, 0, -1]))

        cell_complex.insert_partition(cp1)
        cell_complex.insert_partition(cp2)
        cell_complex.insert_partition(cp3)

        G = nx.to_networkx_graph({0: [1, 2], 1: [0, 3], 2: [0, 3, 4], 3: [1, 2, 5], 4: [2, 5], 5: [3, 4]})
        H = cell_complex.cell_graph()

        self.assertTrue(nx.is_isomorphic(G, H))
        self.assertEqual(12, len(cell_complex.vertices))

    def test_multi_add_big(self):

        cell_complex = models.CellComplex2D(1, 20, 10)

        num_planes = 100

        np.random.seed(12)
        coefs = np.random.rand(num_planes, 4) - 0.5
        coefs[:, 2] = 0
        coefs[:, :3] = coefs[:, :3] / np.linalg.norm(coefs[:, :3], axis=1)[:, None]
        coefs[:, 3] = 19 * coefs[:, 3]

        planes = [models.CuttingPlane(c) for c in coefs]

        for p in planes:
            cell_complex.insert_partition(p)

        # cell_complex.draw()

    def test_multi_add_small(self):
        cell_complex = models.CellComplex2D(0, 10, 10)

        num_planes = 5

        np.random.seed(1)
        coefs = np.random.rand(num_planes, 4) - 0.5
        coefs[:, 2] = 0
        coefs[:, :3] = coefs[:, :3] / np.linalg.norm(coefs[:, :3], axis=1)[:, None]
        coefs[:, 3] = 19 * coefs[:, 3]

        planes = [models.CuttingPlane(c) for c in coefs]
        for p in planes:
            cell_complex.insert_partition(p)

        G = nx.to_networkx_graph({0: [1], 1: [0, 2], 2: [1, 3, 5, 7], 3: [2, 4], 4: [3, 5],
                                  5: [2, 4, 6], 6: [5, 7], 7: [6, 2]})
        H = cell_complex.cell_graph()
        self.assertTrue(nx.is_isomorphic(G, H))
        self.assertEqual(16, len(cell_complex.vertices))
        self.assertEqual(23, len(cell_complex.edges))

    def test_add3_remove1(self):

        cell_complex = models.CellComplex2D(0, 20, 10)
        cp1 = models.CuttingPlane(np.array([0.7071, 0.7071, 0, -1]))
        cp2 = models.CuttingPlane(np.array([0.7071, 0.7071, 0, -2]))
        cp3 = models.CuttingPlane(np.array([-0.7071, 0.7071, 0, -1]))

        cell_complex.insert_partition(cp1)
        cell_complex.insert_partition(cp2)
        cell_complex.insert_partition(cp3)
        cell_complex.delete_partition(cp3)

        G = nx.to_networkx_graph({0: [1], 1: [0, 2], 2: [1]})
        H = cell_complex.cell_graph()

        self.assertTrue(nx.is_isomorphic(G, H))
        self.assertEqual(12, len(cell_complex.vertices))
        self.assertEqual(10, len(cell_complex.edges))

    def test_add_remove_random(self):

        cell_complex = models.CellComplex2D(0, 10, 10)

        num_planes = 4

        np.random.seed(3)
        coefs = np.random.rand(num_planes, 4) - 0.5
        coefs[:, 2] = 0
        coefs[:, :3] = coefs[:, :3] / np.linalg.norm(coefs[:, :3], axis=1)[:, None]
        coefs[:, 3] = 19 * coefs[:, 3]

        planes = [models.CuttingPlane(c) for c in coefs]
        for p in planes:
            cell_complex.insert_partition(p)

        H = cell_complex.cell_graph()
        G = nx.to_networkx_graph({0: [1, 2, 4], 1: [0, 6], 2: [0, 3, 6], 3: [2, 4, 7, 8], 4: [0, 3, 5],
                                  5: [4, 8], 6: [1, 2, 7], 7: [6, 3], 8: [3, 5]})
        self.assertTrue(nx.is_isomorphic(G, H))

        cell_complex.delete_partition(planes[-1])

        H = cell_complex.cell_graph()
        G = nx.to_networkx_graph({0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2, 4], 4: [3]})
        self.assertTrue(nx.is_isomorphic(G, H))

    def test_add_remove_add_random(self):

        cell_complex = models.CellComplex2D(3, 20, 14)

        num_planes = 6

        np.random.seed(3)
        coefs = np.random.rand(num_planes, 4) - 0.5
        coefs[:, 2] = 0
        coefs[:, :3] = coefs[:, :3] / np.linalg.norm(coefs[:, :3], axis=1)[:, None]
        coefs[:, 3] = 27 * coefs[:, 3]

        planes = [models.CuttingPlane(c) for c in coefs]
        for p in planes[:4]:
            cell_complex.insert_partition(p)

        H = cell_complex.cell_graph()
        G = nx.to_networkx_graph({0: [1, 2, 4], 1: [0, 6], 2: [0, 3, 6], 3: [2, 4, 7, 8], 4: [0, 3, 5],
                                  5: [4, 8], 6: [1, 2, 7], 7: [6, 3], 8: [3, 5]})
        self.assertTrue(nx.is_isomorphic(G, H))

        cell_complex.draw()

        cell_complex.delete_partition(planes[1])
        H = cell_complex.cell_graph()
        G = nx.to_networkx_graph({0: [1, 2, 4], 1: [0, 6], 2: [0, 3, 6], 3: [2, 4, 7], 4: [0, 3],
                                  6: [1, 2, 7], 7: [6, 3]})
        self.assertTrue(nx.is_isomorphic(G, H))

        cell_complex.draw()

        cell_complex.delete_partition(planes[3])
        H = cell_complex.cell_graph()
        G = nx.to_networkx_graph({0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]})
        self.assertTrue(nx.is_isomorphic(G, H))

        cell_complex.draw()


if __name__ == '__main__':
    unittest.main()
