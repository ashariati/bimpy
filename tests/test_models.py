import unittest
import numpy as np
import networkx as nx
import models


class TestCellComplex2D(unittest.TestCase):

    def test_simple_add(self):

        cell_complex = models.CellComplex2D(0, 20, 10)
        cutting_plane = models.CuttingPlane(np.array([0.7071, 0.7071, 0, -1]))

        cell_complex.insert_partition(cutting_plane)

        print(cell_complex._vertices)

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

    def test_multi_add(self):

        cell_complex = models.CellComplex2D(1, 20, 10)

        np.random.seed(12)

        coefs = np.random.rand(40, 4)
        coefs[:, :3] = coefs[:, :3] / np.linalg.norm(coefs[:, :3], axis=1)[:, None]
        coefs[:, 3] = 18 * (coefs[:, 3] - 0.5)

        planes = [models.CuttingPlane(c) for c in coefs]

        for p in planes:
            cell_complex.insert_partition(p)

        print(nx.to_dict_of_lists(cell_complex.cell_graph()))


if __name__ == '__main__':
    unittest.main()
