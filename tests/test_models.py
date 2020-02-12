import unittest
import numpy as np
import models


class TestCellComplex2D(unittest.TestCase):

    def test_simple(self):

        cell_complex = models.CellComplex2D(0, 20, 10)
        cutting_plane = models.CuttingPlane(np.array([0.7071, 0.7071, 0, 1]))

        cell_complex.insert_partition(cutting_plane)

        print(cell_complex._edges)


if __name__ == '__main__':
    unittest.main()
