import unittest

import numpy as np

import simulator
import models
import estimators


class TestFloorPlan(unittest.TestCase):

    def test_basic_u(self):

        np.random.seed(401)

        z_ref = -1
        width = 10
        length = 10

        # planes
        x_planes = []
        x_offsets = np.cumsum(np.random.randint(1, 3, 6)) - width / 2
        for x in x_offsets:
            x_planes.append(models.Plane.from_axis_distance(axis=np.array([1, 0, 0]), distance=x))
        y_planes = []
        y_offsets = np.cumsum(np.random.randint(1, 3, 5)) - length / 2
        for y in y_offsets:
            y_planes.append(models.Plane.from_axis_distance(axis=np.array([0, 1, 0]), distance=y))

        # boundaries
        boundaries = [simulator.rectangle(x_planes[4], x=np.mean([y_offsets[3], y_offsets[2]]), y=0, w=y_offsets[3]-y_offsets[2]-0.3, h=2),
                      simulator.rectangle(x_planes[5], x=np.mean([y_offsets[4], y_offsets[1]]), y=0, w=y_offsets[4]-y_offsets[1]-0.5, h=2),
                      simulator.rectangle(y_planes[1], x=-np.mean([x_offsets[5], x_offsets[1]]), y=0, w=x_offsets[5]-x_offsets[1]-0.4, h=2),
                      simulator.rectangle(y_planes[2], x=-np.mean([x_offsets[4], x_offsets[1]]), y=0, w=x_offsets[4]-x_offsets[1]-0.2, h=2),
                      simulator.rectangle(y_planes[3], x=-np.mean([x_offsets[4], x_offsets[1]]), y=0, w=x_offsets[4]-x_offsets[1]-0.25, h=2),
                      simulator.rectangle(y_planes[4], x=-np.mean([x_offsets[5], x_offsets[1]]), y=0, w=x_offsets[5]-x_offsets[1]-0.1, h=2)
                      ]

        # evidence
        evidence_index = [((2, 2), (1, 1)),
                          ((4, 2), (2, 1)),
                          ((5, 2), (4, 1)),
                          ((5, 4), (4, 2)),
                          ((4, 4), (3, 3)),
                          ((3, 4), (2, 3)),
                          ((2, 4), (1, 3))]
        evidence = []
        for ev in evidence_index:
            tr_corner = np.array([x_offsets[ev[0][0]], y_offsets[ev[0][1]]])
            bl_corner = np.array([x_offsets[ev[1][0]], y_offsets[ev[1][1]]])
            diff = tr_corner - bl_corner

            center = np.mean(np.array([tr_corner, bl_corner]), axis=0)
            ellipse = simulator.ellipse(diff[0] / 2, diff[1] / 2, 3).rigid(np.eye(2), center)
            evidence.append(ellipse)

        # construct
        cell_complex = models.CellComplex2D(z_ref=z_ref, width=width, length=length, evidence=evidence)
        # cell_complex = models.CellComplex2D(z_ref=z_ref, width=width, length=length)
        for p in x_planes + y_planes:
            cell_complex.insert_partition(p)
        for b in boundaries:
            cell_complex.insert_boundary(b)

        speculator = estimators.FloorPlanSpeculator(cell_complex, horizon=1)
        scene_graph = speculator.floorplan()
        # scene_graph = cell_complex.cell_graph()

        cell_complex.draw(scene_graph)




if __name__ == '__main__':
    unittest.main()
