import networkx as nx
import numpy as np
import itertools

import models
import utilities


class FloorPlanSpeculator(object):

    def __init__(self, cell_complex, horizon=0):

        assert isinstance(cell_complex, models.CellComplex2D), "Expected type CellComplex2D, got %s instead" % type(cell_complex)

        self.H = cell_complex.cell_graph()

        self.G = utilities.filter_boundary_edges(self.H)

        self.horizon = horizon

        # find arbitrary starting point with evidence and neighbors
        self.source = None
        for u in self.G.nodes:
            if u.free_ratio > 0 and len(list(self.G.neighbors(u))) > 0:
                self.source = u
                break

    def floorplan(self):

        discovered = {self.source}
        unexplored = list(self.G.neighbors(self.source))
        distance = [0 if u.free_ratio > 0 else 1 for u in unexplored]

        while len(unexplored) > 0:

            u = unexplored.pop()
            d = distance.pop()

            if d > self.horizon:
                continue

            discovered.add(u)

            for v in self.G.neighbors(u):

                if v in discovered:
                    continue

                unexplored.append(v)

                if v.free_ratio > 0:
                    distance.append(d)
                else:
                    distance.append(d + 1)

        return self.H.subgraph(discovered)

