import networkx as nx
import numpy as np
import itertools

import models
import utilities


class FloorPlanSpeculator(object):

    def __init__(self, cell_complex, horizon=0, min_ratio=0.0):

        assert isinstance(cell_complex, models.CellComplex2D), "Expected type CellComplex2D, got %s instead" % type(cell_complex)

        self.horizon = horizon
        self.min_ratio = min_ratio
        self.H = cell_complex.cell_graph()
        self.G = utilities.filter_boundary_edges(self.H)

    def floorplan(self):

        remaining = {u for u in self.G.nodes if u.free_ratio > self.min_ratio}

        floorplan_nodes = set()
        while len(remaining) > 0:
            source = remaining.pop()
            conncomp = self.connected_component(source)
            floorplan_nodes |= conncomp
            remaining -= conncomp

        return self.G.subgraph(floorplan_nodes)

    def connected_component(self, source):

        discovered = set()
        distance = [0]
        unexplored = [source]

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

                if v.free_ratio > self.min_ratio:
                    distance.append(d)
                else:
                    distance.append(d + 1)

        return discovered

