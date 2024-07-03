
from unittest import TestCase

import torch

from torchdrug.data import PackedProtein
from torchdrug.layers.geometry import AlphaCarbonNode

from prot_graph.torchdrug.layers.graph import (
    PeptideBondEdge, GetContactsEdge, CompleteEdge, SampleEdge
)
from prot_graph.torchdrug.layers.graph import BondNetworkConstruction


PDB_FPS = ["./tests/test_data/5BQM.pdb", "./tests/test_data/6JXR.pdb"]


class BaseTestEdgeLayer(TestCase):

    __test__ = False

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.protein_pack = PackedProtein.from_pdb(pdb_files=PDB_FPS)
        self.graph_constructor = BondNetworkConstruction(
            node_layers=[AlphaCarbonNode()], edge_layers=[self.edge_layer]
        )
        self.graph_pack = self.graph_constructor(self.protein_pack)

        return

    def test_uniqueness(self):

        assert len(
            torch.unique(self.graph_pack.edge_list, dim=0)
        ) == len(self.graph_pack.edge_list)

    def test_symmetric_edges(self):

        return

    def test_edge_batching(self):

        assert all(
            self.graph_pack.node2graph[
                self.graph_pack.edge_list[:, 0]
            ] == self.graph_pack.node2graph[
                self.graph_pack.edge_list[:, 1]
            ]
        )

        return


class TestPeptideBondEdgeLayer(BaseTestEdgeLayer):

    __test__ = True

    def __init__(self, *args, **kwargs):

        self.edge_layer = PeptideBondEdge()

        super().__init__(*args, **kwargs)

        return


class TestGetContactsEdgeLayer(BaseTestEdgeLayer):

    __test__ = True

    def __init__(self, *args, **kwargs):

        self.edge_layer = GetContactsEdge("hb")

        super().__init__(*args, **kwargs)

        return


class TestCompleteEdgeLayer(BaseTestEdgeLayer):

    __test__ = True

    def __init__(self, *args, **kwargs):

        self.edge_layer = CompleteEdge()

        super().__init__(*args, **kwargs)

        return


class TestSampleEdgeLayer(BaseTestEdgeLayer):

    __test__ = True

    def __init__(self, *args, **kwargs):

        self.edge_layer = SampleEdge(GetContactsEdge("hb"), p=0.5)

        super().__init__(*args, **kwargs)

        return
