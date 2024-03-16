
import functools
from tqdm.contrib.concurrent import process_map
from typing import Any, Dict, List, Union

from torch_geometric.data import Data
from torch_geometric.data import Dataset as TorchDataset

from ...datasets.dataset import Dataset as StructDataset
from ...graphs import ResGraph
from ...graphs.constants import DIST, PEP, HB


class ResGraphDataset(TorchDataset):

    def __init__(
        self, struct_dataset: StructDataset,
        edge_types: Dict[str, Dict[str, Any]],
        node_features: Union[str, List[str]]
    ):

        self.struct_dataset = struct_dataset
        self.graphs = self._build_graphs(struct_dataset, edge_types)

        return

    def _build_graphs(
        self, struct_dataset: StructDataset,
        edge_types: Dict[str, Dict[str, Any]]
    ) -> List[ResGraph]:

        graphs = process_map(
            functools.partial(self._build_graph, struct_dataset, edge_types),
            struct_dataset.ids
        )

        return [graph for graph in graphs if graph is not None]

    def _build_graph(
        self, struct_dataset: StructDataset,
        edge_types: Dict[str, Dict[str, Any]], id: str
    ) -> ResGraph:

        try:

            struct = struct_dataset.load_structure(id)
            graph = ResGraph(struct)

            for edge_type, params in edge_types.items():
                if edge_type == DIST:
                    graph.add_distance_edges(**params)
                elif edge_type == PEP:
                    graph.add_peptide_bonds(**params)
                elif edge_type == HB:
                    graph.add_hydrogen_bonds(**params)

        except:

            graph = None

        return graph

    def __len__(self):

        return len(self.graphs)
