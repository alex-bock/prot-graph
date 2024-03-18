
from ast import literal_eval
import functools
from tqdm.contrib.concurrent import process_map
from typing import Any, Dict, List, Union

import numpy as np
from sklearn.preprocessing import OneHotEncoder

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset as TorchDataset
from torch_geometric.utils import dense_to_sparse

from ...constants import AAS
from ...datasets.dataset import Dataset as StructDataset
from ...graphs import ResGraph
from ...graphs.constants import DIST, PEP, HB


class ResGraphDataset(TorchDataset):

    def __init__(
        self, struct_dataset: StructDataset,
        edge_types: Dict[str, Dict[str, Any]],
        node_features: Union[str, List[str]],
        label_field: int,
        flatten: bool = False
    ):

        self.flat = flatten
        self.struct_dataset = struct_dataset
        self.graphs = self._build_graphs(struct_dataset, edge_types)
        self.node_features = self._featurize_nodes(self.graphs, node_features)
        self.labels = self._load_labels(
            struct_dataset, self.graphs, label_field
        )

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
                else:
                    raise NotImplementedError

        except:

            graph = None

        return graph

    def _featurize_nodes(
        self, graphs: List[ResGraph], node_features: Union[str, List[str]]
    ) -> np.ndarray:

        X = list()
        for graph in graphs:
            X.append(self._featurize_graph(node_features, graph))

        return X

    def _featurize_graph(
        self, node_features: Union[str, List[str]], graph: ResGraph
    ) -> np.ndarray:

        if isinstance(node_features, str):
            node_features = [node_features]

        X_g = np.zeros((len(graph.node_df), 0))
        for feature in node_features:
            if feature == "restype":
                encoder = OneHotEncoder(
                    categories=[AAS],       # annoying sklearn dimension stuff
                    sparse_output=False,
                    handle_unknown="ignore"
                )
                x_g = encoder.fit_transform(
                    graph.node_df.type.values.reshape(-1, 1)
                )
            else:
                raise NotImplementedError
            X_g = np.concatenate((X_g, x_g), axis=1)

        return torch.from_numpy(X_g)

    def _load_labels(
        self, struct_dataset: StructDataset, graphs: List[ResGraph],
        label_field: str
    ) -> List[Any]:

        ids = [graph.id for graph in graphs]
        metadata = struct_dataset.metadata.loc[ids]
        labels = metadata[label_field].apply(
            lambda x: literal_eval(x)[0] if isinstance(literal_eval(x), list) else x
        )

        return labels.values

    def __len__(self) -> int:

        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:

        x = self.node_features[idx]
        graph = self.graphs[idx]
        label = self.labels[idx]

        edge_index, _ = dense_to_sparse(
            torch.from_numpy(graph.get_adj_matrix(flatten=self.flat))
        )

        return Data(x=x, edge_index=edge_index, y=label)
