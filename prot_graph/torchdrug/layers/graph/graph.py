
import torch

from torchdrug import data
from torchdrug.core import Registry as R
from torchdrug.layers.geometry import GraphConstruction


@R.register("layers.BondNetworkConstruction")
class BondNetworkConstruction(GraphConstruction):

    def apply_node_layer(self, protein):

        graph = protein
        for layer in self.node_layers:
            graph = layer(graph)

        protein = protein.subresidue(
            protein.atom2residue[
                protein.atom_name == protein.atom_name2id["CA"]
            ].unique()
        )

        return graph, protein

    def apply_edge_layer(self, res_graph, protein):

        if not self.edge_layers:
            return res_graph

        edge_list = list()
        num_edges = list()
        num_relations = list()
        for layer in self.edge_layers:
            if hasattr(layer, "atom2res") and layer.atom2res:
                edges, n_relation = layer(protein)
                edges = self.to_res_edges(edges, protein)
            else:
                edges, n_relation = layer(res_graph)
            edge_list.append(edges)
            num_edges.append(len(edges))
            num_relations.append(n_relation)

        edge_list = torch.cat(edge_list)
        num_edges = torch.tensor(num_edges, device=protein.device)
        num_relations = torch.tensor(num_relations, device=protein.device)
        num_relation = num_relations.sum()
        offsets = (
            num_relations.cumsum(0) - num_relations
        ).repeat_interleave(num_edges)
        edge_list[:, 2] += offsets

        # reorder edges into a valid PackedGraph
        node_in = edge_list[:, 0]
        edge2graph = res_graph.node2graph[node_in]
        order = edge2graph.argsort()
        edge_list = edge_list[order]
        num_edges = edge2graph.bincount(minlength=res_graph.batch_size)
        offsets = (
            res_graph.num_cum_nodes - res_graph.num_nodes
        ).repeat_interleave(num_edges)

        if hasattr(self, "edge_%s" % self.edge_feature):
            edge_feature = getattr(
                self,
                "edge_%s" % self.edge_feature
            )(res_graph, edge_list, num_relation)

        elif self.edge_feature is None:
            edge_feature = None
        else:
            raise ValueError("Unknown edge feature `%s`" % self.edge_feature)
        data_dict, meta_dict = res_graph.data_by_meta(include=(
            "node", "residue", "node reference", "residue reference", "graph"
        ))

        if isinstance(res_graph, data.PackedProtein):
            data_dict["num_residues"] = res_graph.num_residues
        if isinstance(res_graph, data.PackedMolecule):
            data_dict["bond_type"] = torch.zeros_like(edge_list[:, 2])
        return type(res_graph)(
            edge_list, num_nodes=res_graph.num_nodes, num_edges=num_edges,
            num_relation=num_relation, view=res_graph.view, offsets=offsets,
            edge_feature=edge_feature, meta_dict=meta_dict, **data_dict
        )

    def to_res_edges(self, edges, protein):

        return torch.stack(
            [
                protein.atom2residue[edges[:, 0]],
                protein.atom2residue[edges[:, 1]],
                edges[:, 2]
            ]
        ).t()

    def forward(self, protein):

        res_graph, protein = self.apply_node_layer(protein)
        graph = self.apply_edge_layer(res_graph, protein)

        return graph
