
import json

import torch
from torch_geometric.data import Batch
from torch_geometric.nn.models import MLP

from ..models import ProtGCN
from .task import Task


class EnzymeCommissionClassifier(Task):

    def __init__(self, model: ProtGCN, n_mlp_layer: int = 1):

        super(EnzymeCommissionClassifier, self).__init__()

        with open("./data/gearnet/ec_ids.json") as f:
            self.classes = json.load(f)["ids"]

        self.n_classes = len(self.classes)

        self.model = model
        self.n_mlp_layer = n_mlp_layer
        self.mlp = MLP(
            in_channels=self.model.d_output,
            hidden_channels=self.model.d_output, num_layers=n_mlp_layer,
            out_channels=self.n_classes
        )

        return

    def forward(self, batch: Batch):

        output = self.model(batch.x, batch.edge_index, batch=batch.batch)
        y_hat = self.mlp(output)

        label_is = [
            [self.classes.index(batch.y[i][j]) for j in range(len(batch.y[i]))]
            for i in range(len(batch.y))
        ]
        y = torch.zeros(len(label_is), len(self.classes))
        for i in range(len(label_is)):
            for j in range(len(label_is[i])):
                y[i, label_is[i][j]] = 1

        return y_hat, y.float()