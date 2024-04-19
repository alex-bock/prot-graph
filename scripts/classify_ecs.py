
import json
import os
from pprint import pprint
import sys
sys.path.append(os.getcwd())

import tqdm

import torch
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch import Tensor

from prot_graph.datasets.structure import PDBDataset
from prot_graph.datasets.graph import ResGraphDataset
from prot_graph.models import ProtGCN
from prot_graph.tasks import EnzymeCommissionClassifier
from prot_graph.metrics import f1_max


MODEL_DICT = {"ProtGCN": ProtGCN}


if __name__ == "__main__":

    struct_dataset = PDBDataset(sys.argv[1], n=600)
    struct_dataset.load_metadata("./data/gearnet/ec.csv")

    with open(sys.argv[2], "r") as f:
        params = json.load(f)

    pprint(params)
    graph_params = params["graph_params"]
    model_params = params["model_params"]
    learning_params = params["learning_params"]

    dataset = ResGraphDataset(
        struct_dataset=struct_dataset, label_field="ec",
        split_json="./data/gearnet/splits.json", **graph_params
    )

    train_loader = DataLoader(
        dataset.train, batch_size=learning_params["batch_size"]
    )
    valid_loader = DataLoader(
        dataset.valid, batch_size=learning_params["batch_size"] // 2
    )
    test_loader = DataLoader(
        dataset.test, batch_size=learning_params["batch_size"] // 2
    )

    n_hidden = model_params["n_hidden"]
    d_hidden = model_params["d_hidden"]

    gcn = ProtGCN(
        d_input=dataset.num_features, d_hidden=[d_hidden] * n_hidden,
        concat_hidden=model_params["concat_hidden"]
    )

    classifier = EnzymeCommissionClassifier(model=gcn, n_mlp_layer=3)
    optimizer = AdamW(
        params=classifier.parameters(), **learning_params["optimizer"]
    )

    classifier.train()
    for epoch in range(learning_params["n_epoch"]):
        batch_losses = []
        for batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            y_hat, y = classifier(batch)
            loss = F.binary_cross_entropy_with_logits(y_hat, y)
            batch_losses.append(loss)
            loss.mean(dim=0).backward()
            optimizer.step()
        print(f"Epoch {epoch}: " + str(torch.mean(Tensor(batch_losses))))
        if (epoch + 1) % learning_params["epoch_step"] == 0:
            gcn.eval()
            y_hats = []
            ys = []
            for batch in valid_loader:
                y_hat, y = classifier(batch)
                y_hats.append(y_hat)
                ys.append(y)
            print(f1_max(torch.cat(y_hats), torch.cat(ys)))
            gcn.eval()
            y_hats = []
            ys = []
            for batch in train_loader:
                y_hat, y = classifier(batch)
                y_hats.append(y_hat)
                ys.append(y)
            print(f1_max(torch.cat(y_hats), torch.cat(ys)))
