import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.data import DataLoader
import mol2graph
import requests
import os
import networkx as nx


"""This code uses the pytorch geometric library"""


def get_data():
    train_name = "solubility.train.sdf"
    test_name = "solubility.test.sdf"
    try:
        if not os.path.isfile(f'./{train_name}'):
            url = "https://raw.githubusercontent.com/rdkit/rdkit/master/Docs/Book/data/solubility.train.sdf"
            r = requests.get(url)
            with open(f'./{train_name}', 'wb') as f:
                f.write(r.content)

        if not os.path.isfile(f'./{test_name}'):
            url = "https://raw.githubusercontent.com/rdkit/rdkit/master/Docs/Book/data/solubility.test.sdf"
            r = requests.get(url)
            with open(f'./{test_name}', 'wb') as f:
                f.write(r.content)
    except:
        print("Error downloading data.")


def plot_graph(graph_obj):
    edges_raw = graph_obj.edge_index.numpy()
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
    labels = graph_obj.y.numpy()

    G = nx.Graph()
    G.add_nodes_from(list(range(np.max(edges_raw))))
    G.add_edges_from(edges)
    plt.subplot(111)
    options = {
        'node_size': 30,
        'width': 0.2,
    }
    nx.draw(G, with_labels=False, cmap=plt.cm.tab10, font_weight='bold', **options)
    plt.show()


n_features = 75


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(n_features, 128, cached=False)  # if cache=True, the shape of batch must be same!
        self.bn1 = BatchNorm1d(128)
        self.conv2 = GCNConv(128, 64, cached=False)
        self.bn2 = BatchNorm1d(64)
        self.fc1 = Linear(64, 64)
        self.bn3 = BatchNorm1d(64)
        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, 3)

    def forward(self, data):
        # print(data.batch.shape)
        x, edge_index = data.x, data.edge_index
        # print(x.shape)
        # for gi in range(data.batch.shape[0]):
        #     graph_ids = (data.batch == gi).nonzero().squeeze()
        #     x[graph_ids, :] /= len(graph_ids)
        #     #print(graph_ids)
        #     #print(x)
        #     #break

        # sys.exit()
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = global_add_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        x = self.bn3(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


def get_accuracy(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


def train(model, train_loader, n_epochs, val_loader=None):
    hist = {"loss":[], "acc":[], "val_acc":[]}

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = F.nll_loss(output, batch.y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.num_graphs

        train_acc = get_accuracy(train_loader)
        val_acc = None
        if val_loader:
            val_acc = get_accuracy(val_loader)
            hist['val_acc'].append(val_acc)
        epoch_loss /= (len(train_loader)*train_loader.batch_size)
        hist['loss'].append(epoch_loss)
        hist['acc'].append(train_acc)

        print(f'Epoch: {epoch}, Loss: {epoch_loss:.3f}, Train acc: {train_acc:.3f}, Val acc: {val_acc:.3f}', )

    return hist


if __name__ == '__main__':
    get_data()

    train_mols = [m for m in Chem.SDMolSupplier('solubility.train.sdf')]
    test_mols = [m for m in Chem.SDMolSupplier('solubility.test.sdf')]
    sol_cls_dict = {'(A) low': 0, '(B) medium': 1, '(C) high': 2}

    print(sol_cls_dict)

    train_x = [mol2graph.mol2vec(m) for m in train_mols]
    for i, data in enumerate(train_x):
        y = sol_cls_dict[train_mols[i].GetProp('SOL_classification')]
        data.y = torch.tensor([y], dtype=torch.long)

    test_x = [mol2graph.mol2vec(m) for m in test_mols]
    for i, data in enumerate(test_x):
        y = sol_cls_dict[test_mols[i].GetProp('SOL_classification')]
        data.y = torch.tensor([y], dtype=torch.long)

    print(f'Number of graphs: {len(train_x)}')
    first_sample = train_x[0]
    print(f'Looking at first example ..')
    print(f'\t{train_x[0]}')
    print(f'\t # of nodes: {first_sample.x.shape[0]}')
    print(f'\t # of features per node: {first_sample.x.shape[1]}')
    print(f'\t # of edges: {first_sample.edge_index.shape[1]//2}')
    print(f'\t # of features per edge: {first_sample.edge_attr.shape[1]}')

    # plot_graph(train_x[0])

    train_loader = DataLoader(train_x, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(test_x, batch_size=64, shuffle=True, drop_last=True)

    for batch in train_loader:
        print(batch)
        print(batch.num_graphs)
        break

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net()
    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    n_epochs = 100
    hist = train(model, train_loader, n_epochs, val_loader)
    ax = plt.subplot(1, 1, 1)
    ax.plot([e for e in range(n_epochs)], hist["loss"], label="train_loss")
    ax.plot([e for e in range(n_epochs)], hist["acc"], label="train_acc")
    ax.plot([e for e in range(n_epochs)], hist["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    ax.legend()
    plt.savefig('./outputs/loss_and_acc.png')
    plt.show()

