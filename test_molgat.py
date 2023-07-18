import pytest
import torch
from torch_geometric.data import DataLoader
from molgnn.pygdata import RedDB
from molgat.models import MolGAT
import os.path as osp
from molgnn.molfeatures import  GenMolFeatures

@pytest.fixture
def dataloader():
    ## Load the RedDB dataset in PyG graph format
    path = osp.join(osp.dirname(osp.realpath('__file__')),  'data', 'reddb')
    dataset = RedDB(root_dir=path,
                  name='reddb2.csv',
                  smi_idx=-2,
                  target_idx=-1,pre_transform=GenMolFeatures()).shuffle()
    # Normalize targets to mean = 0 and std = 1.
    dataset.data.y = dataset.data.y*27.2114 #1Hartree=27.2114eV
    r_mean = dataset.data.y.mean()
    r_std = dataset.data.y.std()
    dataset.data.y = (dataset.data.y - r_mean) / r_std
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return loader






@pytest.fixture
def model():
    node_features = 99
    hidden_dim = 512
    edge_features = 12
    num_heads = 4
    dropout = 0.1
    num_conv_layers = 3
    num_fc_layers = 3
    return MolGAT(node_features, hidden_dim, edge_features, num_heads, dropout, num_conv_layers, num_fc_layers)

def test_model_forward(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for data in dataloader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch, data.edge_attr)
        assert output.shape == (data.num_graphs, 1)

