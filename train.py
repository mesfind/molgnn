import warnings
warnings.filterwarnings("ignore")
import os
import os.path as osp
import argparse
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Adam, Optimizer
from molgnn import MolGCNConv, MolGATConv
import pandas as pd
import numpy as np
from pygdata import RedDB
from molfeatures import GenMolGraph, GenMolecules, GenMolFeatures
import torch.nn as nn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.utils.data import DataLoader, Dataset, Sampler
from torch_geometric.data import DataLoader
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn import LayerNorm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from torch.utils.data import random_split
from rdkit.Chem import AllChem, PandasTools, Descriptors
from rdkit.Chem.Draw import IPythonConsole
#IPythonConsole.ipython_useSVG = True
rdDepictor.SetPreferCoordGen(True)
sns.set()

# # misc
from typing import Dict, Iterator, List, Optional, Union, OrderedDict, Tuple
from tqdm.notebook import tqdm
from functools import reduce
from sklearn.metrics import mean_squared_error
from io import BytesIO
from scipy import stats
from IPython.display import SVG
from random import Random
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
#needed for show_mols
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
import cairosvg
import math
#from utils import NoamLR



target = 0
dim = 512

class MyTransform(object):
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, target]
        return data
    
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    def fit_transform(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)
        return (data - self.mean) / self.std
    def transofrm(self, data):
        return (data - self.mean)/self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
    
class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data

## Load the RedDB dataset in PyG graph format
#transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
path = osp.join(osp.dirname(osp.realpath('__file__')),  'data', 'reddb')
mol_reddb = RedDB(root_dir=path,
                  name='reddb2.csv',
                  smi_idx=-2,
                  target_idx=-1,pre_transform=GenMolFeatures()).shuffle()



# Normalize targets to mean = 0 and std = 1.
mol_reddb.data.y = mol_reddb.data.y*27.2114 #1Hartree=27.2114eV
r_mean = mol_reddb.data.y.mean()
r_std = mol_reddb.data.y.std()
mol_reddb.data.y = (mol_reddb.data.y - r_mean) / r_std

# Split datasets. 


train_size = int(0.9 * len(mol_reddb)) 
test_size = len(mol_reddb) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(mol_reddb, [train_size, test_size])



test_loader = DataLoader(test_dataset, batch_size=192, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=192, shuffle=True)



#molgatconv Model
hidden_dim=512
edge_dim = mol_reddb.data.edge_attr.shape[1]
num_features=mol_reddb.num_features
batch=192
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        torch.manual_seed(42)
        self.conv1 = MolGATConv(num_features, hidden_dim, edge_dim,heads=4)
        self.conv2 = MolGATConv(hidden_dim, hidden_dim, edge_dim,heads=4)
        self.conv3 = MolGATConv(hidden_dim, hidden_dim, edge_dim,heads=4)
        self.bn = BatchNorm(in_channels=hidden_dim)
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2,1)
    def forward(self,x, edge_index, batch_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index,edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index,edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv3(x, edge_index,edge_attr))
        x = self.bn(x)
        # Global Pooling (stack different aggregations)
        ### (reason) multiple nodes in one graph....
        ## how to make 1 representation for graph??
        ### use POOLING! 
        ### ( gmp : global MAX pooling, gap : global AVERAGE pooling )
        x = torch.cat([gmp(x, batch_index), 
                            gap(x, batch_index)], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = Net().to(device)
## model parameters
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
#optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)

#  train parameters
class TrainArgs:
    smiles_column = None
    batch_size = 192
    init_lr = 1e-4
    max_lr = 1e-3
    final_lr = 1e-4
    num_lrs = 1
    warmup_epochs = 2.0
    epochs = 301
args = TrainArgs()


# optimizer
params = [{'params': model.parameters(), 'lr': args.init_lr, 'weight_decay': 0}]
optimizer = Adam(params)




class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where :code:`warmup_steps = warmup_epochs * steps_per_epoch`).
    Then the learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr` over the
    course of the remaining :code:`total_steps - warmup_steps` (where :code:`total_steps =
    total_epochs * steps_per_epoch`). This is roughly based on the learning rate
    schedule from `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_, section 5.3.
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):

        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        return list(self.lr)

    def step(self, current_step: int = None):
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]


# scheduler
scheduler = NoamLR(
    optimizer=optimizer,
    warmup_epochs=[args.warmup_epochs],
    total_epochs=[args.epochs] * args.num_lrs,
    steps_per_epoch=len(train_loader) // args.batch_size,
    init_lr=[args.init_lr],
    max_lr=[args.max_lr],
    final_lr=[args.final_lr]
)

# Train function
def train(train_loader_reddb):
    model.train()
    train_loss=0
    for data in train_loader_reddb:
        data = data.to(device)
        model.zero_grad()
        y_pred = model(data.x, data.edge_index, data.batch,data.edge_attr)
        loss_train_mse = F.mse_loss(y_pred, data.y)
        loss_train_mse.backward()
        optimizer.step()
        if isinstance(scheduler, NoamLR):
            scheduler.step()
        train_loss += float(loss_train_mse) * data.num_graphs
    return train_loss / len(train_loader_reddb.dataset)



# Test function

def test(loader):
    model.eval()
    test_loss=0
    for data_t in loader:
        data_t = data_t.to(device)
        with torch.no_grad():
            out = model(data_t.x, data_t.edge_index, data_t.batch,data_t.edge_attr)
            loss_test_mse = F.mse_loss(out, data_t.y)
        test_loss += float(loss_test_mse) * data_t.num_graphs
    return test_loss / len(loader.dataset)




#train  
train_loss = []
val_loss = []
test_loss = []
for epoch in range(1, args.epochs):
    train_mse = train(train_loader)
    test_mse = test(test_loader)
    train_loss.append(train_mse)
    test_loss.append(test_mse)
    if epoch % 1 == 0:
        print(f'Epoch: {epoch:d}, Loss: {train_mse:.7f}, test MSE: {test_mse:.7f}')


        #save the model
PATH='final_models/MolGAT.pt'
torch.save(model.state_dict(),PATH)


# visualize the loss as the network trained
fig = plt.figure(figsize=(6,6))
plt.plot(range(1,len(train_loss)+1),train_loss, label='MolGAT Training Loss')
plt.plot(range(1,len(test_loss)+1),test_loss, label='MolGAT Test Loss')
#plt.plot(range(1,len(val_loss)+1),val_loss, label='Validation loss')
# find position of lowest validation loss
#minposs = train_loss.index(min(train_loss-test_loss))+1
#plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
plt.xlabel('epochs')
plt.ylabel('loss(eV)')
#plt.ylim(0, 0.005) # consistent scale
plt.xlim(0, len(train_loss)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
fig.savefig('figs/loss_plot_molgat.png')



# test the performance of the model

path = osp.join(osp.dirname(osp.realpath('__file__')),  'data', 'reddb')
reddb_mol = GenMolGraph(root_dir=path,
                  name='reddb2.csv',
                  smi_idx=0,pre_transform=GenMolFeatures()).shuffle()


reddb_loader = DataLoader(reddb_mol, batch_size=1, shuffle=False)
yp = []
yr = []
model.eval()
for g in reddb_loader:
    g.to(device)
    out= model(g.x , g.edge_index, g.batch,g.edge_attr)
    yp.append(out.tolist()[0])
    yr.append(g.y.tolist()[0])
df = pd.read_csv('data/reddb/raw/reddb2.csv')
df = df.drop(['reaction_energy'], axis=1)
df['smiles'] = df['smiles']
df['y_real'] = yr
df["y_real"] = df["y_real"].str.get(0)*27.2114
df["y_pred"] = yp
df["y_pred"] = (df["y_pred"].str.get(0)*r_std.numpy())+r_mean.numpy() # inverse transform




r, p = stats.pearsonr(df["y_real"], df["y_pred"])
def plot_oof_preds(ctype, llim, ulim):
        plt.figure(figsize=(6,6))
        mae = mean_absolute_error(df.y_real, df.y_pred)
        rmse = mean_squared_error(df.y_real, df.y_pred, squared=False)
        r, p = stats.pearsonr(df.y_real, df.y_pred)
        sns.scatterplot(x='y_real',y='y_pred',data=df,color='r');
        plt.xlim((llim, ulim))
        plt.ylim((llim, ulim))
        plt.plot([llim, ulim], [llim, ulim],'--k')
        plt.xlabel('Target Reaction Energy(eV)')
        plt.ylabel('Predicted reaction energy(eV)')
        #plt.title(f'{ctype}', fontsize=18)
        ax = plt.gca()
        ax.set_aspect('equal')
        #at = AnchoredText(f"MAE = {mae:.3f}\nRMSE = {rmse:.3f}", prop=dict(size=10),
        #                  frameon=True, loc='upper left')
        #at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        #ax.add_artist(at)
        plt.annotate(f'$MAE = {mae:.3f}, RMSE = {rmse:.3f}$',xy=(0.1, 0.9), xycoords='axes fraction',ha='left', va='center',bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'});
        plt.savefig(f'{ctype}.png', format='png', dpi=300)
        plt.show();
plot_oof_preds('figs/RedoxReaction_molGAT', -6, 6);


