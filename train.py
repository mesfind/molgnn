#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:53:12 2023

@author: Mesfin Diro
"""

import warnings
warnings.filterwarnings("ignore")

import os.path as osp
import torch
import torch.nn.functional as F
from torch.optim import Adam
from molgnn import  MolGATConv
from pygdata import RedDB
from molfeatures import  GenMolFeatures
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.utils.data import DataLoader
from torch_geometric.nn.norm import BatchNorm
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import rdDepictor
<<<<<<< HEAD
rdDepictor.SetPreferCoordGen(True)
sns.set()

#needed for show_mols
from molgat.utils import NoamLR


=======
from utils.utils import NoamLR
rdDepictor.SetPreferCoordGen(True)
sns.set()

>>>>>>> 29045e8145848d15355692a1de3067107f206d91

## Load the RedDB dataset in PyG graph format
#transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
path = osp.join(osp.dirname(osp.realpath('__file__')),  'data', 'reddb')
mol_reddb = RedDB(root_dir=path,
                  name='reddb2.csv',
                  smi_idx=-2,
                  target_idx=-1,pre_transform=GenMolFeatures()).shuffle()

print(mol_reddb.data)


# Normalize targets to mean = 0 and std = 1.
mol_reddb.data.y = mol_reddb.data.y*27.2114 #1Hartree=27.2114eV
r_mean = mol_reddb.data.y.mean()
r_std = mol_reddb.data.y.std()
mol_reddb.data.y = (mol_reddb.data.y - r_mean) / r_std


# Split the dataset into two

train_size = int(0.9 * len(mol_reddb)) 
test_size = len(mol_reddb) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(mol_reddb, [train_size, test_size])


test_loader = DataLoader(test_dataset, batch_size=192, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=192, shuffle=True)
len(test_dataset)

# define MolGAT model


class MolGAT(torch.nn.Module):
    def __init__(self, node_features, hidden_dim, edge_features, num_heads, dropout, num_conv_layers, num_fc_layers):
        super(MolGAT, self).__init__()
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        self.num_fc_layers = num_fc_layers
        
        self.bn_list.append(BatchNorm(hidden_dim))
        self.conv_list.append(MolGATConv(node_features, hidden_dim, edge_features, heads=num_heads))
        for i in range(num_conv_layers-1):
            self.conv_list.append(MolGATConv(hidden_dim, hidden_dim, edge_features, heads=num_heads))
                
        self.fc_list = torch.nn.ModuleList()
        for i in range(num_fc_layers -1):
            if i == 0:
                self.fc_list.append(torch.nn.Linear(hidden_dim*2, hidden_dim*2))
            else:
                self.fc_list.append(torch.nn.Linear(hidden_dim*2, hidden_dim*2))
        
        self.fc_out = torch.nn.Linear(hidden_dim*2, 1)
        
        self.dropout = dropout

    def forward(self, x, edge_index, batch_index, edge_attr):
        for i, (conv, bn) in enumerate(zip(self.conv_list, self.bn_list)):
            x = F.relu(conv(x, edge_index, edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i != (self.num_fc_layers-1):
                x = bn(x)
        
        x = torch.cat([gmp(x, batch_index),
                       gap(x, batch_index)], dim=1)
        
        for i, fc in enumerate(self.fc_list ):
            x = F.relu(fc(x))
            if i != (self.num_fc_layers-1):
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc_out(x)
        return x
    
    
#  train parameters
class TrainArgs:
    edge_features = mol_reddb.data.edge_attr.shape[1]
    num_features=mol_reddb.num_features
    dropout=0.1
    num_fc_layers=3
    num_conv_layers=3
    num_heads=4
    hidden_dim=512
    batch_size = 192
    init_lr = 1e-4
    max_lr = 1e-3
    final_lr = 1e-4
    num_lrs = 1
    warmup_epochs = 2.0
    epochs = 301
args = TrainArgs()


# define the device and the molgat model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MolGAT(node_features=args.num_features,
                            hidden_dim=args.hidden_dim,
                            edge_features=args.edge_features,
                            num_heads=args.num_heads,
                            dropout=args.dropout,
                            num_fc_layers=args.num_fc_layers,
                            num_conv_layers=args.num_conv_layers).to(device)
## print model parameters
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))


# define the optimizer and schedular

# optimizer
params = [{'params': model.parameters(), 'lr': args.init_lr, 'weight_decay': 0}]
optimizer = Adam(params)

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
<<<<<<< HEAD
PATH='final_models/MolGAT30.pt'
=======
PATH='final_models/MolGAT.pt'
>>>>>>> 29045e8145848d15355692a1de3067107f206d91
torch.save(model.state_dict(),PATH)



# visualize the loss as the network trained
fig = plt.figure(figsize=(6,6))
plt.plot(range(1,len(train_loss)+1),train_loss, label='MolGAT Training Loss')
plt.plot(range(1,len(test_loss)+1),test_loss, label='MolGAT Test Loss')

plt.xlabel('epochs')
plt.ylabel('loss(eV)')
plt.xlim(0, len(train_loss)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
fig.savefig('figs/loss_plot_molgat30.png')
plt.show()




<<<<<<< HEAD
    
=======
    
>>>>>>> 29045e8145848d15355692a1de3067107f206d91
