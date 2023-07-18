# -*- coding: utf-8 -*-
"""
Created on Fri Jan 09 14:54:37 2022

@author: Mesfin Diro
"""
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
from custom_pygdata import AqSolDB2, AqSolDB
from solfeatures import GenMolGraph, GenMolecules, GenMolFeatures
import torch.nn as nn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import BatchNorm
from torch.utils.data import DataLoader, Dataset, Sampler
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from torch.utils.data import random_split
from rdkit.Chem import AllChem, PandasTools, Descriptors
from rdkit.Chem.Draw import IPythonConsole
# # misc
from typing import Dict, Iterator, List, Optional, Union, OrderedDict, Tuple

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
import base64
#from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
#from utils import plot_confusion_matrix,draw_roc_curve, optimal_cutoff, summerize_results
# optimizer
#from utils import NoamLR
from torch.utils.data.sampler import WeightedRandomSampler
#IPythonConsole.ipython_useSVG = True
rdDepictor.SetPreferCoordGen(True)
sns.set()
# Set the seed
torch.manual_seed(1234);
import streamlit as st
sns.set()




#MolGAT Model
#class MolGAT(torch.nn.Module):
#    def __init__(self, node_features, hidden_dim, edge_features, num_heads, num_conv_layers, num_fc_layers, dropout):
#        super(MolGAT, self).__init__()
#        torch.manual_seed(42)
#        self.num_conv_layers = num_conv_layers
#        self.num_fc_layers = num_fc_layers
#        self.num_heads = num_heads
#        self.hidden_dim = hidden_dim
#        self.dropout = dropout
#        self.conv_list = torch.nn.ModuleList()
#
#        self.conv_list.append(MolGATConv(node_features, hidden_dim, edge_features, heads=num_heads))
#        for _ in range(num_conv_layers - 1):
#            self.conv_list.append(MolGATConv(hidden_dim, hidden_dim, edge_features, heads=num_heads))
#        self.fc_list = torch.nn.ModuleList()
#        for i in range(num_fc_layers - 1):
#            if i == 0:
#                self.fc_list.append(nn.Linear(hidden_dim * 2, hidden_dim * 2))
#            else:
#                self.fc_list.append(nn.Linear(hidden_dim * 2, hidden_dim))
#
#        self.fc_out = nn.Linear(hidden_dim, 1)
#
#    def forward(self, x, edge_index, batch_index, edge_attr):
#        #x = data.x
#
#        for i, conv in enumerate(self.conv_list):
#            x = F.relu(conv(x, edge_index, edge_attr))
#            if i == (len(self.conv_list) -1):
#                x = F.dropout(x, p=self.dropout, training=self.training)
#
#        x = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)
#
#        for fc in self.fc_list:
#            x = F.relu(fc(x))
#
#        out = self.fc_out(x)
#        return out


hidden_dim=512
edge_dim = 12
num_features=99
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        torch.manual_seed(42)
        self.conv1 = MolGATConv(num_features, hidden_dim, edge_dim,heads=3)
        self.conv2 = MolGATConv(hidden_dim, hidden_dim, edge_dim,heads=3)
        self.conv3 = MolGATConv(hidden_dim, hidden_dim, edge_dim,heads=3)
        self.bn = BatchNorm(in_channels=hidden_dim)
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2,1)
    def forward(self,x, edge_index, batch_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index,edge_attr))
        x = F.relu(self.conv2(x, edge_index,edge_attr))
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



class MolGAT(torch.nn.Module):
    def __init__(self, node_features, hidden_dim, edge_features, num_heads, num_conv_layers, num_fc_layers, dropout=None):
        super(MolGAT, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.num_fc_layers = num_fc_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.conv_list = torch.nn.ModuleList()
        
        self.conv_list.append(MolGATConv(node_features, hidden_dim, edge_features, heads=num_heads))
        for _ in range(num_conv_layers - 1):
            self.conv_list.append(MolGATConv(hidden_dim, hidden_dim, edge_features, heads=num_heads))
        
        self.fc_list = torch.nn.ModuleList()
        
        for i in range(num_fc_layers - 1):
            if i == 0:
                self.fc_list.append(nn.Linear(hidden_dim * 2, hidden_dim * 2))
            elif i // 2 == 0:
                self.fc_list.append(nn.Linear(hidden_dim * 2, hidden_dim))
            else:
                self.fc_list.append(nn.Linear(hidden_dim, hidden_dim))
                
        if num_fc_layers == 1:
            self.fc_out = nn.Linear(hidden_dim * 2, 1)
        else:
            self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch_index, edge_attr):
        for i, conv in enumerate(self.conv_list):
            x = F.relu(conv(x, edge_index, edge_attr))
#             if self.dropout is not None:
#                 x = F.dropout(x, p=self.dropout, training=self.training)

        x = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        for i, fc in enumerate(self.fc_list):
            x = F.relu(fc(x))
            if self.dropout is not None:
                x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.fc_out(x)
        return out


######################
# Page Title
######################

st.write("""# Solubility(LogS)  Prediction""")

#image = Image.open('image.png')
#st.image(image, use_column_width=True)


######################
# Input molecules (Side Panel)
######################


st.sidebar.write('**Type SMILES below**')

## Read SMILES input
SMILES_input = 'CC1=C2C=CC3=CC=CC=C3C2=C(C4=CC=CC=C14)C\nC1=CC=C2C(=C1)NC(=S)S2\nC(C(=O)O)N'


SMILES = st.sidebar.text_area('then press ctrl+enter', SMILES_input, height=20)
SMILES = SMILES.split('\n')
SMILES = list(filter(None, SMILES))


st.sidebar.write("""=======**OR**=======""")
st.sidebar.write("""**Upload a csv file with a column named 'reactant_smiles'** (Max:1000)""")



uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # data
    SMILES=data["reactant_smiles"]  

    # About dataset
    st.write('Data Dimensions: ' + str(data.shape[0]) + ' rows and ' + str(data.shape[1]) + ' columns.')
    
    data_expander = st.beta_expander("Explore the Dataset", expanded=False)
    with data_expander:
		
        st.dataframe(data[0:1000])




st.sidebar.write("By Mesfin Diro")

# st.header('Input SMILES')
# SMILES[1:] # Skips the dummy first item

# Use only top 1000
if len(SMILES)>1000:
    SMILES=SMILES[0:1000]
	
mol= GenMolecules(SMILES, pre_transform=GenMolFeatures())
mol_loader = DataLoader(mol, batch_size=1,shuffle=False)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH='final_models/MolGAT_Sol.pt'

#load Redox prediction model
# redox = Net().to(device)
# redox.load_state_dict(torch.load(PATH,map_location={'cuda:0': 'cpu'}))

model = MolGAT(node_features=30,
                            hidden_dim=192,
                            edge_features=12,
                            num_heads=4,
                            num_fc_layers=4,
                            num_conv_layers=3,dropout=0.0).to(device)

#Load Solubility prediction model
model.load_state_dict(torch.load(PATH,map_location={'cuda:0': 'cpu'}))


r_std = 2.3366
r_mean =-3.5368

yp = []
model.eval()
for data in mol_loader:
    data.to(device)
    out2= model(data.x , data.edge_index, data.batch, data.edge_attr)
    out = out2*r_std + r_mean # inverse transform
    yp.append(out.tolist()[0])

# ---------------------------Result in Mol/L----------------------------
df_results = pd.DataFrame(SMILES, columns=['smiles'])
df_results["LogS"]= np.array(yp)
mask = (df_results.LogS > math.log10(200 * 1e-6))
df_results["isSoluble"] = 'No'  # Initialize the column with default values
df_results.loc[mask, "isSoluble"] = 'Yes'  # Set values to 1 where the condition is met
# ---------------Results DF ---------------------------

st.header('Solubility ')
formatted_df = df_results.style.format({'LogS': '{:.3f}', 'isSoluble': '{:s}'})
st.write(formatted_df)

download=st.button('Download Results File')
# if download:
csv = df_results.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()  # some strings
linko= f'<a href="data:file/csv;base64,{b64}" download="LogS_prediction.csv">Download csv file</a>'
st.markdown(linko, unsafe_allow_html=True)
 

#--------------------Notice------------------------
# About PART

about_part = st.expander("About MolGAT", expanded=False)
with about_part:
    st.write('''
	     #### About
	     MolGAT is a type of deep learning for molecular graph based on GNN model to predict molecular properites. This version of MolGAT model is trained to predict LogS of  organaic  molecules

	     #### Developers

	     - [Mesfin Diro](https://mesfind.github.io)


	''')


contacts = st.sidebar.expander("Contact", expanded=False)
with contacts:
    st.write('''
             #### Contact
             
             For any question you can contact us through email:
                 
             - [Mesfin Diro] (mailto: mesfin.diro@aau.edu.et)
             ''')


