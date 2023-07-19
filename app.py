# -*- coding: utf-8 -*-
"""
Created on Fri July 09 14:54:37 2023

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
from torch.utils.data import DataLoader, Dataset, Sampler
from torch_geometric.data import DataLoader
from rdkit import Chem
from torch.utils.data import random_split
from typing import Dict, Iterator, List, Optional, Union, OrderedDict, Tuple
from functools import reduce
from sklearn.metrics import mean_squared_error
from io import BytesIO
from scipy import stats
from random import Random
from scipy import stats
import math
import base64
torch.manual_seed(1234);
import streamlit as st



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
st.sidebar.write("""**Upload a csv file with a column named 'smiles'** (Max:1000)""")



uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # data
    SMILES=data["smiles"]  

    # About dataset
    st.write('Data Dimensions: ' + str(data.shape[0]) + ' rows and ' + str(data.shape[1]) + ' columns.')
    
    data_expander = st.beta_expander("Explore the Dataset", expanded=False)
    with data_expander:
		
        st.dataframe(data[0:1000])


# Use only top 1000
if len(SMILES)>1000:
    SMILES=SMILES[0:1000]


mol= GenMolecules(SMILES, pre_transform=GenMolFeatures())
mol_loader = DataLoader(mol, batch_size=1,shuffle=False)


def predict_logs(SMILES, mol_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PATH = 'final_models/MolGAT_Sol.pt'
    model = MolGAT(node_features=30,
                   hidden_dim=192,
                   edge_features=12,
                   num_heads=4,
                   num_fc_layers=4,
                   num_conv_layers=3,
                   dropout=0.0).to(device)

    # Load Solubility prediction model
    model.load_state_dict(torch.load(PATH, map_location={'cuda:0': 'cpu'}))

    r_std = 2.3366
    r_mean = -3.5368

    yp = []
    model.eval()
    for data in mol_loader:
        data.to(device)
        out2 = model(data.x, data.edge_index, data.batch, data.edge_attr)
        out = out2 * r_std + r_mean  # Inverse transform
        yp.append(out.tolist()[0])

    # ---------------------------Result in Mol/L----------------------------
    df_results = pd.DataFrame(SMILES, columns=['smiles'])
    df_results["LogS"] = np.array(yp)
    mask = (df_results.LogS > math.log10(200 * 1e-6))
    df_results["isSoluble"] = 'No'  # Initialize the column with default values
    df_results.loc[mask, "isSoluble"] = 'Yes'  # Set values to 1 where the condition is met

    return df_results

logs = predict_logs(SMILES, mol_loader)
# Display the results in the Streamlit app
st.header('Solubility')
formatted_logs = logs.style.format({'LogS': '{:.3f}', 'isSoluble': '{:s}'})
st.write(formatted_logs)
download = st.button('Download Results File')
if download:
    csv = logs1.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    linko = f'<a href="data:file/csv;base64,{b64}" download="LogS_prediction.csv">Download csv file</a>'
    st.markdown(linko, unsafe_allow_html=True)



#--------------------Notice------------------------
# About PART

about_part = st.expander("About MolGAT", expanded=False)
with about_part:
    st.markdown('''
    #### About
    The MolGAT model is a type of deep learning architecture designed for predicting molecular properties from molecular graphs. 
    It leverages an attention mechanism to selectively focus on crucial parts of the molecular graph with n-dimensional node and edge attributes, 
    enabling accurate property predictions. This  graph-based representation help the model to learn and capture the complex relationships and patterns within molecules,
    enabling it to make predictions about various molecular properties. Trained on a large dataset of molecular graphs, the MolGAT model has demonstrated success in tasks
    such as solubility and redox potential prediction, making it a powerful tool in materials science and drug discovery.
    #### Citation

    If you use this code in your own work, please cite our paper and the respective papers of the methods used:

    - [High-Throughput Screening of Promising Redox-Active Molecules with MolGAT](https://pubs.acs.org/doi/full/10.1021/acsomega.3c01295)

    ```
    @article{doi:10.1021/acsomega.3c01295,
    author = {Chaka, Mesfin Diro and Geffe, Chernet Amente and Rodriguez, Alex and Seriani, Nicola and Wu, Qin and Mekonnen, Yedilfana Setarge},
    title = {High-Throughput Screening of Promising Redox-Active Molecules with MolGAT},
    journal = {ACS Omega},
    volume = {8},
    number = {27},
    pages = {24268-24278},
    year = {2023},
    doi = {10.1021/acsomega.3c01295},
    }
    ```
    ''')


contacts = st.sidebar.expander("Contact Authors", expanded=False)
with contacts:
    st.write('''
             #### Contact
             
             For any question you can contact us through email:
                 
             - [Mesfin Diro] (mailto: mesfin.diro@aau.edu.et)
             - [Chernet Amente](mailto: chernet.amente@aau.edu.et)
             - Seriani Nicola(mailto: seriani@ictp.it)
             - Qin Wu (mailto: qinwu@bnl.gov)
             - Yedilfana Setarge(yedilfana.setarge@aau.edu.et)
             ''')


