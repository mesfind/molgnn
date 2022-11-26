#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 11:38:51 2021

@author: Mesfin Diro
"""
import os.path as osp
from typing import Callable, List, Optional
import torch
from torch_geometric.data import InMemoryDataset, Data
import re
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
RDLogger.DisableLog('rdApp.*')


HAR2EV = 27.211386246    # Hartree to EV
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])



symbols = ['H','C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb',
           'Sb','Sn', 'Sr','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt',
           'Hg','Pb','Hf','Te','Tb','Tm','Ce','Eu','Lu','Er','Gd','Nd','Nb','Dy','Os','Sc','Bi','Ta','Re','Mo','Ba','Ga','Ru','Ir','Rh','W','U']

atomic_num = list(range(1, len(symbols)))
types = dict(zip(symbols,atomic_num))
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3 }
hybridizations = [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED
        ]
stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS
        ]

class GenMolGraph(InMemoryDataset):
    r"""
    A molecular graph feature generator for prediction and virtual screening 
    
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"RedDB"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(self, root_dir: Optional[Callable] = None, name:Optional[Callable] = None, 
                 smiles_list:Optional[Callable] = None,smi_idx:Optional[Callable] = None, 
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.root_dir = root_dir
        self.name = name
        self.smi_idx = smi_idx
        self.smiles_list = smiles_list
        #skip calling data
        super(GenMolGraph, self).__init__(root_dir, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_dir(self):
        return osp.join(self.root_dir, 'raw')
 
    @property
    def processed_dir(self):

        return osp.join(self.root_dir,'processed')
 
    @property
    def raw_file_names(self):
        return  osp.join(self.raw_dir, self.name)
 
    @property
    def processed_file_names(self) -> List[str]:
        return osp.splitext(self.name)[0] + '.pt'

    def process(self):
        with open(self.raw_file_names, 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        for line in tqdm(dataset):
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            line = line.split(',')
            smiles = line[self.smi_idx]
            mol = Chem.MolFromSmiles(smiles)

            mol = Chem.AddHs(mol)
            atoms = mol.GetAtoms()
            AllChem.EmbedMolecule(mol, useExpTorsionAnglePrefs=True,useBasicKnowledge=True)
            AllChem.UFFOptimizeMolecule(mol)
            atoms = mol.GetAtoms()
            #string = "\n"
            for _, atom in enumerate(atoms):
                pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                #string += "{} {} {} {}\n".format(atom.GetSymbol(), pos.x, pos.y, pos.z)
                #string += "units angstrom\n"
            pos = torch.tensor(pos, dtype=torch.float)
            atomic_number = []
            xs = []
            for atom in mol.GetAtoms():
                symbol = [0.] * len(symbols)
                symbol[symbols.index(atom.GetSymbol())] = 1.
                atomic_number.append(atom.GetAtomicNum())
                atomic_num_scaled = float((atom.GetAtomicNum() - 1)/91) # H: 1, U: 92 
                valance = [0.] * 8
                valance[atom.GetDegree()] = 1.
                formal_charge = atom.GetFormalCharge()
                radical_electrons = atom.GetNumRadicalElectrons()
                hybridization = [0.] * len(hybridizations)
                hybridization[hybridizations.index(
                atom.GetHybridization())] = 1.
                aromaticity = 1. if atom.GetIsAromatic() else 0.
                hydrogens = [0.] * 5
                hydrogens[atom.GetTotalNumHs()] = 1.
                chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
                chirality_type = [0.] * 2
                if atom.HasProp('_CIPCode'):
                    chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.
                #scaled features
                atomic_mass = float((atom.GetMass() - 1.008)/237.021) # H: 1.008  U: 238.029
                vdw_radius = float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.2)/1.35)   #  H(min):1.2, Sr(max): 2.55   
                covalent_radius = float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.23)/1.71) #  H:0.23, Yb(max): 1.94 minmax
            
                x1 = torch.tensor([atomic_num_scaled]+symbol+ valance + [formal_charge] +
                             [radical_electrons] + hybridization +
                             [aromaticity] + hydrogens + [chirality] +
                             chirality_type + [atomic_mass]+[vdw_radius]+[covalent_radius]  ,dtype=torch.float)
                xs.append(x1)
                x = torch.stack(xs, dim=0)

            z = torch.tensor(atomic_number, dtype=torch.long)
            
            row, col,  edge_attrs = [], [], []

            for bond in mol.GetBonds():
                start = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
 
                row +=[start, end]
                col += [end, start]
                
               
                bond_type = bond.GetBondType()
                single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
                double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
                triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
                aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
                conjugation = 1. if bond.GetIsConjugated() else 0.
                ring = 1. if bond.IsInRing() else 0.
                stereo = [0.] * 6
                stereo[stereos.index(bond.GetStereo())] = 1.

                edge_attr = torch.tensor([single, double, triple, aromatic, conjugation, ring] + stereo)

                edge_attrs += [edge_attr, edge_attr]
            edge_attr = torch.stack(edge_attrs, dim=0)
            edge_index = torch.tensor([row, col], dtype=torch.long)
            
          
            # Sort indices.
            if edge_index.numel() > 0:
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]
    
            data = Data(x=x, z=z,pos=pos, edge_index=edge_index, edge_attr=edge_attr, y=None,
                        smiles=smiles)
 
            if self.pre_filter is not None:
                data = [d for d in data if self.pre_filter(d)]
 
            if self.pre_transform is not None:
                data = self.pre_transform(data)
 
            data_list.append(data)
        #return data_list
 
        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)





class GenMolFeatures(object):
    r"""
    A molecular graph feature generator for prediction and virtual screening 
    
    Args:
        smiles (string): list of smile strings
 
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    
    usage: 
        from genmolfeatures import GenMolecules, GenMolFeatures
        from torch_geometric.loader import DataLoader
        SMILES="O=C1CC(=O)C=C1C(=O)O\nClC(Cl)C(Cl)Cl\nOc1ccc(O)cc1\nOc1ccccc1O"
        SMILES = SMILES.split('\n')
        SMILES = list(filter(None, SMILES))
        dataset = GenMolecules(SMILES,pre_transform=GenMolFeatures())
        
        ds_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        yp = []
        yr = []
        model.eval()
        for g in ds_loader:
            g.to(device)
            out= model(g.x , g.edge_index, g.batch,g.edge_attr)
            yp.append(out.tolist()[0])
    """
    def __init__(self):
        self.symbols = ['H','C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb',
           'Sb','Sn', 'Sr','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt',
           'Hg','Pb','Hf','Te','Tb','Tm','Ce','Eu','Nd','Er','Gd','Nb','Lu','Dy','Os','Sc','Bi','Ta','Re','Mo','Ba','Ga','Ru','Ir','Rh','W','U']

        self.hybridizations = [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED
        ]

        self.stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS
        ]

    def __call__(self, data):

        mol = Chem.MolFromSmiles(data.smiles)
        mol = Chem.AddHs(mol)
        xs = []
        atomic_number = []
        for atom in mol.GetAtoms():
            symbol = [0.] * len(self.symbols)
            symbol[self.symbols.index(atom.GetSymbol())] = 1.
            atomic_number.append(atom.GetAtomicNum())
            atomic_num_scaled = float((atom.GetAtomicNum() - 1)/91)
            valance = [0.] * 8
            valance[atom.GetDegree()] = 1.
            formal_charge = atom.GetFormalCharge()
            radical_electrons = atom.GetNumRadicalElectrons()
            hybridization = [0.] * len(self.hybridizations)
            hybridization[self.hybridizations.index(
                atom.GetHybridization())] = 1.
            aromaticity = 1. if atom.GetIsAromatic() else 0.
            hydrogens = [0.] * 5
            hydrogens[atom.GetTotalNumHs()] = 1.
            chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
            chirality_type = [0.] * 2
            if atom.HasProp('_CIPCode'):
                chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.
            #scaled features
            atomic_mass = float((atom.GetMass() - 1.008)/237.021) # H: 1.008  U: 238.029
            vdw_radius = float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.2)/1.35)   #  H(min):1.2, Sr(max): 2.55   
            covalent_radius = float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.23)/1.71) #  H:0.23, Yb(max): 1.94 minmax
           

            x = torch.tensor([atomic_num_scaled]+symbol+ valance + [formal_charge] +
                             [radical_electrons] + hybridization +
                             [aromaticity] + hydrogens + [chirality] +
                             chirality_type+[atomic_mass] + [vdw_radius]+ [covalent_radius],dtype=torch.float)
            xs.append(x)
        
        data.x = torch.stack(xs, dim=0)
        
        z = torch.tensor(atomic_number, dtype=torch.long)
        data.z = z
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]

            bond_type = bond.GetBondType()
            single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
            double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
            triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
            aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
            conjugation = 1. if bond.GetIsConjugated() else 0.
            ring = 1. if bond.IsInRing() else 0.
            stereo = [0.] * 6
            stereo[self.stereos.index(bond.GetStereo())] = 1.

            edge_attr = torch.tensor(
                [single, double, triple, aromatic, conjugation, ring] + stereo)

            edge_attrs += [edge_attr, edge_attr]

        if len(edge_attrs) == 0:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            data.edge_index = torch.tensor(edge_indices).t().contiguous()
            data.edge_attr = torch.stack(edge_attrs, dim=0)

        return data



def GenMolecules(smiles_list, pre_transform=None, pre_filter=None):
    from rdkit import Chem
    data_list = []
    for smi in tqdm(smiles_list):
        smi = re.sub(r'\".*\"', '', smi)  # Replace ".*" strings.
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        smiles = smi
        mol = Chem.AddHs(mol)
        atomic_number = []
        xs = []
        for atom in mol.GetAtoms():
            symbol = [0.] * len(symbols)
            symbol[symbols.index(atom.GetSymbol())] = 1.
            atomic_number.append(atom.GetAtomicNum())
            atomic_num_scaled =  float((atom.GetAtomicNum() - 1)/91) # H : 1, U: 92
            valance = [0.] * 8
            valance[atom.GetDegree()] = 1.
            formal_charge = atom.GetFormalCharge()   # [-3, -2, -1, 0, 1, 2, 3]
            radical_electrons = atom.GetNumRadicalElectrons()
            hybridization = [0.] * len(hybridizations)
            hybridization[hybridizations.index(
            atom.GetHybridization())] = 1.
            aromaticity = 1. if atom.GetIsAromatic() else 0.
            hydrogens = [0.] * 5
            hydrogens[atom.GetTotalNumHs()] = 1.
            chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
            chirality_type = [0.] * 2
            if atom.HasProp('_CIPCode'):
                chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.
            #scaled features
            atomic_mass = float((atom.GetMass() - 1.008)/237.021) # H: 1.008  U: 238.029
            vdw_radius = float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.2)/1.35)   #  H(min):1.2, Sr(max): 2.55   
            covalent_radius = float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.23)/1.71) #  H:0.23, Yb(max): 1.94 minmax
             
            
            x1 = torch.tensor([atomic_num_scaled]+symbol+ valance + [formal_charge] +
                         [radical_electrons] + hybridization +
                         [aromaticity] + hydrogens + [chirality] +
                         chirality_type + [atomic_mass] + [vdw_radius]+[covalent_radius],dtype=torch.float)
            xs.append(x1)
            x = torch.stack(xs, dim=0)

        z = torch.tensor(atomic_number, dtype=torch.long)
        
        row, col,  edge_attrs = [], [], []

        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()

            row +=[start, end]
            col += [end, start]
           
            bond_type = bond.GetBondType()
            single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
            double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
            triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
            aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
            conjugation = 1. if bond.GetIsConjugated() else 0.
            ring = 1. if bond.IsInRing() else 0.
            stereo = [0.] * 4
            stereo[stereos.index(bond.GetStereo())] = 1.

            edge_attr = torch.tensor([single, double, triple, aromatic, conjugation, ring] + stereo)

            edge_attrs += [edge_attr, edge_attr]
        edge_attr = torch.stack(edge_attrs, dim=0)
        edge_index = torch.tensor([row, col], dtype=torch.long)
        
        
        
        # Sort indices.
        if edge_index.numel() > 0:
            perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
            edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]


        data = Data(x=x, z=z, edge_index=edge_index, edge_attr=edge_attr, y=None,
                    smiles=smiles)
        if pre_filter is not None and not pre_filter(data):
            continue
        if pre_transform is not None:
            data = pre_transform(data)
        data_list.append(data)
    return data_list




