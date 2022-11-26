#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 11:38:51 2021

@author: Mesfin Diro
"""
from typing import Callable, Optional
import os.path as osp
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

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}

 

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
            Chem.rdchem.BondStereo.STEREOTRANS,
        ]


class RedDB(InMemoryDataset):
    r"""
  RedDB is a computational database that covers a focused chemical space of two
  classes of organic molecules (quinones and aza-aromatics) that have found to be 
  highly promising for aqueous redox flow batteries. RedDB's data is generated using
  simulation tools that apply cheminformatics, machine learning, molecular mechanics,
  and quantum chemistry methods. RedDB contains structural information and several
  physicochemical properties of molecules that are candidates for their function 
  as electroactive materials in aqueous redox flow batteries. (2021-04-10)
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
  
    url = 'https://dataverse.harvard.edu/api/access/datafile/4461991'
  
 
 
 
    def __init__(self, root_dir: str, name, smi_idx, target_idx,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.root_dir = root_dir
        self.name = name
        self.smi_idx = smi_idx
        self.target_idx = target_idx
        #skip calling data
        super(RedDB, self).__init__(root_dir, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])


    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    def atomref(self, target) -> Optional[torch.Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor(atomic_num)] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None
 
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
    def processed_file_names(self) -> str:
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
            ys = line[self.target_idx]
            ys = ys if isinstance(ys, list) else [ys]
 
            ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
            y = torch.tensor(ys, dtype=torch.float).view(-1, 1)
            
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
                atomic_num_scaled =  float((atom.GetAtomicNum() - 1)/91) # x - min/(max-min)
                valance = [0.] * 8    # the maximum valence electron is 8
                valance[atom.GetDegree()] = 1.
                formal_charge = atom.GetFormalCharge() #  [-1, -2, 1, 2, 0]
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
                             [aromaticity] + hydrogens+  [chirality] +
                             chirality_type + [atomic_mass] +[vdw_radius]+[covalent_radius], dtype=torch.float)
                xs.append(x1)
                x = torch.stack(xs, dim=0)

            z = torch.tensor(atomic_number, dtype=torch.long)
            
            row, col,edge_attrs = [], [], []
            xs = []
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
            #sort indices
            #perm = (edge_index[0] * N + edge_index[1]).argsort()
            #edge_index = edge_index[:, perm]
            #edge_type = edge_type[perm]
            #edge_attr = edge_attr[perm]
            
            # Sort indices.
            if edge_index.numel() > 0:
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]
    

            data = Data(x=x, z=z, pos=pos, edge_index=edge_index, edge_attr=edge_attr, y=y,
                        smiles=smiles)
 
            if self.pre_filter is not None:
                data = [d for d in data if self.pre_filter(d)]
 
            if self.pre_transform is not None:
                data = self.pre_transform(data)
 
            data_list.append(data)
 
        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
