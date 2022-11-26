from rdkit import Chem
from rdkit.Chem import rdmolops
import igraph
import numpy as np
# mol2graph function can convert molecule to graph. And add some node and edge attribute.
def mol2graph( mol ):
    admatrix = rdmolops.GetAdjacencyMatrix( mol )
    bondidxs = [ ( b.GetBeginAtomIdx(),b.GetEndAtomIdx() ) for b in mol.GetBonds() ]
    adlist = np.ndarray.tolist( admatrix )
    graph = igraph.Graph()
    g = graph.Adjacency( adlist ).as_undirected()
    for idx in g.vs.indices:
        g.vs[ idx ][ "AtomicNum" ] = mol.GetAtomWithIdx( idx ).GetAtomicNum()
        g.vs[ idx ][ "AtomicSymbole" ] = mol.GetAtomWithIdx( idx ).GetSymbol()
    for bd in bondidxs:
        btype = mol.GetBondBetweenAtoms( bd[0], bd[1] ).GetBondTypeAsDouble()
        g.es[ g.get_eid(bd[0], bd[1]) ][ "BondType" ] = btype
        print( bd, mol.GetBondBetweenAtoms( bd[0], bd[1] ).GetBondTypeAsDouble() )
    return g
