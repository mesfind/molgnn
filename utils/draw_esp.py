import ipymol
from psikit import Psikit
pk = Psikit()
v = ipymol.viewer
v.start() # pymol will launch

pk.view_on_pymol(target='ESP')
pk.view_on_pymol(target='DUAL', maprange=0.001)
pk.view_on_pymol('FRONTIER')