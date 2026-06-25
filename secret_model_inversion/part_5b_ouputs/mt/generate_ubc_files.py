import numpy as np
import os
from discretize import TreeMesh

#%% Load Files

frequencies = np.load('../../part_1_outputs/frequencies_mt.npy')

dobs = np.load('../../part_2_outputs/dobs_mt_ubcgif.npy')
unc = np.load('../../part_2_outputs/unc_mt_ubcgif.npy')

locations = np.load('../../part_3_outputs/locations_mt_shifted.npy')

mesh = TreeMesh.read_UBC('../../part_3_outputs/octree_mesh_ne.txt')
active_cells = np.load('../../part_3_outputs/active_cells_ne.npy')
active_cells = np.array(active_cells, dtype=int)
ubc_order = mesh._ubc_order
active_cells = active_cells[ubc_order]
np.savetxt('./active_cells_topo.txt', active_cells, fmt='%i')

#%% EXTRACT DATA BEING INVERTED

n_freq = len(frequencies)
n_locations = len(locations)
n_comp = 8

dobs = dobs.reshape((n_freq, n_comp, n_locations))
unc = unc.reshape((n_freq, n_comp, n_locations))

xmin, xmax = 3475., 1e8
ymin, ymax = 2475., 1e8
inds = (locations[:, 0] >= xmin) & (locations[:, 0] <= xmax) & (locations[:, 1] >= ymin) & (locations[:, 1] <= ymax)
locations = locations[inds, :]
dobs = dobs[:, :, inds]
unc = unc[:, :, inds]
n_loc = len(locations)

# 10, 25.1, 63.1, 158.5, 398.1, 1000
inds = [0, 4, 8, 12, 16, 20]
frequencies = frequencies[inds]
dobs = dobs[inds, :, :]
unc = unc[inds, :, :]
n_freq = len(frequencies)

#%%

fname = './survey.dat'
fid = open(fname, 'w')
fid.write(f"N_TRX {n_freq}\n")
fid.write("IGNORE -9999\n")
fid.close()

for ii, f in enumerate(frequencies):
    
    fid = open(fname, 'a')
    fid.write("\n\nDATATYPE MTZ\n")
    fid.write(f"FREQUENCY {f}\n")
    fid.write(f"N_RECV {n_loc}\n")
    
    Aout = np.zeros((n_loc, 3 + 2*n_comp))
    Aout[:, :3] = locations
    Aout[:, 3::2] = dobs[ii, :, :].T
    Aout[:, 4::2] = unc[ii, :, :].T
    
    np.savetxt(fid, Aout, delimiter=' ', fmt='%.6e')
    
    fid.close()
    
    
    







