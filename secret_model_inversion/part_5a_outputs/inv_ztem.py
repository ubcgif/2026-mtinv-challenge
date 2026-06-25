#!/usr/bin/env python
# coding: utf-8

# # Inversion

# In[ ]:


import numpy as np
from discretize import TreeMesh, TensorMesh
from discretize.utils import mkvc, ndgrid, sdiag
from simpeg.utils import validate_type, get_default_solver
from simpeg.electromagnetics import natural_source as nsem
import os
from datetime import datetime

from simpeg import (
    maps,
    data,
    inverse_problem,
    data_misfit,
    regularization,
    optimization,
    directives,
    inversion,
    utils
)

work_dir = os.getcwd()
dsep = os.path.sep

date_str = datetime.now()
date_str = date_str.strftime("%d_%m_%Y_%H_%M")

########################################################################
# LOAD STUFF
########################################################################

frequencies = np.load('../part_1_outputs/frequencies_ztem.npy')

dobs = np.load('../part_2_outputs/dobs_ztem_simpeg.npy')
unc = np.load('../part_2_outputs/unc_ztem_simpeg.npy')

locations = np.load('../part_3_outputs/locations_ztem_shifted.npy')
location_base = np.load('../part_3_outputs/location_base_shifted.npy')
mesh = TreeMesh.read_UBC('../part_3_outputs/octree_mesh_ne.txt')
active_cells = np.load('../part_3_outputs/active_cells_ne.npy')
sigma_1d = np.load('../part_3_outputs/sigma_1d.npy')

out_dir = './ztem'

########################################################################
# EXTRACT DATA BEING INVERTED
########################################################################

n_freq = len(frequencies)
n_locations = len(locations)
n_comp = 4

dobs = dobs.reshape((n_freq, n_comp, n_locations))
unc = unc.reshape((n_freq, n_comp, n_locations))

xmin, xmax = 3475., 1e8
ymin, ymax = 2475., 1e8
inds = (locations[:, 0] >= xmin) & (locations[:, 0] <= xmax) & (locations[:, 1] >= ymin) & (locations[:, 1] <= ymax)
locations = locations[inds, :]
dobs = dobs[:, :, inds]
unc = unc[:, :, inds]

inds = (frequencies >= 10.) & (frequencies <= 1000.)
frequencies = frequencies[inds]
dobs = dobs[inds, :, :]
unc = unc[inds, :, :]

dobs = dobs.reshape(-1)
unc = unc.reshape(-1)


########################################################################
# DEFINE SURVEY
########################################################################

source_list = []

for f in frequencies:

    temp = np.repeat(base_station.reshape((1, 3)), n_locations, axis=0)
        
    receiver_list = []

    for rx_type in ['zx', 'zy']:

        receiver_list.append(
            nsem.receivers.Tipper(
                locations_h=locations, locations_base=temp, component="real", orientation=rx_type
            )
        )

        receiver_list.append(
            nsem.receivers.Tipper(
                locations_h=locations, locations_base=temp, component="imag", orientation=rx_type
            )
        )
    
    source_list.append(nsem.sources.FictitiousSource(receiver_list, f))

survey = nsem.survey.Survey(source_list)

########################################################################
# LOAD DATA AND DEFINE DATA OBJECTS
########################################################################

data_obj = data.Data(survey, dobs=dobs, standard_deviation=unc)

########################################################################
# MAPPING
########################################################################

# Find cells that are active in the forward modeling (cells below surface)
nP = int(active_cells.sum())

# Define mapping from model to active cells
air_conductivity = 1e-8
active_map = maps.InjectActiveCells(mesh, active_cells, air_conductivity)
conductivity_map = active_map * maps.ExpMap()

########################################################################
# SURFACE WEIGHTS
########################################################################

# dz = np.min(mesh.h[2])
# wt = np.ones(mesh.nC)
# ccz = mesh.cell_centers[:, -1]

# min_topo = np.min(ccz[~active_cells])

# wt[(ccz > min_topo-dz)] = 2.5
# wt[(ccz < min_topo-dz) & (ccz > min_topo-2*dz)] = 10.
# wt = wt[active_cells]

########################################################################
# INVERSION
########################################################################

out_dir_full = out_dir.copy()
if not os.path.exists(out_dir_full):
    os.mkdir(out_dir_full)

print('OUT_DIR: {}'.format(out_dir_full))

# Starting model is best hs
starting_model = np.log(0.02) * np.ones(nP)

# simulation
sim = nsem.simulation.Simulation3DElectricFieldFictitious(
    mesh,
    survey=survey,
    sigma_background=sigma_1d,
    sigmaMap=conductivity_map,
    solver=get_default_solver()
)

dmis = data_misfit.L2DataMisfit(data=data_obj, simulation=sim)
reg = regularization.WeightedLeastSquares(
    mesh,
    active_cells=active_cells,
    mapping=maps.IdentityMap(nP=nP),
)

# reg.objfcts[3].set_weights(surface=wt)
# reg.objfcts[5].set_weights(surface=wt)

reg.alpha_s = 1e-8
reg.alpha_x = 1.
reg.alpha_y = 1.
reg.alpha_z = 1.

# reg.mrefInSmooth = True
opt = optimization.ProjectedGNCG(
    maxIter=39, maxIterLS=20, cg_maxiter=250, cg_atol=0.01, eps=1e-6, tolF=0.001, tolX=0.001, tolG=0.01, nbfgs=20
)
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=10)


# In[14]:

# starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e2)
beta_schedule = directives.BetaSchedule(coolingFactor=2.5, coolingRate=3)
save_convergence = directives.SaveOutputEveryIteration(name=dsep.join([out_dir_full, 'convergence']))
save_dictionary = directives.SaveOutputDictEveryIteration(on_disk=True, directory=out_dir_full, name="iter")
target_misfit = directives.TargetMisfit(chifact=1.05)

print('DIRECTORY {}'.format(save_dictionary.directory))

directives_list = [
    beta_schedule,
    save_convergence,
    save_dictionary,
    target_misfit,
]

# In[ ]:
inv = inversion.BaseInversion(inv_prob, directiveList=directives_list)
recovered_model = inv.run(starting_model)


