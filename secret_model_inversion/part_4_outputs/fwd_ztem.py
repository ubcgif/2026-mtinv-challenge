import numpy as np
from discretize import TreeMesh, TensorMesh
from discretize.utils import mkvc, ndgrid, sdiag
from simpeg.utils import validate_type, get_default_solver
from simpeg.electromagnetics import natural_source as nsem
from simpeg import maps
import os

work_dir = os.getcwd()
dsep = os.path.sep


########################################################################
# SURVEY LOAD FILES
########################################################################

frequencies = np.load('../part_1_outputs/frequencies_ztem.npy')

locations = np.load('../part_3_outputs/locations_ztem_shifted.npy')

location_base = np.load('../part_3_outputs/location_base_shifted.npy')

mesh = TreeMesh.read_UBC('../part_3_outputs/octree_mesh_ne.txt')

active_cells = np.load('../part_3_outputs/active_cells_ne.npy')

sigma_1d = np.load('../part_3_outputs/sigma_1d.npy')


xmin, xmax = 3475., 1e8
ymin, ymax = 2475., 1e8

inds = (locations[:, 0] >= xmin) & (locations[:, 0] <= xmax) & (locations[:, 1] >= ymin) & (locations[:, 1] <= ymax)
locations = locations[inds, :]

inds = (frequencies >= 10.) & (frequencies <= 1000.)
frequencies = frequencies[inds]

n_freq = len(frequencies)
n_locations = len(locations)
n_comp = 4


########################################################################
# DEFINE SURVEY
########################################################################

source_list = []

for f in frequencies:

    temp = np.repeat(location_base.reshape((1, 3)), n_locations, axis=0)
        
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

# MAPS AND MODELS

# Find cells that are active in the forward modeling (cells below surface)
nP = int(active_cells.sum())

# Define mapping from model to active cells
air_conductivity = 1e-8
active_map = maps.InjectActiveCells(mesh, active_cells, air_conductivity)

conductivity_model = 0.02 * np.ones(nP)

# SIMULTION


# simulation
sim = nsem.simulation.Simulation3DElectricFieldFictitious(
    mesh,
    survey=survey,
    sigma_background=sigma_1d,
    sigmaMap=active_map,
    solver=get_default_solver()
)


dpred = sim.dpred(conductivity_model)


dpred = np.reshape(dpred, (n_freq, n_comp, n_locations))


np.save('./dpred_ztem', dpred)