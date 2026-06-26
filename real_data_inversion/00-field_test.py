from simpeg import maps, data, optimization, maps, regularization, inverse_problem, directives, inversion, data_misfit
from simpeg.utils import (
    model_builder, get_default_solver, shift_to_discrete_topography, ndgrid, plot2Ddata
)
from simpeg.objective_function import BaseObjectiveFunction
import discretize
import numpy as np
import matplotlib.pyplot as plt
from simpeg.electromagnetics import natural_source as nsem
import discretize
from discretize.utils import mkvc, refine_tree_xyz
import numpy as np
from pymatsolver import Pardiso as Solver
from simpeg.electromagnetics.static import resistivity as dc, utils as dcutils
import scipy.sparse as sp
import utm
import mtpy as mt
from mt_metadata import TF_XML
from pathlib import Path
import json
from simpeg.meta import MultiprocessingMetaSimulation
# Python Version
import sys
print(sys.version)


# append_dir = "/home/juanito"
append_dir = "/Users/johnkuttai"


# ---------------------------------------------------------------------------

# helper functions

#


# ---------------------------------------------------------------------------

# load the MT data and do some preprocessing — or restore from JSON cache

#

_MT_CACHE_FILE = Path("mt_survey_data.json")

if _MT_CACHE_FILE.exists():
    print(f'[INFO] Loading MT data from cache: {_MT_CACHE_FILE}')
    with open(_MT_CACHE_FILE) as _f:
        _cache = json.load(_f)
    rx_locs          = [list(r) for r in _cache["rx_locs"]]
    rx_locs_tipper   = [list(r) for r in _cache["rx_locs_tipper"]]
    elevation        = _cache["elevation"]
    elevation_tipper = _cache["elevation_tipper"]
    data_col_yx      = {float(k): v for k, v in _cache["data_col_yx"].items()}
    data_col_xy      = {float(k): v for k, v in _cache["data_col_xy"].items()}

else:
    print('[INFO] Preprocessing MT data...')

    directory_path = Path(f"{append_dir}/Dropbox/JohnLindsey/2026-mt3dinv/3d")
    mtc = mt.MTCollection()
    mtc.open_collection(Path().cwd().joinpath("test_collection3dvt4.h5"))

    for file_path in directory_path.iterdir():
        if file_path.is_file():
            if ".DS_Store" == file_path.name:
                continue
            mt_object = mt.MT()
            mt_object.read(file_path)
            mt_object.station = file_path.stem
            mt_object.station_metadata.id = file_path.stem
            mt_object.survey_metadata.id = "grid"
            # mt_object.tf_id = file_path.stem
            mtc.add_tf(mt_object, tf_id_extra=file_path.stem)

    mtc.working_dataframe = mtc.master_dataframe.loc[mtc.master_dataframe.survey == "grid"]
    mtc.working_dataframe

    # station_plot = mtc.plot_stations(pad=0.0005)

    mtd = mtc.to_mt_data()

    # # Station map
    # station_plot = mtc.plot_stations(pad=0.0005)
    # station_plot.fig.savefig("station_map.png", dpi=150, bbox_inches="tight")
    # plt.close(station_plot.fig)

    mtc.close_collection()

    # collect the data into a nice list and convert the data and locations
    _impUnitEDI2SI = 4 * np.pi * 1e-4

    rx_locs = []
    rx_locs_tipper = []
    elevation = []
    elevation_tipper = []

    for key in list(mtd.keys()):
        rx_locs += [utm.from_latlon(mtd[key].latitude, mtd[key].longitude)[:2]]
        elevation += [mtd[key].elevation]

        if mtd[key].has_tipper():
            rx_locs_tipper += [utm.from_latlon(mtd[key].latitude, mtd[key].longitude)[:2]]
            elevation_tipper += [mtd[key].elevation]

    # # since this is a 2D inversion rotate the coordinates of the location to inline
    # rotated_points = rotate_points(rx_locs, rx_locs[-4], -40)
    # rotated_points_tipper = rotate_points(rx_locs_tipper, rx_locs[-4], -40)

    # rx_locs2d = np.vstack([rotated_points[:, 0], elevation]).T
    # rx_locs2d_tipper = np.vstack([rotated_points_tipper[:, 0], elevation_tipper]).T

    mtd.compute_model_errors()

    if False:
        # Per-station apparent resistivity + phase PDF (one page per station)
        from matplotlib.backends.backend_pdf import PdfPages as _PdfPages
        with _PdfPages("mtpy_station_responses.pdf") as _pdf:
            for _key in mtd.keys():
                _p = mtd[_key].plot_mt_response(show=False)
                _pdf.savefig(_p.fig)
                plt.close(_p.fig)
        print('[INFO] Saved mtpy station response plots to mtpy_station_responses.pdf')

        # Phase tensor spatial maps — one page per frequency
        _freqs_sorted = sorted({
            freq
            for _key in mtd.keys()
            for freq in mtd[_key].Z.frequency
        })
        with _PdfPages("phase_tensor_maps.pdf") as _pdf:
            for _freq in _freqs_sorted:
                _period = 1.0 / _freq
                try:
                    _ptm = mtd.plot_phase_tensor_map(
                        plot_period=_period,
                        ellipse_size=0.001,
                        show=False,
                    )
                    _ptm.fig.suptitle(
                        f"Phase Tensor  |  f = {_freq:.4g} Hz  T = {_period:.4g} s",
                        fontsize=11,
                    )
                    _pdf.savefig(_ptm.fig)
                    plt.close(_ptm.fig)
                except Exception as _e:
                    print(f'[WARN] PT map skipped f={_freq:.4g} Hz: {_e}')
        print(f'[INFO] Saved {len(_freqs_sorted)} phase tensor maps to phase_tensor_maps.pdf')

        # Apparent resistivity + phase spatial maps — one page per frequency
        with _PdfPages("resphase_maps.pdf") as _pdf:
            for _freq in _freqs_sorted:
                _period = 1.0 / _freq
                try:
                    _rpm = mtd.plot_resistivity_phase_maps(plot_period=_period, show=False)
                    _pdf.savefig(_rpm.fig)
                    plt.close(_rpm.fig)
                except Exception as _e:
                    print(f'[WARN] Res/phase map skipped f={_freq:.4g} Hz: {_e}')
        print(f'[INFO] Saved {len(_freqs_sorted)} res/phase maps to resphase_maps.pdf')

    # collect the data into something simpeg can use

    data_col_yx = {}
    data_col_xy = {}

    for key in mtd.keys():

        for ii, freq in enumerate(mtd[key].Z.frequency):

            data_col_xy[freq] = {

                'real': [],
                'imag': [],
                'stn': [],

            }
            data_col_yx[freq] = {

                'real': [],
                'imag': [],
                'stn': [],

            }

    for key in mtd.keys():

        for ii, freq in enumerate(mtd[key].Z.frequency):
            # print(f"{key} freq: {freq}")
            # zrot = rotate_impedance_tensor(mtd[key].impedance[ii].values, 45.0)
            zrot = mtd[key].impedance[ii].values
            data_col_yx[freq]['real'] += [zrot[0, 1].real * _impUnitEDI2SI]
            # real data yx imag
            data_col_yx[freq]['imag'] += [zrot[0, 1].imag * _impUnitEDI2SI]
            locs = utm.from_latlon(mtd[key].latitude, mtd[key].longitude)[:2]
            data_col_yx[freq]['stn']  += [[locs[0], locs[1], mtd[key].elevation]]

            data_col_xy[freq]['real'] += [zrot[1, 0].real * _impUnitEDI2SI]
            # real data xy imag
            data_col_xy[freq]['imag'] += [zrot[1, 0].imag * _impUnitEDI2SI]
            data_col_xy[freq]['stn']  += [[locs[0], locs[1], mtd[key].elevation]]

    # Save everything needed to rebuild the SimPEG survey and data objects without MTpy
    _cache_out = {
        "rx_locs":          [list(map(float, r)) for r in rx_locs],
        "rx_locs_tipper":   [list(map(float, r)) for r in rx_locs_tipper],
        "elevation":        [float(e) for e in elevation],
        "elevation_tipper": [float(e) for e in elevation_tipper],
        "data_col_yx": {
            str(freq): {
                "real": [float(v) for v in data_col_yx[freq]["real"]],
                "imag": [float(v) for v in data_col_yx[freq]["imag"]],
                "stn":  [list(map(float, s)) for s in data_col_yx[freq]["stn"]],
            }
            for freq in data_col_yx
        },
        "data_col_xy": {
            str(freq): {
                "real": [float(v) for v in data_col_xy[freq]["real"]],
                "imag": [float(v) for v in data_col_xy[freq]["imag"]],
                "stn":  [list(map(float, s)) for s in data_col_xy[freq]["stn"]],
            }
            for freq in data_col_xy
        },
    }
    with open(_MT_CACHE_FILE, "w") as _f:
        json.dump(_cache_out, _f, indent=2)
    print(f'[INFO] MT survey data cached to {_MT_CACHE_FILE}')


# now determine the frequencies that all stations share
frequencies_2_use = []

print('[INFO] creating OcTree Mesh...')

# Build 3D receiver location array (easting, northing, elevation)
rx_locs = np.c_[np.array(rx_locs), np.array(elevation)]
np.save("raglan_locations.npy", rx_locs)
rx_locs[:, -1] = rx_locs[:, -1] - 5.0

load topo
topo = np.genfromtxt(f"{append_dir}/Dropbox/JohnLindsey/2026-mt3dinv/ArcticDEM_30m_EPSG26918.xyz")
topo = topo[~np.isnan(topo).any(axis=1)]

cs = 75  # finest cell size (m)
rx_east  = rx_locs[:, 0]
rx_north = rx_locs[:, 1]

# Same horizontal padding and depth as the tensor mesh
horiz_pad = [
    (16000, 1), (8000, 1), (4000, 1), (2000, 2), (1000, 2), (500, 2),
    (250, 2), (175, 2), (125, 2), (100, 5), (80, 20),
]
depth_cells = [
    (16000, 1), (8000, 1), (4000, 1), (2000, 1), (1000, 1), (750, 1),
    (500, 1), (375, 2), (225, 3), (175, 5), (125, 5), (100, 5),
    (90, 5), (80, 5), (75, 10),
]

pad_width   = sum(w * n for w, n in horiz_pad)
depth_total = sum(w * n for w, n in depth_cells)

# Survey centre — mesh origin is derived from this so the domain is symmetric
x_center = (rx_east.min()  + rx_east.max())  / 2
y_center = (rx_north.min() + rx_north.max()) / 2

# Total domain extents needed to contain the receiver footprint plus padding
total_width_x = (rx_east.max()  - rx_east.min()) + 2 * cs + 2 * pad_width
total_width_y = (rx_north.max() - rx_north.min()) + 2 * cs + 2 * pad_width

# Number of base cells — power of 2, large enough to cover the required extents
nCx = 2**int(np.ceil(np.log2(total_width_x / cs)))
nCy = 2**int(np.ceil(np.log2(total_width_y / cs)))
nCz = 2**int(np.ceil(np.log2(depth_total   / cs)))

# Domain origin: centre the mesh on the survey area; depth starts at surface
x_min = x_center - nCx * cs / 2
y_min = y_center - nCy * cs / 2
z_min = -depth_total

mesh = discretize.TreeMesh(
    [nCx * [cs], nCy * [cs], nCz * [cs]],
    x0=[x_min, y_min, z_min],
)

# Refine the receiver footprint box to the finest cell size (cs).
# octree_levels=[0, 2, 4] means: inside the box use finest cells (0 padding),
# then 2 cells at 2×cs, then 4 cells at 4×cs, grading outward.
mesh = refine_tree_xyz(
    mesh,
    topo,
    octree_levels=[0, 1],
    method='surface',
    finalize=False,
)
mesh = refine_tree_xyz(
    mesh,
    rx_locs,
    octree_levels=[4, 6, 1],
    method='surface',
    finalize=False,
)
mesh.finalize()

active_cells = discretize.utils.mesh_utils.active_from_xyz(mesh, rx_locs)

np.save("active_octree.npy", active_cells)
mesh.write_UBC("mesh_octree.txt")
# active_cells = np.load("active_octree.npy")
# mesh = discretize.TreeMesh.read_UBC("mesh_octree.txt")

# ---------------------------------------------------------------------------

# Setup the simpeg objects for survey and simulations for each TE & TM modes

#
src_list = []

# now determine the frequencies that all stations share
frequencies_2_use = []

for freq in data_col_xy.keys():
    frequencies_2_use += [freq]

# frequencies_2_use = frequencies_2_use[::10]

data_vec = []

data_real = []
data_imag = []

# correct for topography

station_number_count = np.zeros(len([frequencies_2_use[20], frequencies_2_use[25], frequencies_2_use[30]]))
for ii, freq in enumerate([frequencies_2_use[20], frequencies_2_use[25], frequencies_2_use[30]]):

    locations_mt_shifted = shift_to_discrete_topography(
        mesh, np.asarray(data_col_yx[freq]['stn']), active_cells, topo_cell_cutoff='top', shift_horizontal=False, heights=0
    )

    rx_list = [

        nsem.receivers.Impedance(
            locations_mt_shifted, orientation="yx", component="real"
        ),
        nsem.receivers.Impedance(
            locations_mt_shifted, orientation="yx", component="imag"
        ),

    ]

    station_number_count[ii] = locations_mt_shifted.shape[0]

    data_vec += [data_col_yx[freq]['real']]
    data_vec += [data_col_yx[freq]['imag']]

    locations_mt_shifted = shift_to_discrete_topography(
        mesh, np.asarray(data_col_xy[freq]['stn']), active_cells, topo_cell_cutoff='top', shift_horizontal=False, heights=0
    )

    rx_list += [
        nsem.receivers.Impedance(
            locations_mt_shifted, orientation="xy", component="real"
        ),
        nsem.receivers.Impedance(
            locations_mt_shifted, orientation="xy", component="imag"
        ),
    ]

    data_vec += [data_col_xy[freq]['real']]
    data_vec += [data_col_xy[freq]['imag']]
    # data_real += [data_col_yx[freq]['real']]
    # data_imag += [data_col_yx[freq]['imag']]

    src_list += [nsem.sources.PlanewaveXYPrimary(rx_list, frequency=freq)]
    # src_list += [nsem.sources.FictitiousSource(receiver_list=rx_list, frequency=freq)]

data_vec = np.hstack(data_vec)
# data_vec_tm = np.hstack(data_vec_tm)

# setup the survey
survey = nsem.Survey(src_list)

data_obj = data.Data(survey, data_vec)

# now the simulations
deriv_type = "sigma"
sim_type = "h"
fixed_boundary=True

if False:
    from matplotlib.backends.backend_pdf import PdfPages
    from plot_station_data import plot_station_data, find_nearest_station

    predicted = np.load("predicted_data.npy")

    src0 = survey.source_list[0]
    n_stations = data_obj[src0, src0.receiver_list[0]].shape[0]
    pdf_path = "station_plots_all_freq_v2.pdf"
    with PdfPages(pdf_path) as pdf:
        for idx in range(n_stations):
            fig, axes = plot_station_data(data_obj, station_idx=idx, predicted_data=predicted)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved {n_stations} station plots to {pdf_path}")


actmap = maps.InjectActiveCells(

    mesh, active_cells=active_cells, value_inactive=np.log(1e-8)

)

sigBG = np.zeros(mesh.nC) + 1 / 3000
sigBG[~active_cells] = 1e-8

m0 = (np.ones(mesh.nC) * np.log(1/3000))[active_cells]
# Set the mapping
actMap = maps.InjectActiveCells(
    mesh=mesh, active_cells=active_cells, value_inactive=np.log(1e-8)
)
mapping = maps.ExpMap(mesh) * actMap

# Setup the problem (As a multiprocessing meta sim split by source)
# If you don't want to use the MultiprocessingMetaSim branch, you can just comment
# the below lines out and replace sim with the normal sim that is commented out
# below
mappings = []
sims = []
for src in src_list:
    mappings.append(maps.IdentityMap())
    srv_piece = nsem.Survey([src,])
    # sims.append(nsem.Simulation3DPrimarySecondary(
    #     mesh, survey=srv_piece, sigmaMap=mapping, sigmaPrimary=sigBG, solver=Solver
    # ))

    sims.append(nsem.Simulation3DElectricFieldFictitious(
        mesh, survey=srv_piece, sigmaMap=mapping, sigma_background=sigBG, solver=Solver
    ))

sim_mt = MultiprocessingMetaSimulation(sims, mappings, n_processes=11)
    
# sim_mt = nsem.simulation.Simulation3DPrimarySecondary(
#     mesh,
#     survey=survey,
#     sigmaMap=mapping,
#     sigmaPrimary=sigBG,
#     solver=Solver
# )

sim_mt.model = sigBG[active_cells]

# ---------------------------------------------------------------------------

# create the data misfits for each mode

#

print('[INFO] Getting things started on inversion...')

# TE mode
# data_obj.standard_deviation = np.abs(data_vec) * 0.05

dmis_admm = data_misfit.L2DataMisfit(data=data_obj, simulation=sim_mt)

# assign the weights
dmis_admm.W = 1. / (np.abs(data_obj.dobs) * 0.05 + 0.3)

# Map for a regularization
regmap = maps.IdentityMap(nP=int(active_cells.sum()))

m0 = (np.ones(mesh.nC) * np.log(1/8000))[active_cells]
z0 = m0.copy()
u0 = np.zeros_like(z0) # np.random.randn(z0.shape[0])
idenMap = maps.IdentityMap(nP=m0.shape[0])

m = m0.copy()
z = z0.copy()
u = u0.copy()

print("calculating half-space response...")
halfspace_data = sim_mt.dpred(m0)
np.save("halfspace_response.npy", halfspace_data)

opt = optimization.ProjectedGNCG(
    maxIter=1,
    maxIterCG=10,
    upper=np.inf,
    lower=-np.inf,
    tolCG=1E-5,
    maxIterLS=20,
)

opt.remember('xc')

# solver_opts = dmis_admm.simulation.solver_opts

# reg = regularization.Smallness(
#     mesh=mesh,
#     active_cells=active_cells,
#     reference_model=(z + u),
# )

reg = regularization.WeightedLeastSquares(
    mesh=mesh,
    active_cells=active_cells,
    mapping=idenMap,
    reference_model=z + u,
)
reg.alpha_x = 100
reg.alpha_y = 100

reg.alpha_s = 1e-10

coolingFactor = 2
coolingRate = 2
beta0_ratio = 1e0

opt_tetm = optimization.ProjectedGNCG(maxIter=15, upper=np.inf, lower=-np.inf)
invProb_tetm = inverse_problem.BaseInvProblem(dmis_admm, reg, opt_tetm)
beta = directives.BetaSchedule(
    coolingFactor=coolingFactor, coolingRate=coolingRate
)
betaest = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)
target = directives.TargetMisfit()
savedict = directives.SaveOutputEveryIteration()
save_all = directives.SaveOutputDictEveryIteration()

directiveList = [
    beta,
    betaest,
    target,
    savedict,
    save_all,
]

inv_tetm = inversion.BaseInversion(
    invProb_tetm, directiveList=directiveList)
# opt.LSshorten = 0.5
opt_tetm.remember('xc')

# Run Inversion
minv_tetm = inv_tetm.run(m0)
np.save("final_model.npy", minv_tetm)
predicted_data = sim_mt.dpred(minv_tetm)
np.save("predicted_data.npy", predicted_data)
print(f"number of data is: {data_obj.dobs.shape[0]}")

