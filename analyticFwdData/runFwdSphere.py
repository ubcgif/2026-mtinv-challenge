import numpy as np
import os
from scipy.constants import mu_0, pi

from layeredearthfunctions import PlaneWaveImpedance, apparentresistivity
from mtsphere3d import mtsphere3d


# =========================================================
# 1. HALF-SPACE MODEL
# =========================================================
rho_halfspace = 1000.0

nlyr = 1

res = np.zeros(nlyr + 1)
res[1] = rho_halfspace

thk = np.zeros(nlyr)


# =========================================================
# 2. SPHERE PARAMETERS
# =========================================================
radius = 500.0
depth = 1000.0
sphres = 0.1   # sphere conductivity


# =========================================================
# 3. FREQUENCIES
# =========================================================
nf = 71
freq = np.logspace(-3, 4, nf)   # 0.001 to 10000 Hz

# =========================================================
# 4. RECEIVER LOCATIONS
# =========================================================
x_list = np.array([
-5000,-2000,-1500,-1000,-750,-500,-400,-300,-250,-200,-150,-125,-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,
0,10,20,30,40,50,60,70,80,90,100,125,150,200,250,300,400,500,750,1000,1500,2000,5000
])

y_list = np.array([
-5000,-2000,-1500,-1000,-750,-500,-400,-300,-250,-200,-150,-125,-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,
0,10,20,30,40,50,60,70,80,90,100,125,150,200,250,300,400,500,750,1000,1500,2000,5000
])

nx = len(x_list)
ny = len(y_list)


pxy = np.zeros((nx, ny, 3))
for i, x in enumerate(x_list):
    for j, y in enumerate(y_list):
        pxy[i, j, 0] = x
        pxy[i, j, 1] = y


# =========================================================
# 5. BACKGROUND FIELD
# =========================================================

outfile = open(os.devnull, 'w')

Zhat, E = PlaneWaveImpedance(
    outfile,
    nlyr,
    thk,
    res,
    nf,
    freq
)

imp, Es, Hs, Et, Ht = mtsphere3d(
    outfile,
    nf,
    nlyr,
    6,
    nx,
    ny,
    freq,
    pxy,
    thk,
    res,
    depth,
    radius,
    sphres,
    Zhat,
    E,
    {'dlf': 'key_401_2009'}
)

outfile.close()


# =========================================================
# 6. APPARENT RESISTIVITY + PHASE
# =========================================================
appres, phase = apparentresistivity(
    nf,
    nx,
    ny,
    freq,
    imp
)


print("Simulation complete")
print("Apparent resistivity shape:", appres.shape)
print("Phase shape:", phase.shape)

np.save('dataOut/imp0.npy', imp)
np.save('dataOut/appres0.npy', appres)
np.save('dataOut/phase0.npy', phase)