"""
Potential field transformations, like upward continuation and derivatives.

.. note:: Most, if not all, functions here required gridded data.

**Transformations**

* :func:`~fatiando.gravmag.transform.upcontinue`: Upward continuation of
  gridded potential field data on a level surface.
* :func:`~fatiando.gravmag.transform.reduce_to_pole`: Reduce the total field
  magnetic anomaly to the pole.
* :func:`~fatiando.gravmag.transform.tga`: Calculate the amplitude of the
  total gradient (also called the analytic signal)
* :func:`~fatiando.gravmag.transform.tilt`: Calculates the tilt angle
* :func:`~fatiando.gravmag.transform.power_density_spectra`: Calculates
  the Power Density Spectra of a gridded potential field data.
* :func:`~fatiando.gravmag.transform.radial_average`: Calculates the
  the radial average of a Power Density Spectra using concentring rings.

**Derivatives**

* :func:`~fatiando.gravmag.transform.derivx`: Calculate the n-th order
  derivative of a potential field in the x-direction (North-South)
* :func:`~fatiando.gravmag.transform.derivy`: Calculate the n-th order
  derivative of a potential field in the y-direction (East-West)
* :func:`~fatiando.gravmag.transform.derivz`: Calculate the n-th order
  derivative of a potential field in the z-direction

----

"""

import warnings
import numpy as np
from SimPEG.utils import mkvc
from scipy.spatial import cKDTree, ConvexHull, Delaunay
from tqdm import tqdm
from matplotlib.colors import ListedColormap


def upward_continuation(x, y, data, shape, height):
    r"""
    Upward continuation of potential field data.

    Calculates the continuation through the Fast Fourier Transform in the
    wavenumber domain (Blakely, 1996):

    .. math::

        F\{h_{up}\} = F\{h\} e^{-\Delta z |k|}

    and then transformed back to the space domain. :math:`h_{up}` is the upward
    continue data, :math:`\Delta z` is the height increase, :math:`F` denotes
    the Fourier Transform,  and :math:`|k|` is the wavenumber modulus.

    .. note:: Requires gridded data.

    .. note:: x, y, z and height should be in meters.

    .. note::

        It is not possible to get the FFT of a masked grid. The default
        :func:`fatiando.gridder.interp` call using minimum curvature will not
        be suitable.  Use ``extrapolate=True`` or ``algorithm='nearest'`` to
        get an unmasked grid.

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    * height : float
        The height increase (delta z) in meters.

    Returns:

    * cont : array
        The upward continued data

    References:

    Blakely, R. J. (1996), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

    """
    assert x.shape == y.shape, \
        "x and y arrays must have same shape"
    if height <= 0:
        warnings.warn("Using 'height' <= 0 means downward continuation, " +
                      "which is known to be unstable.")
    nx, ny = shape
    # Pad the array with the edge values to avoid instability
    padded, padx, pady = _pad_data(data, shape)
    kx, ky = _fftfreqs(x, y, shape, padded.shape)
    kz = np.sqrt(kx**2 + ky**2)
    upcont_ft = np.fft.fft2(padded)*np.exp(-height*kz)
    cont = np.real(np.fft.ifft2(upcont_ft))
    # Remove padding
    cont = cont[padx: padx + nx, pady: pady + ny].ravel()
    return cont



def _upcontinue_space(x, y, data, shape, height):
    """
    Upward continuation using the space-domain formula.

    DEPRECATED. Use the better implementation using FFT. Kept here for
    historical reasons.

    """
    nx, ny = shape
    dx = (x.max() - x.min())/(nx - 1)
    dy = (y.max() - y.min())/(ny - 1)
    area = dx*dy
    deltaz_sqr = (height)**2
    cont = np.zeros_like(data)
    for i, j, g in zip(x, y, data):
        cont += g*area*((x - i)**2 + (y - j)**2 + deltaz_sqr)**(-1.5)
    cont *= abs(height)/(2*np.pi)
    return cont


def _pad_data(data, shape):
    n = _nextpow2(np.max(shape))
    nx, ny = shape
    padx = (n - nx)//2
    pady = (n - ny)//2
    padded = np.pad(data.reshape(shape), ((padx, padx), (pady, pady)),
                       mode='edge')
    return padded, padx, pady


def _nextpow2(i):
    buf = np.ceil(np.log(i)/np.log(2))
    return int(2**buf)


def _fftfreqs(x, y, shape, padshape):
    """
    Get two 2D-arrays with the wave numbers in the x and y directions.
    """
    nx, ny = shape
    dx = (x.max() - x.min())/(nx - 1)
    fx = 2*np.pi*np.fft.fftfreq(padshape[0], dx)
    dy = (y.max() - y.min())/(ny - 1)
    fy = 2*np.pi*np.fft.fftfreq(padshape[1], dy)
    return np.meshgrid(fy, fx)[::-1]


def downsample_xy(locations, radius):
    """
    downsample_xy(locations)

    Function to downsample a cloud of points in 2D based on
    distance between neighbours

    Parameter
    ---------

    locations: np.ndarray
        Point locations [nx2]

    radius: float
        Minimum radial distance between points


    Return
    ------

    index: bool
        Array of bool of shape n for points to stay

    """

    tree = cKDTree(locations[:, :2])

    nstn = locations.shape[0]
    # Initialize the filter
    index = np.ones(nstn, dtype='bool')

    count = -1
    print("Begin filtering for radius= " + str(radius))

    for ii in tqdm(range(nstn)):

        if index[ii]:

            ind = tree.query_ball_point(locations[ii, :2], radius)

            index[ind] = False
            index[ii] = True

        # count = progress(ii, count, nstn)

    return index



def transform_to_local(r0, phi, xyzd):

    r0 = mkvc(r0)

    xy_new = np.c_[
        (xyzd[:, 0]-r0[0])*np.cos(phi) - (xyzd[:, 1]-r0[1])*np.sin(phi),
        (xyzd[:, 0]-r0[0])*np.sin(phi) + (xyzd[:, 1]-r0[1])*np.cos(phi)
    ]

    if np.shape(xyzd)[1] > 2:
        return np.c_[xy_new, xyzd[:, 2:]]
    else:
        return xy_new


def transform_to_global(r0, phi, xyzd):

    r0 = mkvc(r0)

    xy_new = np.c_[
        xyzd[:, 0]*np.cos(-phi) - xyzd[:, 1]*np.sin(-phi),
        xyzd[:, 0]*np.sin(-phi) + xyzd[:, 1]*np.cos(-phi)
    ]

    xy_new = xy_new + np.c_[r0[0], r0[1]]

    if np.shape(xyzd)[1] > 2:
        return np.c_[xy_new, xyzd[:, 2:]]
    else:
        return xy_new


def extract_from_polygon(poly_pts, xyzd, index_only=False):

    delaunay_object = Delaunay(poly_pts)  # Make a Delaunay interpolation object
    k = delaunay_object.find_simplex(xyzd[:, 0:2])>=0  # Figure out which RTF data locations are within the Delaunay simplex
    if index_only:
        return k
    else:
        return xyzd[k, :]




def generate_mag_colormap(vmin, vmax, mid):
    """Generate magnetics colormap

    vmin: minimum value for colorscale
    vmax: maximum value for colorscale
    mid: the value being assigned as white (no anomaly)
    """
    N = 105
    
    k_neg = np.int16(np.floor(105*(mid-vmin)/(vmax-vmin)))
    k_pos = 105 - k_neg
    
    C = 105 / np.max([k_neg, k_pos])
    
    k_neg = np.int16(C*k_neg)
    k_pos = np.int16(C*k_pos)
    
    v = np.linspace(0, 1, 40)
    
    neg_seg1 = np.c_[np.zeros(5), np.zeros(5), 0.5*np.ones(5)]
    neg_seg2 = np.c_[np.zeros_like(v), 0.5*v, 0.5*np.ones_like(v)]
    neg_seg3 = np.c_[np.zeros_like(v), 0.5*(1+v), 0.5*(1+v)]
    neg_seg4 = np.c_[v[0::2], np.ones_like(v[0::2]), np.ones_like(v[0::2])]
    neg_base = np.r_[neg_seg1, neg_seg2, neg_seg3, neg_seg4]
    
    pos_seg1 = np.c_[np.ones_like(v[0::2]), 1-0.5*v[0::2], 1-v[0::2]]
    pos_seg2 = np.c_[np.ones_like(v), 0.5*(1-v), np.zeros_like(v)]
    pos_seg3 = np.c_[1-0.4*v, np.zeros_like(v), 0.2*v]
    pos_seg4 = np.c_[0.6*np.ones(5), np.zeros(5), 0.2*np.ones(5)]
    pos_base = np.r_[pos_seg1, pos_seg2, pos_seg3, pos_seg4]
    
    base = np.r_[neg_base[N-k_neg:, :], pos_base[:k_pos, :]]
    base = np.c_[base, np.ones((np.shape(base)[0]))]
    
    return ListedColormap(base)
    
    