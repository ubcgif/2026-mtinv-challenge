"""
How do you put a sphere onto a rectangular mesh?

DETERMINNIG THE FRACTION
cell_fill_fraction estimates what fraction of each cell's volume lies inside the sphere by creating a regular grid.
For each cell, it places evenly-spaced points along each axis, spanning the cell's width.
With subsamples=8 in 2-D that's a 8×8 = 64-point grid per cell.
The points are offset from the cell centre, so they tile the interior of the cell uniformly without touching its edges.

It then computes the Euclidean distance from every sub-point to the sphere centre and counts how many fall within RADIUS.
That count divided by the total number of sub-points is the fraction.

THE AVERAGE
There are multiple ways to average the conductivity. I found that the harmonic one is doing the best job.
1.0 / (frac / SIGMA_SPHERE + (1.0 - frac) / SIGMA_HOST)

"""

import os
import matplotlib.pyplot as plt
import numpy as np

SIGMA_SPHERE = 10.0
SIGMA_HOST = 1e-3
RADIUS = 500.0
CENTRE = np.array([0.0,0.0])

RULES = ["hard", "harmonic"]
RULE_TITLES = {
    "hard":     "Hard edge (staircase)",
    "harmonic": "Harmonic blend (average rho)",
}


def cell_fill_fraction(centers, widths, subsamples=8):
    """Fraction of each cell that lies inside the sphere"""
    ndim = centers.shape[1]
    offs = (np.arange(subsamples) + 0.5) / subsamples - 0.5
    sub = np.stack(np.meshgrid(*([offs] * ndim), indexing="ij"), axis=-1).reshape(-1, ndim)
    pts = centers[:, None, :] + sub[None, :, :] * widths[:, None, :]
    return (np.linalg.norm(pts - CENTRE, axis=2) <= RADIUS).mean(axis=1)


def uniform_grid_2d(extent, cell):
    """Square grid of cells over [-extent, extent]^2 with side length ``cell``."""
    n = int(round(2 * extent / cell))
    edges = np.linspace(-extent, extent, n + 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    xx, zz = np.meshgrid(c, c, indexing="xy")
    centers = np.column_stack([xx.ravel(), zz.ravel()])
    widths = np.full_like(centers, cell)
    return centers, widths, xx.shape


def build_model_2d(centers, widths, rule, subsamples=8):
    """Conductivity for every cell of a 2-D grid under the chosen ``rule``."""
    if rule == "hard":
        return np.where(np.linalg.norm(centers - CENTRE, axis=1) <= RADIUS, SIGMA_SPHERE, SIGMA_HOST)
    frac = cell_fill_fraction(centers, widths, subsamples)
    return 1.0 / (frac / SIGMA_SPHERE + (1.0 - frac) / SIGMA_HOST)


def figure_models(cell=160.0, extent=1100.0):
    centers, widths, shape = uniform_grid_2d(extent, cell)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_x = CENTRE[0] + RADIUS * np.cos(theta)
    circle_z = CENTRE[1] + RADIUS * np.sin(theta)

    im = None
    for ax, rule in zip(axes, RULES):
        sigma = build_model_2d(centers, widths, rule).reshape(shape)
        im = ax.imshow(
            np.log10(sigma), origin="lower", extent=[-extent, extent, -extent, extent],
            cmap="viridis", vmin=np.log10(SIGMA_HOST), vmax=np.log10(SIGMA_SPHERE),
        )
        ax.plot(circle_x, circle_z, "w--", lw=1.5)
        ax.set_title(RULE_TITLES[rule])
        ax.set_aspect("equal")

    cbar = fig.colorbar(im, ax=axes)
    return fig


def main():
    fig = figure_models()
    fig.savefig("models.pdf")
    plt.show()


if __name__ == "__main__":
    main()
