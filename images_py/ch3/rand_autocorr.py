#!/usr/bin/env python3

import h5py
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"

out_filename = "./rand_autocorr.pdf"


def main():
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(6, 2.5))

    ax = axes[0]

    filename = "../data/mcmc_ising/L10_rand.hdf5"
    print(filename)
    with h5py.File(filename, "r") as f:
        energies = np.asarray(f["E"][:100])

    ax.plot(energies, linewidth=1)

    ax = axes[1]

    filename = "../data/mcmc_ising/L10_beta0.hdf5"
    print(filename)
    with h5py.File(filename, "r") as f:
        energies = np.asarray(f["E"][:100])

    ax.plot(energies, linewidth=1)

    # Panel labels
    for i, ax in enumerate(axes.flatten()):
        c = chr(ord("a") + i)
        ax.text(0.05, 0.9, f"({c})", transform=ax.transAxes)

    axes[0].set_ylabel("$E$")
    axes[0].set_ylim(-50, 50)
    for ax in axes:
        ax.set_xlabel("Step")

    fig.tight_layout()
    print(out_filename)
    fig.savefig(out_filename, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
