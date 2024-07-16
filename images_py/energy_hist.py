#!/usr/bin/env python3

import h5py
import numpy as np
from KDEpy import FFTKDE
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"

out_filename = "./energy_hist.pdf"


def main():
    in_filename = "../data/mcmc_ising/L10_rand.hdf5"
    print(in_filename)
    with h5py.File(in_filename, "r") as f:
        energies = np.asarray(f["E"])
    energies = np.concatenate([energies, -energies])

    xs, hist = FFTKDE(bw="silverman").fit(energies).evaluate()

    in_filename = "../data/mcmc_ising/L10_beta0.2.hdf5"
    print(in_filename)
    with h5py.File(in_filename, "r") as f:
        energies_beta = np.asarray(f["E"][10**6 : 10**6 + 3 * 10**4])

    xs_beta, hist_beta = FFTKDE(bw="silverman").fit(energies_beta).evaluate()

    fig, ax = plt.subplots(figsize=(4, 2.5))

    ax.plot(xs, hist, color="C1", label="$\\beta = 0.0$", linewidth=1)
    ax.plot(xs_beta, hist_beta, color="C0", label="$\\beta = 0.2$", linewidth=1)

    ax.set_xlabel("$E$")
    ax.set_ylabel("PDF")
    ax.set_xlim(-100, 50)
    ax.set_ylim(0, 0.03)
    ax.set_yticks([0, 0.01, 0.02, 0.03])
    ax.legend()

    fig.tight_layout()
    print(out_filename)
    fig.savefig(out_filename, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
