#!/usr/bin/env python3

import h5py
import numpy as np
from KDEpy import FFTKDE
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"

out_filename = "./mag_hist.pdf"


def main():
    fig, axes = plt.subplots(ncols=2, figsize=(6, 2.5))

    ax = axes[0]

    in_filename = "../data/mcmc_ising/L10_beta0.44.hdf5"
    print(in_filename)
    with h5py.File(in_filename, "r") as f:
        mags = np.asarray(f["M"][10**6 : 10**6 + 10**5])
    mags = np.concatenate([mags, -mags])

    xs, hist = FFTKDE(bw="silverman").fit(mags).evaluate()

    ax.plot(xs, hist, linewidth=1)

    ax.set_xlabel("$M$")
    ax.set_ylabel("PDF")
    ax.set_xlim(-125, 125)
    ax.set_ylim(0, 0.015)
    ax.set_yticks([0, 0.01])

    ax = axes[1]

    in_filename = "../data/mcmc_ising/L10_beta0.44.hdf5"
    print(in_filename)
    with h5py.File(in_filename, "r") as f:
        mags = np.asarray(f["M"][10**6 : 2 * 10**6 : 10**2])
    steps = np.arange(0, 10**6, 10**2)

    ax.plot(steps, -mags, linewidth=1)

    ax.set_xlabel("Step")
    ax.set_ylabel("$M$")

    # Panel labels
    for i, ax in enumerate(axes.flatten()):
        c = chr(ord("a") + i)
        ax.text(0.05, 0.9, f"({c})", transform=ax.transAxes)

    fig.tight_layout()
    print(out_filename)
    fig.savefig(out_filename, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
