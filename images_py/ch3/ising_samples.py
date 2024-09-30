#!/usr/bin/env python3

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"

in_template = "../data/mcmc_ising/L100_beta{beta:g}.hdf5"
out_filename = "./ising_samples.pdf"
betas = [0.3, 0.44, 0.6]


def main():
    fig, axes = plt.subplots(ncols=len(betas), figsize=(6, 2))

    for ax, beta in zip(axes, betas):
        in_filename = in_template.format(beta=beta)
        print(in_filename)
        with h5py.File(in_filename, "r") as f:
            sample = np.asarray(f["sample"])

        cmap = ListedColormap(["C1", "C0"])
        ax.imshow(sample, cmap=cmap, interpolation="none")
        ax.set_axis_off()

        ax.set_title(f"$\\beta = {beta:.2f}$", fontsize="medium")

    fig.tight_layout()
    print(out_filename)
    fig.savefig(out_filename, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
