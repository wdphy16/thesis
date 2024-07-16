#!/usr/bin/env python3

import h5py
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"

in_template = "../data/mcmc_ising/L10_beta0.2{seed_str}.hdf5"
out_filename = "./multi_chains.pdf"


def ma(arr, *, window=10**4):
    if not window:
        return arr

    out = np.nancumsum(arr)
    out[window:] -= out[:-window]
    counts = np.cumsum(np.isfinite(arr))
    counts[window:] -= counts[:-window]
    out = out / np.maximum(counts, 1)
    return out


def main():
    fig, ax = plt.subplots(figsize=(4, 2.5))

    for seed in range(5):
        if seed == 0:
            seed_str = ""
        else:
            seed_str = f"_seed{seed}"
        in_filename = in_template.format(seed_str=seed_str)
        print(in_filename)
        with h5py.File(in_filename, "r") as f:
            energies = np.asarray(f["E"][: 10**4])
        energies = ma(energies)

        ax.plot(energies, linewidth=0.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("$\\mathrm{\\mathbb{E}}[E]$")

    fig.tight_layout()
    print(out_filename)
    fig.savefig(out_filename, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
