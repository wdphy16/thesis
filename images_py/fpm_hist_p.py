#!/usr/bin/env python3

import h5py
import numpy as np
from KDEpy import FFTKDE
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"

config = [
    (16, 0.211, 0.212, 0.21136, "C1"),
    (24, 0.214, 0.215, 0.21408, "C2"),
    (32, 0.215, 0.216, 0.21522, "C0"),
]
min_energy = -4.5
max_energy = -0.5
in_template = "../data/mcmc_fpm/L{L}_beta{beta:g}_rep{rep}.hdf5"
out_filename = "./fpm_hist_p.pdf"


def get_hist(L, beta):
    energies = []
    for rep in range(1, 4):
        filename = in_template.format(L=L, beta=beta, rep=rep)
        with h5py.File(filename, "r") as f:
            _energies = np.asarray(f["energies"])
        _energies = _energies / L**2
        energies.append(_energies)

    bin_density = L**2 // 4
    energy_ticks = np.linspace(
        min_energy, max_energy, round((max_energy - min_energy) * bin_density)
    )
    hists = []
    for _energies in energies:
        print(L, beta, _energies.min(), _energies.max())
        hist = FFTKDE(bw="silverman").fit(energies)(energy_ticks)
        hists.append(hist)

    hist = np.stack(hists, axis=1)
    hist_mean = hist.mean(axis=1)
    hist_std = hist.std(axis=1)
    hist_std = np.nan_to_num(hist_std)

    return energy_ticks, hist_mean, hist_std


def adjust_beta(hist, L, beta, beta_target, energy_ticks):
    delta_beta_energy = (beta - beta_target) * energy_ticks * L**2
    hist *= np.exp(delta_beta_energy)
    bin_density = L**2 // 4
    hist /= (hist / bin_density).sum()
    return hist


def main():
    fig, ax = plt.subplots(figsize=(5, 3))

    for L, beta_1, beta_2, beta_target, color in config:
        energy_ticks, hist_1, _ = get_hist(L, beta_1)
        energy_ticks, hist_2, _ = get_hist(L, beta_2)

        hist_1 = adjust_beta(hist_1, L, beta_1, beta_target, energy_ticks)
        hist_2 = adjust_beta(hist_2, L, beta_2, beta_target, energy_ticks)

        lam = (beta_2 - beta_target) / (beta_2 - beta_1)
        hist = lam * hist_1 + (1 - lam) * hist_2

        ax.plot(energy_ticks, hist, color=color, label=f"$L = {L}$", linewidth=1)

    ax.set_xlabel("$\\epsilon$")
    ax.set_ylabel("PDF")
    ax.set_xticks([-3.5, -3, -2.5, -2])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8])
    ax.set_xlim(-3.575, -1.925)
    ax.set_ylim(0.45, 0.85)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_filename, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
