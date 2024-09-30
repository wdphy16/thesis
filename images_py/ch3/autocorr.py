#!/usr/bin/env python3

import h5py
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"

configs = [
    (0.2, "C1"),
    (0.44, "C2"),
    (1, "C0"),
]

in_template = "../data/mcmc_ising/L10_beta{beta:g}.hdf5"
out_filename = "./autocorr.pdf"


def get_autocorr_times(x, size=None):
    if size is None:
        size = x.size

    x_c = x - x.mean()
    f = np.fft.fft(x_c, 2 * x.size)
    t = np.fft.ifft(f * f.conj())[:size].real
    t /= size - np.arange(size)

    if t[0] < 1e-15:
        t[:] = 0
    else:
        t /= t[0]

    return t


def main():
    fig, ax = plt.subplots(figsize=(4, 2.5))

    ax.axhline(0, color="0.7", linestyle="--", linewidth=0.5)

    for beta, color in configs:
        in_filename = in_template.format(beta=beta)
        print(in_filename)
        with h5py.File(in_filename, "r") as f:
            energies = np.asarray(f["E"])
        autocorr = get_autocorr_times(energies)
        autocorr = autocorr[: 10**4]

        ax.plot(autocorr, color=color, label=f"$\\beta={beta:.2f}$", linewidth=1)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$C_E(t)$")
    ax.legend()

    fig.tight_layout()
    print(out_filename)
    fig.savefig(out_filename, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
