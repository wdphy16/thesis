#!/usr/bin/env python3
#
# 2D classical Ising model

import h5py
import numpy as np
from numba import njit

L = 10
max_step = 10**3
seed = 0
out_filename = f"../data/mcmc_ising/L{L}_rand.hdf5"


def get_init_sample(rng):
    return rng.integers(2, size=(L, L)) * 2 - 1


@njit
def get_energy(s):
    energy = 0
    for i in range(L):
        for j in range(L):
            h = s[(i + 1) % L, j] + s[i, (j + 1) % L]
            # Triangular lattice
            # h += s[(i + 1) % L, (j + 1) % L]
            # Ferromagnetic
            h *= -1
            energy += h * s[i, j]
    return energy


def main():
    rng = np.random.default_rng(seed)
    energies = np.empty([max_step], dtype=np.int32)
    mags = np.empty([max_step], dtype=np.int32)
    for i in range(max_step):
        sample = get_init_sample(rng)
        energies[i] = get_energy(sample)
        mags[i] = sample.sum()

    print(out_filename)
    with h5py.File(out_filename, "w") as f:
        f.create_dataset("sample", data=sample, compression="gzip", shuffle=True)
        f.create_dataset("E", data=energies, compression="gzip", shuffle=True)
        f.create_dataset("M", data=mags, compression="gzip", shuffle=True)


if __name__ == "__main__":
    main()
