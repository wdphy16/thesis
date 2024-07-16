#!/usr/bin/env python3
#
# 2D classical Ising model

import random
from math import exp

import h5py
import numpy as np
from numba import njit

L = 100
beta = 0.8
max_step = 10**9
print_step = 10**6
seed = 0
out_filename = f"../data/mcmc_ising/L{L}_beta{beta:g}.hdf5"


def get_init_sample():
    return np.random.default_rng(seed).integers(2, size=(L, L)) * 2 - 1


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


@njit
def do_sample(s):
    random.seed(seed)
    energy = get_energy(s)
    mag = s.sum()
    for step in range(max_step):
        if (step + 1) % print_step == 0:
            print(int((step + 1) / max_step * 100), "%")

        i = random.randint(0, L - 1)
        j = random.randint(0, L - 1)
        h = (
            s[(i + 1) % L, j]
            + s[(i - 1) % L, j]
            + s[i, (j + 1) % L]
            + s[i, (j - 1) % L]
        )
        # Triangular lattice
        # h += s[(i + 1) % L, (j + 1) % L] + s[(i - 1) % L, (j - 1) % L]
        # Ferromagnetic
        h *= -1
        delta_energy = 2 * h * s[i, j]

        if delta_energy > 0 or random.random() < exp(beta * delta_energy):
            energy -= delta_energy
            mag -= 2 * s[i, j]
            s[i, j] *= -1


def main():
    sample = get_init_sample()
    do_sample(sample)

    print(out_filename)
    with h5py.File(out_filename, "w") as f:
        f.create_dataset("sample", data=sample, compression="gzip", shuffle=True)


if __name__ == "__main__":
    main()
