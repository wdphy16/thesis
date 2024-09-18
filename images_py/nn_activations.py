#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"

out_filename = "./nn_activations.pdf"


def main():
    fig, ax = plt.subplots(figsize=(4, 2.5))

    xs = np.linspace(-3, 3, 101)

    ax.axhline(0, color="0.7", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="0.7", linestyle="--", linewidth=0.5)

    ax.plot(
        xs,
        np.maximum(xs, 0),
        color="C0",
        label="$y = \\mathrm{ReLU}(x)$",
        linewidth=1,
    )
    ax.plot(
        xs,
        xs * norm.cdf(xs),
        color="C1",
        label="$y = \\mathrm{GELU}(x)$",
        linewidth=1,
    )
    ax.plot(
        xs,
        1 / (1 + np.exp(-xs)),
        color="C2",
        label="$y = \\mathrm{Sigmoid}(x)$",
        linewidth=1,
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    # ax.set_xticks([-1, 0, 1])
    # ax.set_yticks([-1, 0, 1])
    ax.legend(fontsize="small")

    fig.tight_layout()
    print(out_filename)
    fig.savefig(out_filename, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
