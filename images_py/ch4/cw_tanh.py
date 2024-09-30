#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"

out_filename = "./cw_tanh.pdf"


def main():
    fig, ax = plt.subplots(figsize=(4, 2.5))

    xs = np.linspace(-1.2, 1.2, 101)

    ax.axhline(0, color="0.7", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="0.7", linestyle="--", linewidth=0.5)

    ax.plot(xs, xs, color="k", label="$y = x$", linewidth=1)
    ax.plot(
        xs,
        np.tanh(0.5 * xs),
        color="C1",
        label="$y = \\tanh \\frac{1}{2} x$",
        linewidth=1,
    )
    ax.plot(xs, np.tanh(xs), color="C2", label="$y = \\tanh x$", linewidth=1)
    ax.plot(xs, np.tanh(2 * xs), color="C0", label="$y = \\tanh 2 x$", linewidth=1)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.legend(fontsize="small")

    fig.tight_layout()
    print(out_filename)
    fig.savefig(out_filename, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
