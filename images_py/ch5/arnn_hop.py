#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"

plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.5

out_filename = "./arnn_hop.pdf"


def main():
    row_A = ["A0", "cA0", ".", "A1", "cA1", ".", "A2", "cA2"]
    row_B = ["B0", "cB0", ".", "B1", "cB1", ".", "B2", "cB2"]
    fig, axes = plt.subplot_mosaic(
        [row_A, row_B],
        width_ratios=[20, 1, 1, 20, 1, 1, 20, 1],
        per_subplot_kw={("A0", "A1", "A2"): {"projection": "3d"}},
        figsize=(6, 3),
        layout="constrained",
    )

    for i, beta in enumerate([0.3, 1, 1.5]):
        data = np.loadtxt(f"../data/arnn/hop/beta{beta:g}.txt")
        zs = data[:, 0]
        xs = data[:, 2]
        ys = data[:, 1]

        ax = axes[f"A{i}"]

        im = ax.plot_trisurf(xs, ys, zs, cmap="rainbow")

        ax.set_title(f"$\\beta = {beta:.1f}$", fontsize="medium")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-80, -10)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_zticks([-60, -40, -20])
        ax.set_xticks([-0.5, 0.5], minor=True)
        ax.set_yticks([-0.5, 0.5], minor=True)
        ax.tick_params(axis="x", which="major", labelsize="small", pad=-2)
        ax.tick_params(axis="y", which="major", labelsize="small", pad=-2)
        ax.tick_params(axis="z", which="major", labelsize="small", pad=0)
        ax.set_box_aspect((1, 1, 0.8))
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_rotate_label(False)
        ax.set_xlabel("$O_1$", labelpad=-5)
        ax.set_ylabel("$O_2$", labelpad=-5)
        if i == 0:
            ax.set_zlabel("$\\ln q$", labelpad=0, rotation=90)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_pane_color("w")
        ax.view_init(elev=15, azim=-120)

        cax = axes[f"cA{i}"]
        fig.colorbar(im, cax=cax)
        cax.set_xlabel("$\\ln q$")
        if i == 0:
            cax.set_yticks([-69, -68, -67])
        elif i == 1:
            cax.set_yticks([-70, -60, -50, -40])
        elif i == 2:
            cax.set_yticks([-60, -40, -20])

        ax = axes[f"B{i}"]

        ax.tripcolor(xs, ys, zs, cmap="rainbow", zorder=2)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_xticks([-0.5, 0.5], minor=True)
        ax.set_yticks([-0.5, 0.5], minor=True)
        ax.set_aspect("equal")
        ax.set_xlabel("$O_1$")
        if i == 0:
            ax.set_ylabel("$O_2$")
        ax.grid(which="both")

        cax = axes[f"cB{i}"]
        fig.colorbar(im, cax=cax)
        cax.set_xlabel("$\\ln q$")
        if i == 0:
            cax.set_yticks([-69, -68, -67])
        elif i == 1:
            cax.set_yticks([-70, -60, -50, -40])
        elif i == 2:
            cax.set_yticks([-60, -40, -20])

    print(out_filename)
    fig.savefig(out_filename, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
