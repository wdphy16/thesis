#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"


out_filename = "./arnn_sk.pdf"


def plot_sk(fig, ax):
    in_template = "../data/arnn/sk_f/{}.txt"
    betas = np.linspace(0.3, 5, 48)
    configs = [
        ("Dense", "C1", "s", 4),
        ("NMF", "C2", "x", 4),
        ("Bethe", "C0", "D", 3.5),
    ]

    in_filename = in_template.format("exact")
    print(in_filename)
    F_exact = np.loadtxt(in_filename)[:, 3]

    F_data = {}

    in_filename = in_template.format("dense")
    print(in_filename)
    F_data["Dense"] = np.loadtxt(in_filename)[:, 3]

    in_filename = in_template.format("nmf")
    print(in_filename)
    F_data["NMF"] = np.loadtxt(in_filename)[:, 3]

    in_filename = in_template.format("bethe")
    print(in_filename)
    F_data["Bethe"] = np.loadtxt(in_filename)[:, 3]

    for label, color, marker, marker_size in configs:
        _betas = betas
        _data = F_data[label]
        if label == "Bethe":
            idx = _betas < 1.6
            _betas = _betas[idx]
            _data = _data[idx]

        ax.plot(
            _betas,
            _data,
            color=color,
            label=label,
            linestyle="",
            marker=marker,
            markeredgewidth=0.5,
            markerfacecolor="none",
            markersize=marker_size,
        )

    ax.plot(betas, F_exact, color="k", label="Exact", linewidth=0.5)

    ax.set_xlabel("$\\beta$")
    ax.set_ylabel("$F / N$")
    ax.set_xlim(0.25, 1.75)
    ax.set_ylim(-1.65, -0.75)
    ax.legend(
        loc="upper left",
        fontsize="small",
        frameon=False,
        handlelength=1,
        borderaxespad=0,
    )

    ax_in = ax.inset_axes([0.5, 0.15, 0.45, 0.35])
    for label, color, _, _ in configs:
        _betas = betas
        _data = F_data[label]
        _err = (_data - F_exact) / abs(F_exact)
        if label == "Bethe":
            idx = _betas < 1.6
            _betas = _betas[idx]
            _err = _err[idx]

        ax_in.plot(_betas, _err, color=color, linewidth=0.5)

    ax_in.set_ylabel("Rel. Err.")
    ax_in.set_xlim(0, 5)
    ax_in.set_ylim(1e-7, 1)
    ax_in.set_yscale("log")
    ax_in.set_yticks([1e-7, 1e-4, 1e-1])


def plot_inv_sk(fig, ax):
    in_template = "../data/arnn/inv_sk/{}.txt"
    configs = [
        ("Dense", "dense", 3, "C1"),
        ("NMF", "mf", 3, "C2"),
        ("SM", "mf", 5, "C4"),
        ("Bethe", "mf", 6, "C0"),
    ]

    for label, file, col, color in configs:
        in_filename = in_template.format(file)
        print(in_filename)
        _data = np.loadtxt(in_filename)
        betas = _data[:, 2]
        err = _data[:, col]
        if label != "Dense":
            betas = np.concatenate([[0], betas])
            err = np.concatenate([[0], err])

        ax.plot(betas, err, color=color, label=label, linewidth=0.5)

    ax.set_xlabel("$\\beta$")
    ax.set_ylabel("Recons. Err.")
    ax.set_xlim(0.2, 1.8)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(
        loc="upper left",
        fontsize="small",
        frameon=False,
        handlelength=1,
        borderaxespad=0,
    )


def main():
    fig, axes = plt.subplots(ncols=2, figsize=(6, 2.5))

    plot_sk(fig, axes[0])
    plot_inv_sk(fig, axes[1])

    # Panel labels
    for i, ax in enumerate(axes.flatten()):
        c = chr(ord("a") + i)
        ax.text(0.05, 0.1, f"({c})", transform=ax.transAxes)

    fig.tight_layout()
    print(out_filename)
    fig.savefig(out_filename, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
