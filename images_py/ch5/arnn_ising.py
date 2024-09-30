#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"


out_filename = "./arnn_ising.pdf"


def plot_f(fig, ax):
    in_template = "../data/arnn/fm_sqr_f/{}.txt"
    betas = np.linspace(0.1, 1, 10)
    configs = [
        ("Conv", "C3", "o", 4.5),
        ("Dense", "C1", "s", 4),
        ("NMF", "C2", "x", 4),
        ("Bethe", "C0", "D", 3.5),
    ]

    in_filename = in_template.format("exact")
    print(in_filename)
    _data = np.loadtxt(in_filename)
    betas_all = _data[:, 0]
    F_exact_all = _data[:, 1]
    idx = (abs(betas_all[:, None] - betas[None, :]) < 1e-8).any(axis=1)
    F_exact = F_exact_all[idx]

    F_data = {}

    in_filename = in_template.format("nmf")
    print(in_filename)
    F_data["NMF"] = np.loadtxt(in_filename)[:, 1]

    in_filename = in_template.format("bethe")
    print(in_filename)
    F_data["Bethe"] = np.loadtxt(in_filename)[:, 1]

    in_filename = in_template.format("conv_relerr")
    print(in_filename)
    F_data["Conv"] = F_exact + abs(F_exact) * np.loadtxt(in_filename)[:, 1]

    in_filename = in_template.format("dense_relerr")
    print(in_filename)
    F_data["Dense"] = F_exact + abs(F_exact) * np.loadtxt(in_filename)[:, 1]

    for label, color, marker, marker_size in configs:
        ax.plot(
            betas,
            F_data[label],
            color=color,
            label=label,
            linestyle="",
            marker=marker,
            markeredgewidth=0.5,
            markerfacecolor="none",
            markersize=marker_size,
        )

    ax.plot(betas_all, F_exact_all, color="k", label="Exact", linewidth=0.5)

    ax.set_xlabel("$\\beta$")
    ax.set_ylabel("$F / N$")
    ax.set_xlim(0.25, 1.05)
    ax.set_ylim(-2.21, -1.99)
    ax.legend(
        loc="upper left",
        fontsize="small",
        frameon=False,
        handlelength=1,
        borderaxespad=0,
    )

    ax_in = ax.inset_axes([0.5, 0.15, 0.45, 0.35])
    for label, color, marker, marker_size in configs:
        _data = F_data[label]
        _err = (_data - F_exact) / abs(F_exact)
        ax_in.plot(
            betas,
            _err,
            color=color,
            linestyle="",
            marker=marker,
            markeredgewidth=0.5,
            markerfacecolor="none",
            markersize=marker_size,
        )

    ax_in.set_ylabel("Rel. Err.")
    ax_in.set_xlim(0, 1.1)
    ax_in.set_ylim(1e-7, 1)
    ax_in.set_yscale("log")
    ax_in.set_xticks([0, 0.5, 1])
    ax_in.set_yticks([1e-7, 1e-4, 1e-1])


def exp_const(x, a, b, c):
    return a * np.exp(-b * x) + c


def plot_s(fig, ax):
    in_template = "../data/arnn/afm_tri_s/L{L}.txt"
    Ls = [4, 6, 8, 10, 12, 14, 16]
    cmap = plt.get_cmap("turbo")
    colors = cmap(np.linspace(0.9, 0.1, len(Ls)))
    betas = np.linspace(1, 5, 9)

    S_data = {}
    for L in Ls:
        in_filename = in_template.format(L=L)
        print(in_filename)
        S_data[L] = np.loadtxt(in_filename)[:, 1]

    S_exact = 0.323
    ax.axhline(S_exact, color="0.7", linestyle="--", linewidth=0.5)

    _betas = np.linspace(1, 5, 100)
    for L, color in zip(Ls, colors):
        (a, b, c), _ = curve_fit(exp_const, betas, S_data[L])
        ax.plot(_betas, exp_const(_betas, a, b, c), color=color, linewidth=0.5)

    for L, color in zip(Ls, colors):
        ax.plot(
            betas,
            S_data[L],
            color=color,
            label=f"$L = {L}$",
            linestyle="",
            marker="o",
            markeredgewidth=0.5,
            markerfacecolor="none",
            markersize=4.5,
        )

    ax.set_xlabel("$\\beta$")
    ax.set_ylabel("$S / N$")
    ax.legend(
        loc="upper right",
        ncols=2,
        fontsize="small",
        frameon=False,
        handlelength=1,
        borderaxespad=0,
        columnspacing=1,
    )


def main():
    fig, axes = plt.subplots(ncols=2, figsize=(6, 2.5))

    plot_f(fig, axes[0])
    plot_s(fig, axes[1])

    # Panel labels
    for i, ax in enumerate(axes.flatten()):
        c = chr(ord("a") + i)
        ax.text(0.05, 0.1, f"({c})", transform=ax.transAxes)

    fig.tight_layout()
    print(out_filename)
    fig.savefig(out_filename, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
