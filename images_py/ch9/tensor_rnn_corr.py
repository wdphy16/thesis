#!/usr/bin/env python3

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb

plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"

out_filename = "./tensor_rnn_corr.pdf"


def lighten(c, w=0.25):
    c = to_rgb(c)
    c = tuple(w + (1 - w) * x for x in c)
    return c


def darken(c, w=0.25):
    c = to_rgb(c)
    c = tuple((1 - w) * x for x in c)
    return c


def plot_2d(filename, ax):
    print(filename)
    with h5py.File(filename, "r") as f:
        data = np.asarray(f["data"])

    im = ax.imshow(
        data, cmap="bwr", vmin=-1, vmax=1, origin="lower", extent=(0.5, 10.5, 0.5, 10.5)
    )
    return im


def plot_1d(filename, ax, label, color, marker, markersize):
    xs = range(1, 10)

    print(filename)
    with open(filename, "r") as f:
        data = f.read().splitlines()
    data = [float(x) for x in data]

    ax.plot(
        xs,
        data,
        label=label,
        color=color,
        linestyle="",
        marker=marker,
        markersize=markersize,
    )


def main():
    fig = plt.figure(figsize=(8.5, 2.5))
    gs = fig.add_gridspec(
        nrows=2, ncols=5, wspace=0.1, hspace=0.1, width_ratios=[1, 1, 0.1, 1.1, 3.3]
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax1)
    cax = fig.add_subplot(gs[:, 2])
    blank = fig.add_subplot(gs[:, 3])
    ax5 = fig.add_subplot(gs[:, 4])

    plot_2d("../../data/tensor_rnn/corr_afhm_1d_center.hdf5", ax1)
    plot_2d("../../data/tensor_rnn/corr_afhm_1d_left.hdf5", ax2)
    plot_2d("../../data/tensor_rnn/corr_afhm_2d_center.hdf5", ax3)
    im = plot_2d("../../data/tensor_rnn/corr_afhm_2d_left.hdf5", ax4)

    ax3.set_xlabel("$x$")
    ax4.set_xlabel("$x$")
    ax1.set_ylabel("1D\n\n$y$")
    ax3.set_ylabel("2D\n\n$y$")
    ax1.set_xticks([1, 10])
    ax1.set_yticks([1, 10])
    ax1.axes.get_xaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax4.axes.get_yaxis().set_visible(False)

    cb = fig.colorbar(im, cax=cax)
    cax.set_yticks([-1, -0.5, 0, 0.5, 1])
    for t in cb.ax.get_yticklabels():
        t.set_horizontalalignment("right")
        t.set_x(3.5)

    blank.axes.set_visible(False)

    plot_1d(
        "../../data/tensor_rnn/corr_tfim_1d_h.txt",
        ax5,
        "1D MPS-RNN, horizontal",
        lighten("C1"),
        "o",
        3,
    )
    plot_1d(
        "../../data/tensor_rnn/corr_tfim_1d_v.txt",
        ax5,
        "1D MPS-RNN, vertical",
        darken("C1"),
        "s",
        3,
    )
    plot_1d(
        "../../data/tensor_rnn/corr_tfim_2d_h.txt",
        ax5,
        "1D MPS-RNN, horizontal",
        lighten("C2"),
        "D",
        2.5,
    )
    plot_1d(
        "../../data/tensor_rnn/corr_tfim_2d_v.txt",
        ax5,
        "2D MPS-RNN, vertical",
        darken("C2"),
        "^",
        3,
    )

    ax5.legend(fontsize="small", handlelength=1, handletextpad=0.4)
    ax5.set_xlabel("$d$\n\n(b) $20 \\times 20$ TFIM")
    ax5.set_ylabel("$C(d)$")
    ax5.set_xscale("log")
    ax5.set_yscale("log")

    fig.text(0.24, -0.16, "(a) $10 \\times 10$ AFHM", horizontalalignment="center")

    print(out_filename)
    fig.savefig(out_filename, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
