#!/usr/bin/env python3

import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"

out_filename = "./twobo_grid_2d.pdf"
L = 5
node_i = 12


def main():
    G = nx.grid_2d_graph(L, L)

    highlighted_edges = [
        (
            (int((node_i - k) / L), (node_i - k) % L),
            (int((node_i - k) / L) + 1, (node_i - k) % L),
        )
        for k in range(1, L + 1)
    ]
    highlighted_edges.append(
        ((int((node_i - 1) / L), (node_i - 1) % L), (int((node_i) / L), (node_i) % L))
    )

    labels = {(i, j): f"{i * L + j + 1}" for i in range(L) for j in range(L)}

    max_i = max(i for i, j in G.nodes())
    pos = {(i, j): (j, max_i - i) for (i, j) in G.nodes()}

    target_node = (int(node_i / L), node_i % L)
    node_colors = [
        "#2ca02c" if node == target_node else "#ff7f0e" for node in G.nodes()
    ]

    plt.figure(figsize=(4, 4))

    nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=1000)
    nx.draw_networkx_edges(
        G, pos, edgelist=highlighted_edges, edge_color="#1f77b4", width=5.5
    )
    nx.draw_networkx_labels(
        G, pos, labels=labels, font_family="serif", verticalalignment="center_baseline"
    )

    print(out_filename)
    plt.savefig(out_filename, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
