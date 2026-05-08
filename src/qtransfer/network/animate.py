import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from IPython.display import HTML
from matplotlib.patches import FancyArrowPatch


def animateExcitationFlow(adjs, rhos, ts, sink_connections):
    n_sink = len(sink_connections)
    n_network = adjs[0].shape[0] - 1 - n_sink
    # --- graph setup ---
    G_base = nx.from_numpy_array(np.abs(adjs[0]))
    pos = circular_pos(n_network)
    center = np.mean([pos[i] for i in range(n_network)], axis=0)

    if n_sink == 1:
        sink_pos = [0]
    else:
        sink_pos = np.linspace(0.5, -0.5, n_sink)

    for n in range(n_network, n_network + n_sink):
        pos[n] = center + np.array([1.4, sink_pos[n - n_network]])
    
    pos[n_sink + n_network] = center + np.array([-1.4, 0])

    fig, (ax, ax_plot) = plt.subplots(1, 2, figsize=(10, 5), dpi=80)

    # --- time setup ---
    populations_over_time = np.array([np.diag(rho).real for rho in rhos])
    T = len(populations_over_time)
    time = np.arange(T)

    # --- plot lines ---
    lines = {}
    for node in [n for n in range(n_network, n_network + n_sink + 1)] + [-1]:
        if node == n_network + n_sink:
            label = 'Environment'
        elif node == -1:
            label = 'Network'
        else:
            label = f'Sink {node}'

        line, = ax_plot.plot([], [], label=label)
        lines[node] = line

    ax_plot.set_xlim(0, T)
    ax_plot.set_ylim(0, np.max(populations_over_time))
    ax_plot.set_xlabel("Time")
    ax_plot.set_ylabel("Population")
    ax_plot.legend()

    # --- helper: compute currents ---
    def compute_currents(H, rho):
        # J_ij = 2 Im(H_ij * rho_ji)
        return 2 * np.imag(H * rho.T)

    # --- update ---
    def update(frame):
        ax.clear()

        pops = populations_over_time[frame]
        H = adjs[frame]          # you provide this
        rho = rhos[frame]      # you provide this

        # --- draw nodes ---
        nx.draw_networkx_nodes(
            G_base, pos,
            node_size=2000 * (0.05 + pops),
            node_color=pops,
            cmap="cividis",
            ax=ax
        )

        # --- labels ---
        x, y = pos[n_network]
        ax.text(x, y + n_sink * 0.1, "Sink", fontsize=12, ha='center', color='red')

        x, y = pos[n_sink + n_network]
        ax.text(x, y + 0.2, "Env", fontsize=12, ha='center', color='red')

        # --- compute currents ---
        J = compute_currents(H, rho)

        # normalisation for linewidth
        max_flow = np.max(np.abs(J)) + 1e-12

        # --- draw arrows ---
        for i in range(len(J)):
            for j in range(len(J)):

                if i == j:
                    continue

                flow = J[i, j]

                # threshold (important for readability)
                if abs(flow) < 1e-3:
                    continue

                # direction
                if flow < 0:
                    start, end = pos[i], pos[j]
                else:
                    start, end = pos[j], pos[i]

                width = 1 + 5 * abs(flow) / max_flow
                arrow = FancyArrowPatch(
                    start, end,
                    arrowstyle='-|>',
                    mutation_scale=10,
                    linewidth=width,
                    color='blue',
                    alpha=0.7,
                    connectionstyle="arc3,rad=0.1"  # slight curve helps visibility
                )
                ax.add_patch(arrow)

        dissipative_flows = [(sink_connections[i],n_network+i, pops[sink_connections[i]]) for i in range(n_sink)]

        for (a, b, flow) in dissipative_flows:

            if flow < 1e-6:
                continue

            start, end = pos[a], pos[b]

            arrow = FancyArrowPatch(
                start, end,
                arrowstyle='-|>',
                mutation_scale=15,
                linewidth=0.1 + 20 * flow,
                color="green",          # ← distinguish from Hamiltonian
                alpha=0.8,
                connectionstyle="arc3,rad=0.2"  # curved to avoid overlap
            )
            ax.add_patch(arrow)

        ax.set_title(f"t = {np.round(ts * frame, 2)}")
        ax.set_axis_off()

        # --- update time-series ---
        for node in [n for n in range(n_network, n_network + n_sink + 1)] + [-1]:
            if node == -1:
                ydata = np.sum(populations_over_time[:frame+1, :n_network], axis=1)
            else:
                ydata = populations_over_time[:frame+1, node]

            xdata = time[:frame+1]
            lines[node].set_data(xdata, ydata)

        return list(lines.values())

    # --- animation ---
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        interval=200,
        blit=False
    )

    return HTML(ani.to_html5_video())


def animateConnections(adjs, rhos, ts, sink_connections):
    n_sink = len(sink_connections)
    n_network = adjs[0].shape[0] - 1 - n_sink
    # --- graph setup ---
    G_base = nx.from_numpy_array(np.abs(adjs[0]))
    pos = circular_pos(n_network)
    center = np.mean([pos[i] for i in range(n_network)], axis=0)

    if n_sink == 1:
        sink_pos = [0]
    else:
        sink_pos = np.linspace(0.5, -0.5, n_sink)

    for n in range(n_network, n_network + n_sink):
        pos[n] = center + np.array([1.4, sink_pos[n - n_network]])
    
    pos[n_sink + n_network] = center + np.array([-1.4, 0])

    fig, (ax, ax_plot) = plt.subplots(1, 2, figsize=(10, 5), dpi=80)

    # --- time setup ---
    populations_over_time = np.array([np.diag(rho).real for rho in rhos])
    T = len(populations_over_time)
    time = np.arange(T) * ts

    # --- plot lines ---
    lines = {}
    for node in [n for n in range(n_network, n_network + n_sink + 1)] + [-1]:
        if node == n_network + n_sink:
            label = 'Environment'
        elif node == -1:
            label = 'Network'
        else:
            label = f'Sink {node}'

        line, = ax_plot.plot([], [], label=label)
        lines[node] = line

    ax_plot.set_xlim(0, T*ts)
    ax_plot.set_ylim(0, np.max(populations_over_time))
    ax_plot.set_xlabel("Time")
    ax_plot.set_ylabel("Population")
    ax_plot.legend()

    # --- helper: compute currents ---
    def compute_currents(H, rho):
        # J_ij = 2 Im(H_ij * rho_ji)
        return 2 * np.imag(H * rho.T)

    # --- update ---
    def update(frame):
        ax.clear()

        pops = populations_over_time[frame]
        weights = np.abs(adjs[frame])

        # --- nodes ---
        nx.draw_networkx_nodes(
            G_base, pos,
            node_size=2000 * (0.05 + pops),
            node_color=pops,
            cmap="cividis",
            ax=ax
        )

        adj = adjs[frame]

        for i, n in enumerate(sink_connections):
            adj[n, n_network + i] = 1
            adj[n_network + i, n] = 1

        G = nx.from_numpy_array(np.abs(adjs[frame]))

        # extract weights
        
        weights = np.abs(adjs[frame])

        edge_widths = [
            5 * weights[i, j] for (i, j) in G.edges()
        ]

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            width=edge_widths,
            edge_color="blue",
            alpha=0.7
        )

        # --- labels ---
        x, y = pos[n_network]
        ax.text(x, y + n_sink * 0.1, "Sink", fontsize=12, ha='center', color='red')

        x, y = pos[n_sink + n_network]
        ax.text(x, y + 0.2, "Env", fontsize=12, ha='center', color='red')

        ax.set_title(f"t = {np.round(ts * frame, 2)}")
        ax.set_axis_off()

        # --- time-series ---
        for node in [n for n in range(n_network, n_network + n_sink + 1)] + [-1]:
            if node == -1:
                ydata = np.sum(populations_over_time[:frame+1, :n_network], axis=1)
            else:
                ydata = populations_over_time[:frame+1, node]

            xdata = time[:frame+1]
            lines[node].set_data(xdata, ydata)

        return list(lines.values())

    # --- animation ---
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        interval=200,
        blit=False
    )

    return HTML(ani.to_html5_video())


def animateConnectionsSquare(adjs, rhos, ts, sink_connections):
    n_sink = len(sink_connections)
    n_network = adjs[0].shape[0] - 1 - n_sink
    # --- graph setup ---
    G_base = nx.from_numpy_array(np.abs(adjs[0]))
    pos = lattice_positions(n_network)
    center = np.mean([pos[i] for i in range(n_network)], axis=0)

    if n_sink == 1:
        sink_pos = [0]
    else:
        sink_pos = np.linspace(0.5, -0.5, n_sink)

    for n in range(n_network, n_network + n_sink):
        pos[n] = center + np.array([1.4, sink_pos[n - n_network]])
    
    pos[n_sink + n_network] = center + np.array([-1.4, 0])

    fig, (ax, ax_plot) = plt.subplots(1, 2, figsize=(10, 5), dpi=80)

    # --- time setup ---
    populations_over_time = np.array([np.diag(rho).real for rho in rhos])
    T = len(populations_over_time)
    time = np.arange(T) * ts

    # --- plot lines ---
    lines = {}
    for node in [n for n in range(n_network, n_network + n_sink + 1)] + [-1]:
        if node == n_network + n_sink:
            label = 'Environment'
        elif node == -1:
            label = 'Network'
        else:
            label = f'Sink {node}'

        line, = ax_plot.plot([], [], label=label)
        lines[node] = line

    ax_plot.set_xlim(0, T*ts)
    ax_plot.set_ylim(0, np.max(populations_over_time))
    ax_plot.set_xlabel("Time")
    ax_plot.set_ylabel("Population")
    ax_plot.legend()

    # --- helper: compute currents ---
    def compute_currents(H, rho):
        # J_ij = 2 Im(H_ij * rho_ji)
        return 2 * np.imag(H * rho.T)

    # --- update ---
    def update(frame):
        ax.clear()

        pops = populations_over_time[frame]
        weights = np.abs(adjs[frame])

        # --- nodes ---
        nx.draw_networkx_nodes(
            G_base, pos,
            node_size=2000 * (0.05 + pops),
            node_color=pops,
            cmap="cividis",
            ax=ax
        )

        adj = adjs[frame]

        for i, n in enumerate(sink_connections):
            adj[n, n_network + i] = 1
            adj[n_network + i, n] = 1

        G = nx.from_numpy_array(np.abs(adjs[frame]))

        # extract weights
        
        weights = np.abs(adjs[frame])

        edge_widths = [
            5 * weights[i, j] for (i, j) in G.edges()
        ]

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            width=edge_widths,
            edge_color="blue",
            alpha=0.7
        )

        # --- labels ---
        x, y = pos[n_network]
        ax.text(x, y + n_sink * 0.1, "Sink", fontsize=12, ha='center', color='red')

        x, y = pos[n_sink + n_network]
        ax.text(x, y + 0.2, "Env", fontsize=12, ha='center', color='red')

        ax.set_title(f"t = {np.round(ts * frame, 2)}")
        ax.set_axis_off()

        # --- time-series ---
        for node in [n for n in range(n_network, n_network + n_sink + 1)] + [-1]:
            if node == -1:
                ydata = np.sum(populations_over_time[:frame+1, :n_network], axis=1)
            else:
                ydata = populations_over_time[:frame+1, node]

            xdata = time[:frame+1]
            lines[node].set_data(xdata, ydata)

        return list(lines.values())

    # --- animation ---
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        interval=200,
        blit=False
    )

    return HTML(ani.to_html5_video())


def circular_pos(n, radius=1.0, center=(0, 0)):
    pos = {}
    cx, cy = center

    for i in range(n):
        theta = 2 * np.pi * i / n
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        pos[i] = np.array([x, y])

    return pos

def lattice_positions(n, spacing=1.0, origin=(0, 0)):
    L = int(np.ceil(n**0.5))
    pos = {}
    ox, oy = origin

    for i in range(L):
        for j in range(L):
            node = i * L + j
            if node == n:
                break
            x = ox + j * spacing
            y = oy - i * spacing   # negative so it looks like a grid (top-down)
            pos[node] = (x, y)

    return pos

def symmetric_points(n):
    if n == 1:
        return np.array([0.0])
    return np.linspace(0.5, -0.5, n)