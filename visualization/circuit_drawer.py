# # quantum_sim/visualization/circuit_drawer.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from quantum_sim.core.circuit import QuantumCircuit


class CircuitDrawer:
    """
    Utility class to visualize the quantum circuit topology using Matplotlib.
    Renders gates, CNOT connections, and composite blocks (layers).
    """

    def draw(self, circuit: "QuantumCircuit", filename: str = None):
        # Initial pass to determine required figure width
        initial_coords = {idx: -idx * 0.8 for idx in range(circuit.num_qubits)}
        info_pass = circuit.get_visualization_info(offset_x=0.5, offset_y=0, qubit_y_coords=initial_coords)

        max_x = 5.0
        if info_pass:
            x_coords = [item.get('x', 0) for item in info_pass]
            x_ends = [item.get('x_end', 0) for item in info_pass]
            max_x = max(max(x_coords, default=0), max(x_ends, default=0))

        fig, ax = plt.subplots(figsize=(max_x + 1, circuit.num_qubits * 0.8 + 1))
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim(-1 * circuit.num_qubits * 0.8 - 0.5, 0.5)
        ax.set_xlim(-0.5, max_x + 1)

        qubit_y = {idx: -idx * 0.8 for idx in range(circuit.num_qubits)}
        for idx in range(circuit.num_qubits):
            ax.axhline(y=qubit_y[idx], color='gray', linestyle='-', linewidth=0.8)
            ax.text(-0.3, qubit_y[idx], f'q[{idx}]', va='center', ha='right', fontsize=9)

        circuit_info = circuit.get_visualization_info(offset_x=0.5, offset_y=0, qubit_y_coords=qubit_y)

        for item in circuit_info:
            if item['type'] == 'gate':
                name = item['name']
                x = item['x']
                n_q = item['num_qubits']
                params = item.get('params', {})
                p_str = ", ".join(f"{k}={v:.2f}" for k, v in params.items()) if params else ""

                if name == "CNOT" and n_q == 2:
                    c_y, t_y = item['control_y'], item['target_y']
                    ax.plot([x, x], [c_y, t_y], color='black', linewidth=1)
                    ax.add_patch(patches.Circle((x, c_y), 0.15, color='black', zorder=2))
                    ax.add_patch(patches.Circle((x, t_y), 0.25, fc='white', ec='black', zorder=2))
                    ax.text(x, t_y, '⊕', va='center', ha='center', fontsize=12, zorder=3)
                    if p_str:
                        ax.text(x, min(c_y, t_y) - 0.3, f"({p_str})", va='top', ha='center', fontsize=7)

                elif n_q == 1:
                    y = item['y']
                    rect = patches.Rectangle((x - 0.25, y - 0.25), 0.5, 0.5, fc='aliceblue', ec='black', lw=1)
                    ax.add_patch(rect)
                    ax.text(x, y, name, va='center', ha='center', fontsize=8)
                    if p_str:
                        ax.text(x, y - 0.35, f"({p_str})", va='top', ha='center', fontsize=7)

            elif item['type'] in ['sub_circuit_box', 'block_box', 'layer_box']:
                colors = {
                    'sub_circuit_box': ('lightgoldenrodyellow', 'orange'),
                    'layer_box': ('lightsalmon', 'red'),
                    'block_box': ('lightblue', 'blue')
                }
                fc, ec = colors.get(item['type'])
                rect = patches.Rectangle(
                    (item['x_start'], item['y_min']), item['x_end'] - item['x_start'],
                    item['y_max'] - item['y_min'], facecolor=fc, edgecolor=ec,
                    linewidth=1, linestyle='--', alpha=0.5
                )
                ax.add_patch(rect)
                ax.text(item['offset_x'], item['y_max'] + 0.1, item['name'], va='bottom', fontsize=8, color=ec)

        ax.axis('off')
        ax.set_title(f"Circuit: {circuit.name}")

        if filename:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
quantum_sim/visualization/__init__.py

# This file marks 'visualization' as a Python package.
