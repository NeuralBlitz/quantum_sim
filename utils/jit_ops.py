# quantum_sim/visualization/circuit_drawer.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from quantum_sim.core.circuit import QuantumCircuit

class CircuitDrawer:
    """
    Utility class to visualize the quantum circuit topology using Matplotlib.
    """
    def draw(self, circuit: "QuantumCircuit", filename: str = None):
        circuit_info_initial_pass = circuit.get_visualization_info(offset_x=0.5, offset_y=0, qubit_y_coords={i: -i * 0.8 for i in range(circuit.num_qubits)})
        max_x_for_figsize = 5.0
        if circuit_info_initial_pass:
            max_x_info = [item.get('x', -np.inf) for item in circuit_info_initial_pass]
            max_x_info.extend([item.get('x_end', -np.inf) for item in circuit_info_initial_pass])
            if max_x_info:
                max_x_for_figsize = max(max_x_info)
        
        fig, ax = plt.subplots(figsize=(max_x_for_figsize * 1.0 + 1, circuit.num_qubits * 0.8 + 1))
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim(-1 * circuit.num_qubits * 0.8 - 0.5, 0.5)
        ax.set_xlim(-0.5, max_x_for_figsize + 1)

        qubit_y_coords = {i: -i * 0.8 for i in range(circuit.num_qubits)}
        for i in range(circuit.num_qubits):
            ax.axhline(y=qubit_y_coords[i], color='gray', linestyle='-', linewidth=0.8)
            ax.text(-0.3, qubit_y_coords[i], f'q[{i}]', va='center', ha='right', fontsize=9)

        circuit_info = circuit.get_visualization_info(offset_x=0.5, offset_y=0, qubit_y_coords=qubit_y_coords)

        for item in circuit_info:
            if item['type'] == 'gate':
                gate_name = item['name']
                x_pos = item['x']
                num_qubits = item['num_qubits']
                params_str = ""
                if 'params' in item and item['params']:
                    params_str = ", ".join(f"{k}={v:.2f}" for k,v in item['params'].items())

                if gate_name == "CNOT" and num_qubits == 2:
                    control_y = item['control_y']
                    target_y = item['target_y']
                    
                    ax.plot([x_pos, x_pos], [control_y, target_y], color='black', linewidth=1)
                    ax.add_patch(patches.Circle((x_pos, control_y), 0.15, facecolor='black', edgecolor='black', zorder=2))
                    ax.add_patch(patches.Circle((x_pos, target_y), 0.25, facecolor='white', edgecolor='black', zorder=2))
                    ax.text(x_pos, target_y, '⊕', va='center', ha='center', fontsize=12, color='black', zorder=3)
                    if params_str:
                         ax.text(x_pos, min(control_y, target_y) - 0.3, f"({params_str})", va='top', ha='center', fontsize=7)
                
                elif num_qubits == 1:
                    y_pos_center = item['y']
                    rect = patches.Rectangle((x_pos - 0.25, y_pos_center - 0.25), 0.5, 0.5, facecolor='aliceblue', edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
                    ax.text(x_pos, y_pos_center, gate_name, va='center', ha='center', fontsize=8)
                    if params_str:
                         ax.text(x_pos, y_pos_center - 0.35, f"({params_str})", va='top', ha='center', fontsize=7)
                
                elif item['type'] == 'multi_qubit_gate_box':
                    y_min_gate = item['y_min']
                    y_max_gate = item['y_max']
                    rect = patches.Rectangle((x_pos - 0.3, y_min_gate), 0.6, y_max_gate - y_min_gate,
                                             facecolor='lightgray', edgecolor='black', linewidth=1)
                    ax.add_patch(rect)
                    ax.text(x_pos, (y_min_gate + y_max_gate) / 2, item['name'], va='center', ha='center', fontsize=8, rotation=90)
                    if params_str:
                         ax.text(x_pos, y_min_gate - 0.1, f"({params_str})", va='top', ha='center', fontsize=7, rotation=90)


            elif item['type'] == 'sub_circuit_box' or item['type'] == 'block_box' or item['type'] == 'layer_box':
                rect_color = 'lightgoldenrodyellow' if item['type'] == 'sub_circuit_box' else ('lightsalmon' if item['type'] == 'layer_box' else 'lightblue')
                edge_color = 'orange' if item['type'] == 'sub_circuit_box' else ('red' if item['type'] == 'layer_box' else 'blue')

                rect = patches.Rectangle((item['x_start'], item['y_min']),
                                         item['x_end'] - item['x_start'],
                                         item['y_max'] - item['y_min'],
                                         facecolor=rect_color, edgecolor=edge_color, linewidth=1, linestyle='--', alpha=0.5)
                ax.add_patch(rect)
                ax.text(item['offset_x'], item['y_max'] + 0.1, item['name'], va='bottom', ha='left', fontsize=8, color=edge_color)


        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Circuit: {circuit.name}")
        ax.axis('off')

        if filename:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
