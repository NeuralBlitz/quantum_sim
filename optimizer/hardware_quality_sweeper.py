# quantum_sim/optimizer/hardware_quality_sweeper.py

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from quantum_sim.optimizer.sweet_spot_mapper import SweetSpotMapper, create_qaoa_ansatz_for_mapper


class HardwareQualitySweeper:
    """
    Orchestrates multiple SweetSpotMap runs to show how hardware 
    improvements (sweeping T1) shift the optimal circuit depth p.
    """
    def __init__(self, graph: nx.Graph, t1_range: List[float], t2_to_t1_ratio: float = 0.8,
                 depolarizing_noise_prob: float = 0.0, p_ex: float = 0.0):
        
        self.graph = graph
        self.num_qubits = len(graph.nodes)
        self.t1_range = sorted(t1_range)
        self.t2_ratio = t2_to_t1_ratio
        self.depolarizing_p = depolarizing_noise_prob
        self.p_ex = p_ex
        self.sweep_data: Dict[float, Dict[int, float]] = {}
        self.all_p_layers: List[int] = []

        if not (0 < self.t2_ratio <= 1.0):
            raise ValueError("t2_to_t1_ratio must be between 0 and 1.0 (inclusive).")


    def run_sweep(self, max_p_layers: int = 6, optimizer_maxiter: int = 100) -> Dict[float, Dict[int, float]]:
        print(f"--- Launching Hardware Quality Sweep (Max p={max_p_layers}) ---")
        
        self.all_p_layers = list(range(1, max_p_layers + 1))
        
        for t1 in self.t1_range:
            t2 = t1 * self.t2_ratio
            
            if t2 > 2 * t1:
                print(f"Warning: Calculated T2 ({t2*1e6:.1f}us) > 2*T1 ({t1*1e6:.1f}us). Adjusting T2 to 2*T1.")
                t2 = 2 * t1
            
            print(f"\n>>> Sweeping Hardware Tier: T1={t1*1e6:.1f}us, T2={t2*1e6:.1f}us (Depol_p={self.depolarizing_p})")
            
            t1_times_for_mapper = {i: t1 for i in range(self.num_qubits)}
            t2_times_for_mapper = {i: t2 for i in range(self.num_qubits)}
            
            mapper = SweetSpotMapper(
                graph=self.graph,
                num_qubits=self.num_qubits,
                t1_times=t1_times_for_mapper,
                t2_times=t2_times_for_mapper,
                p_ex=self.p_ex,
                depolarizing_noise_prob=self.depolarizing_p
            )
            
            self.sweep_data[t1] = mapper.map_sweet_spot(max_p_layers=max_p_layers, optimizer_maxiter=optimizer_maxiter)
            
        print("\n--- Hardware Quality Sweep Complete ---")
        return self.sweep_data


    def plot_sweep_results(self, filename: str = "hardware_quality_sweep.png"):
        if not self.sweep_data:
            print("No sweep data to plot. Run run_sweep first.")
            return

        plt.figure(figsize=(12, 7))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.t1_range)))
        
        sweet_spot_markers = {}
        
        for i, t1 in enumerate(self.t1_range):
            p_vals_for_t1 = sorted(self.sweep_data[t1].keys())
            energies_for_t1 = [self.sweep_data[t1][p] for p in p_vals_for_t1]
            
            t2_val = t1 * self.t2_ratio
            label = f"T1={t1*1e6:.0f}µs (T2={t2_val*1e6:.0f}µs)"
            plt.plot(p_vals_for_t1, energies_for_t1, marker='o', label=label, color=colors[i], linewidth=2)
            
            if energies_for_t1:
                best_p_idx = np.argmin(energies_for_t1)
                best_p_val = p_vals_for_t1[best_p_idx]
                min_energy_val = energies_for_t1[best_p_idx]
                
                sweet_spot_markers[t1] = (best_p_val, min_energy_val)
                plt.annotate(f'p*={best_p_val}', xy=(best_p_val, min_energy_val), 
                             textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color=colors[i],
                             arrowprops=dict(facecolor=colors[i], shrink=0.05, width=1, headwidth=5))

        plt.title("QAOA Hardware Quality Sweep: Shifting the Sweet Spot", fontsize=14)
        plt.xlabel("Circuit Depth (p)", fontsize=12)
        plt.ylabel("Min Max-Cut Hamiltonian Expectation Value (Lower is Better)", fontsize=12)
        plt.legend(title="Hardware Tiers")
        plt.grid(True, alpha=0.3)
        plt.xticks(self.all_p_layers)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        print(f"Hardware quality sweep plot saved to {filename}")
