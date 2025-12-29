# quantum_sim/optimizer/qaoa_optimizer.py

import numpy as np
import networkx as nx
from scipy.optimize import minimize
from typing import List, Callable, Any, Tuple

from quantum_sim.utils.expectation_value import ExpectationValueCalculator


class QAOAOptimizer:
    """
    Classical optimization wrapper for the QAOA ansatz.
    Finds optimal beta and gamma parameters under noise via a hybrid loop.
    """

    def __init__(self, ansatz: Any, backend: Any, graph: nx.Graph,
                 cost_op_calculator: ExpectationValueCalculator,
                 method: str = 'COBYLA', maxiter: int = 200,
                 callback: Callable[[np.ndarray], None] = None):

        self.ansatz = ansatz
        self.backend = backend
        self.graph = graph
        self.calculator = cost_op_calculator
        self.method = method
        self.maxiter = maxiter
        self.history: List[float] = []
        self.callback = callback

    def _cost_function(self, params_vector: np.ndarray) -> float:
        """
        The objective function to be minimized by the classical optimizer.
        Calculates the expectation value of the QAOA Max-Cut Hamiltonian.
        """
        param_names = sorted(list(self.ansatz.get_parameters().keys()))
        # E741 Fix: Renamed loop variable 'i' to 'idx'
        bindings = {name: params_vector[idx] for idx, name in enumerate(param_names)}

        self.ansatz.bind_parameters(bindings)

        # Execute noisy simulation on the density matrix backend
        rho_final = self.backend.run_circuit(self.ansatz)

        total_energy = self.calculator.calculate_qaoa_maxcut_energy(rho_final, self.graph)

        self.history.append(total_energy)

        if self.callback:
            self.callback(params_vector)

        return total_energy

    def optimize(self, initial_guess: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Runs the classical optimization loop to find the optimal QAOA parameters.
        Returns: (minimum_energy, optimal_parameters_array)
        """
        if not isinstance(initial_guess, np.ndarray):
            raise TypeError("Initial guess must be a NumPy array.")

        num_params = len(self.ansatz.get_parameters())
        if len(initial_guess) != num_params:
            raise ValueError(f"Initial guess length ({len(initial_guess)}) does not match "
                             f"unique parameters ({num_params}).")

        # Standard bounds for beta [0, pi] and gamma [0, 2pi]
        # Applied uniformly to all parameters here for flexibility
        bounds = [(0, 2 * np.pi) for _ in range(num_params)]

        res = minimize(
            self._cost_function,
            initial_guess,
            method=self.method,
            bounds=bounds,
            options={'maxiter': self.maxiter}
        )

        if not res.success:
            print(f"Warning: Optimizer failed to converge: {res.message}")

        return res.fun, res.x
