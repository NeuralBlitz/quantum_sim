# quantum_sim/core/parameter.py

import numpy as np


class Parameter:
    """
    Represents a symbolic parameter that can be bound to a numerical value later.
    Used for parametric gates (e.g., RX(theta)).
    """

    def __init__(self, name: str):
        if not isinstance(name, str) or not name:
            raise ValueError("Parameter name must be a non-empty string.")
        self.name = name
        self._value = None  # Initially unbound

    def bind(self, value: float):
        """Binds a numerical value to this parameter."""
        if not isinstance(value, (int, float, np.floating)):
            raise TypeError(f"Parameter value must be a number, got {type(value)}.")
        self._value = float(value)  # Ensure float type

    def get_value(self) -> float:
        """Returns the bound numerical value, raising an error if unbound."""
        if self._value is None:
            raise ValueError(f"Parameter '{self.name}' is not bound to a value.")
        return self._value

    def is_bound(self) -> bool:
        """Checks if the parameter has been assigned a numerical value."""
        return self._value is not None

    def __repr__(self) -> str:
        status = self._value if self._value is not None else 'unbound'
        return f"Parameter('{self.name}', value={status})"

    def __hash__(self) -> int:
        """Hash based on name to allow unique parameter tracking in circuits."""
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Two parameters are considered equal if they share the same name."""
        if not isinstance(other, Parameter):
            return NotImplemented
        return self.name == other.name
