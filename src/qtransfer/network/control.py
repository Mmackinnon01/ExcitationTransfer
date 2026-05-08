from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qgene.individual import Individual


@dataclass
class ControlSequence:
    times: list[float]
    operators: list[np.ndarray]

    def to_individual(self, control_mask: np.ndarray | None=None) -> Individual:
        if control_mask is None:
            return Individual(genome = np.concatenate([op[np.tril_indices(op.shape[0])] for op in self.operators]).flatten())
        else:
            return Individual(genome = np.concatenate([op[control_mask] for op in self.operators]).flatten())

    @classmethod
    def from_individual(cls, control_times: list[float], indiv: Individual, n_nodes: int, control_mask: np.ndarray | None=None) -> ControlSequence:
        n_elements = int(len(indiv.genome)/len(control_times))
        chunks = [indiv.genome[i * n_elements : (i+1) * n_elements] for i in range(len(control_times))]
        
        if control_mask is None:
            control_mask = (np.zeros(shape = (n_nodes,n_nodes)) == 1)
            control_mask[np.tril_indices(n_nodes)] = True
        
        ops = []

        for chunk in chunks:
            new_op = np.zeros(shape=(n_nodes,n_nodes), dtype=np.complex128)
            new_op[control_mask] = chunk
            i_lower, j_lower = np.tril_indices(n_nodes, k=-1)
            new_op[j_lower, i_lower] = new_op[i_lower, j_lower]
            ops.append(new_op)

        return cls(control_times, ops)