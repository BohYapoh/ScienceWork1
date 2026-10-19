import numpy as np
from qiskit_dynamics.models import HamiltonianModel
from qiskit_dynamics import Solver
from qiskit.quantum_info import Pauli

# drift и control как матрицы
omega = 2*np.pi*5.0   # частота кубита (в условных ед.)
H_drift = 0.5 * omega * Pauli("Z").to_matrix()
H_drive = 0.5 * Pauli("X").to_matrix()

model = HamiltonianModel(drift=H_drift, control=[H_drive])

solver = Solver(model)

# сигнальная функция: гаусс
def gauss(t, t0=20, amp=0.2, sigma=5):
    return amp * np.exp(-(t-t0)**2/(2*sigma**2))

t_span = np.linspace(0, 40, 1000)
signals = [gauss(t_span)]   # формат: list of arrays (n_controls, n_times)

# начальное состояние |0>
y0 = np.array([1.0, 0.0], dtype=complex)

res = solver.solve(t_span=t_span, y0=y0, signals=signals)
# expectations:
expect = res.expectation([Pauli("X"), Pauli("Y"), Pauli("Z")])