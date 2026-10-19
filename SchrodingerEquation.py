from pygments.styles.gh_dark import BLUE_2, RED_3, RED_2, GREEN_1
from qiskit_dynamics.solvers import Solver
from qiskit_dynamics.signals import Signal
from qiskit.quantum_info import Pauli
from qiskit.quantum_info.states import Statevector
import matplotlib.pyplot as plt
import random
import qutip
import numpy as np


X = Pauli("X").to_matrix()
Y = Pauli("Y").to_matrix()
Z = Pauli("Z").to_matrix()


t_ev = np.linspace(0., 1, 1000)
Freq_0 = 200.0 # Собственная частота кубита, в неких условных единицах


Y_POL_PI = np.array([[1/ np.sqrt(2), -1/ np.sqrt(2)],
           [1/ np.sqrt(2), 1/ np.sqrt(2)]])

HADAMAR = np.array([[-1j / np.sqrt(2), -1j / np.sqrt(2)],
           [-1j / np.sqrt(2), 1j / np.sqrt(2)]])


def make_Rabi_impulse(Amp, sigma, T):
    def Rabi_impulse(t):
        global sumd
        return np.where(t < T,
                        Amp,
                        0.0)
    return Rabi_impulse

def f(x):
    return 1


#Создаем случайную унитарную матрицу
def Make_random_Unitar():
    norm = np.random.randn(3)
    norm /= np.linalg.norm(norm)
    theta = np.random.uniform(0, 2 * np.pi)
    Unitar = np.zeros((2,2), dtype = complex)
    Unitar[0][0] = np.cos(theta / 2) - 1j * norm[2] * np.sin(theta / 2)
    Unitar[1][1] = np.cos(theta / 2) + 1j * norm[2] * np.sin(theta / 2)
    Unitar[0][1] = - np.sin(theta / 2) * norm[1] - 1j * norm[0] * np.sin(theta / 2)
    Unitar[1][0] = np.sin(theta / 2) * norm[1] - 1j * norm[0] * np.sin(theta / 2)

    return Unitar, norm

#Делаем разложение по базису Паули унитарной матрицы
#Чтобы в таком виде передавать в нейросеть
def Pauli_from_Unitar(Unitar):
    global X
    global Y
    global Z
    a0 = 0.5 * np.trace(Unitar)
    ax = 0.5 * np.trace(Unitar @ X)
    ay = 0.5 * np.trace(Unitar @ Y)
    az = 0.5 * np.trace(Unitar @ Z)
    return [a0.real, ax.imag, ay.imag, az.imag]

# Тут мы считаем конечное ожидаемое состояние
#просто умножая вектор на унитар
# можно сделать случайное начальное состояние
def Generate_random_state(gate, alpha = complex(random.random(), random.random()),
                          beta = complex(random.random(), random.random())):
    module_alpha = np.abs(alpha)
    module_beta = np.abs(beta)
    alpha = alpha / (np.sqrt(module_alpha ** 2 + module_beta ** 2))
    beta = beta / (np.sqrt(module_alpha ** 2 + module_beta ** 2))
    y_rand = np.array([alpha, beta])
    sv_init = Statevector(y_rand)
    y_rand = gate @ y_rand
    sv_final = Statevector(y_rand)
    bloch_sv_init = np.array([sv_init.expectation_value(X).real, sv_init.expectation_value(Y).real, sv_init.expectation_value(Z).real])
    bloch_sv_final = np.array([sv_final.expectation_value(X).real, sv_final.expectation_value(Y).real, sv_final.expectation_value(Z).real])
    return sv_init, sv_final, bloch_sv_final

#Получаем из матрицы параметры прямоугольного Раби импульса
def Calculate_impulse(gate):
    Gate = gate
    angle = 2 * np.arccos((Gate[1][1] + Gate[0][0]) / 2)
    norm = [ - (Gate[1][0] + Gate[0][1]) / (2 * np.sin(angle / 2)), (Gate[1][0] - Gate[0][1]) / (2 * np.sin(angle / 2)),
        - (Gate[0][0] - Gate [1][1]) / (2 * np.sin(angle / 2))]
    norm = np.array([norm[0].imag, norm[1].real, norm[2].imag])
    Freq_E = Freq_0 - norm[2] * angle / (2 * np.pi)
    phase = np.arctan2(norm[1],norm[0])
    Freq_Rabi = angle / (2 * np.pi) * np.sqrt(1 - (norm[2] ** 2))
    Amp =  1
    return phase, Freq_E, Freq_Rabi, norm

# Физическая симуляция
def Simulate_impulse(sv_init, t_eval, phase, Freq_E, Freq_Rabi):
    signal = Signal(envelope = 1, carrier_freq = Freq_E, phase = phase)
    signal.draw(0, 1, 1000)
    H_0 = 0.5 * 2 * np.pi * Freq_0 * Z
    H_field = 2 * np.pi * Freq_Rabi * X
    evolution = Solver( static_hamiltonian = H_0, hamiltonian_operators = [H_field], rotating_frame = (Freq_E/200)*H_0)
    result = evolution.solve(t_span = [0., 1], y0 = sv_init, signals = [signal], t_eval = t_eval,     atol=3e-14,rtol=3e-14)
    res = result
    bloch = np.zeros((len(res.y), 3))
    for i in range (len(res.y)):
        bloch[i] = [res.y[i].expectation_value(X).real, res.y[i].expectation_value(Y).real, res.y[i].expectation_value(Z).real]
    return bloch

#Рисуем график
def Visualize (bloch, norm, bloch_sv_final = np.zeros(3)):
    b = qutip.Bloch()
    bloch = bloch.T
    if bloch_sv_final[0] !=0:
        b.add_vectors(bloch_sv_final, colors=GREEN_1)
    print(bloch[:, 1])
    b.add_vectors(bloch[:, 0], colors=RED_3)
    b.add_vectors(bloch[:, 999], colors=RED_3)
    b.add_vectors(norm, colors=BLUE_2)
    b.add_points(bloch, colors=BLUE_2)
    b.show()
    plt.show()

if __name__ == "__main__":

    New_Unitar, norm = Make_random_Unitar()
    New_Unitar = HADAMAR
    print(New_Unitar)
    phase, Freq_E, Freq_Rabi, norm = Calculate_impulse(New_Unitar)
    sv_init, sv_final, bloch_sv_final = Generate_random_state(New_Unitar, 1, 0)
    bloch = Simulate_impulse(sv_init, t_ev, phase, Freq_E, Freq_Rabi)
    Visualize(bloch, norm )
