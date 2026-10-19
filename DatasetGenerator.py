import numpy as np
import SchrodingerEquation as Sch

input_arr = np.zeros((5000000,4), dtype=complex )
output_arr = np.zeros((5000000,3), dtype=complex)
New_Unitar = Sch.Make_random_Unitar()
for i in range(5000000):
    if i % 100000 == 0:
        print(i)
    New_Unitar, norm = Sch.Make_random_Unitar()
    Unitar_Pauli = Sch.Pauli_from_Unitar(New_Unitar)
    phase, Freq_E, Freq_Rabi, norm  = Sch.Calculate_impulse(New_Unitar)
    Freq_E = Freq_E.real
    Freq_Rabi = Freq_Rabi.real
    output_arr[i] = [phase, Freq_E, Freq_Rabi]
    input_arr[i] = Unitar_Pauli
print (input_arr)

np.save("input.npy", input_arr)
np.save("output.npy", output_arr)