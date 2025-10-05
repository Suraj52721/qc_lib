from func import *

# Hamiltonian of System
j=1

sigmax = Operator(Operator.pauli_x)
sigmay = Operator(Operator.pauli_y)
sigmaz = Operator(Operator.pauli_z)


a = sigmax.tensor(sigmax,Operator(Operator.identity),Operator(Operator.identity)).matrix + sigmay.tensor(sigmay,Operator(Operator.identity),Operator(Operator.identity)).matrix + sigmaz.tensor(sigmaz,Operator(Operator.identity),Operator(Operator.identity)).matrix
b = Operator(Operator.identity).tensor(sigmax,sigmax,Operator(Operator.identity)).matrix + Operator(Operator.identity).tensor(sigmay,sigmay,Operator(Operator.identity)).matrix + Operator(Operator.identity).tensor(sigmaz,sigmaz,Operator(Operator.identity)).matrix
c = Operator(Operator.identity).tensor(Operator(Operator.identity),sigmax,sigmax).matrix + Operator(Operator.identity).tensor(Operator(Operator.identity),sigmay,sigmay).matrix + Operator(Operator.identity).tensor(Operator(Operator.identity),sigmaz,sigmaz).matrix
Hs = j * (a + b + c)
print(Hs)

#  A truncated Quantum Harmonic Oscillator with Hamiltonian Hb
n = 2
a = np.zeros((n, n))
for i in range(1, n):
    a[i-1][i] = np.sqrt(i)
    a[i][i-1] = np.sqrt(i)
ad = a.T
Hb = ad @ a + 0.5 * np.identity(n)
print(Hb)

# interaction Hamiltonian Hint
g = 0.5
Hint = g * sigmaz.tensor(Operator(Operator.identity),Operator(Operator.identity),Operator(Operator.identity),Operator(Hb)).matrix
print(Hint)

# Total Hamiltonian H
H = Operator(Hs).tensor(Operator(np.identity(n))).matrix + Operator(np.identity(16)).tensor(Operator(Hb)).matrix + Hint
print(H)


