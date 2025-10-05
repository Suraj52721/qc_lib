from func import *
import matplotlib.pyplot as plt
import scipy.linalg as la

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
n = 15
a = np.zeros((n, n))
for i in range(n):
    a[i][i] = i +(1/2)
Hb = np.array(a)
print(Hb)

# interaction Hamiltonian Hint
g = 0.5
Hint = g * sigmaz.tensor(Operator(Operator.identity),Operator(Operator.identity),Operator(Operator.identity),Operator(Hb)).matrix
print(Hint)

# Total Hamiltonian H
H = Operator(Hs).tensor(Operator(np.identity(n))).matrix + Operator(np.identity(16)).tensor(Operator(Hb)).matrix + Hint
print(H)



# (1) System Néel state |↑↓↑↓>
up = Ket([1, 0])
down = Ket([0, 1])
psi_s = up.tensor(down, up, down)
print(psi_s)

rho0_s = psi_s.outer_product(psi_s.dagger())
print(rho0_s)

# bath begins in a thermal Gibbs state at a finite temperature T, defined by the density matrix:

T = 2
rho0_b = np.exp(-Hb / T) / np.trace(np.exp(-Hb / T))
print(rho0_b)

#  Compute the time evolution of the total density matrix using the Schrödinger-von Neumann equation:
rho0 = Operator(rho0_s).tensor(Operator(rho0_b))

# 1. Magnetization Decay
t=np.linspace(0.1,60,1000)
mz1, mz2, mz3, mz4 = [], [], [], []
for time in t:
    U = la.expm(-1j * H * time)
    rho_t = U @ rho0.matrix @ Operator(U).dagger().matrix

    def partial_trace(rho, dims, subsystem):
        dims = np.array(dims)
        total_dim = np.prod(dims)
        assert rho.shape == (total_dim, total_dim), "rho shape mismatch"

        # Reshape into 4D tensor: (A,B,A',B')
        rho = rho.reshape(dims[0], dims[1], dims[0], dims[1])

        if subsystem == 0:
            # trace out subsystem A
            return np.trace(rho, axis1=0, axis2=2)
        elif subsystem == 1:
            # trace out subsystem B
            return np.trace(rho, axis1=1, axis2=3)
        else:
            raise ValueError("subsystem must be 0 or 1")
    rho_s_t = Operator(partial_trace(rho_t, [16, n], 1))
    print(rho_s_t)
    mz1.append(np.trace(rho_s_t.matrix @ sigmaz.tensor(Operator(Operator.identity),Operator(Operator.identity),Operator(Operator.identity)).matrix).real)
    mz2.append(np.trace(rho_s_t.matrix @ Operator(Operator.identity).tensor(sigmaz,Operator(Operator.identity),Operator(Operator.identity)).matrix).real)
    mz3.append(np.trace(rho_s_t.matrix @ Operator(Operator.identity).tensor(Operator(Operator.identity),sigmaz,Operator(Operator.identity)).matrix).real)
    mz4.append(np.trace(rho_s_t.matrix @ Operator(Operator.identity).tensor(Operator(Operator.identity),Operator(Operator.identity),sigmaz).matrix).real)

plt.plot(t, mz1, label='Site 1')
plt.xlabel('Time')
plt.ylabel('Magnetization')
plt.title('Magnetization Decay')
plt.grid()
plt.legend()
plt.show()

# 2. Entropy Growth
entropy = []
for time in t:
    U = la.expm(-1j * H * time)
    rho_t = U @ rho0.matrix @ Operator(U).dagger().matrix
    rho_s_t = Operator(partial_trace(rho_t, [16, n], 1))
    print(rho_s_t)
    def von_neumann_entropy(rho):
        eigenvalues = np.linalg.eigvalsh(rho.matrix)
        eigenvalues = eigenvalues[eigenvalues > 0] 
        return -np.sum(eigenvalues * np.log(eigenvalues))
    
    entropy.append(von_neumann_entropy(rho_s_t))
plt.plot(t, entropy)
plt.xlabel('Time')
plt.ylabel('Von Neumann Entropy')
plt.grid()
plt.show()

#Create a dynamic bar-chart animation showing the evolution of the site magneti-zations ⟨σ iz (t)⟩.

import matplotlib.animation as animation
fig, ax = plt.subplots()
bars = ax.bar(['Site 1', 'Site 2', 'Site 3', 'Site 4'], [0, 0, 0, 0], color=['blue', 'green', 'orange', 'red'])
ax.set_ylim(-1, 1)
def update(frame):
    time = t[frame]
    U = la.expm(-1j * H * time)
    rho_t = U @ rho0.matrix @ Operator(U).dagger().matrix
    rho_s_t = Operator(partial_trace(rho_t, [16, n], 1))
    mz_values = [
        np.trace(rho_s_t.matrix @ sigmaz.tensor(Operator(Operator.identity),Operator(Operator.identity),Operator(Operator.identity)).matrix).real,
        np.trace(rho_s_t.matrix @ Operator(Operator.identity).tensor(sigmaz,Operator(Operator.identity),Operator(Operator.identity)).matrix).real,
        np.trace(rho_s_t.matrix @ Operator(Operator.identity).tensor(Operator(Operator.identity),sigmaz,Operator(Operator.identity)).matrix).real,
        np.trace(rho_s_t.matrix @ Operator(Operator.identity).tensor(Operator(Operator.identity),Operator(Operator.identity),sigmaz).matrix).real
    ]
    for bar, height in zip(bars, mz_values):
        bar.set_height(height)
    ax.set_title(f'Time: {time:.2f}')
    return bars
ani = animation.FuncAnimation(fig, update, frames=len(t), blit=False, repeat=False)
plt.show()
