from func import *
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Hamiltonian of System
j=1

sigmax = Operator(Operator.pauli_x)
sigmay = Operator(Operator.pauli_y)
sigmaz = Operator(Operator.pauli_z)


a = sigmax.tensor(sigmax,Operator(Operator.identity),Operator(Operator.identity)).matrix + sigmay.tensor(sigmay,Operator(Operator.identity),Operator(Operator.identity)).matrix + sigmaz.tensor(sigmaz,Operator(Operator.identity),Operator(Operator.identity)).matrix
b = Operator(Operator.identity).tensor(sigmax,sigmax,Operator(Operator.identity)).matrix + Operator(Operator.identity).tensor(sigmay,sigmay,Operator(Operator.identity)).matrix + Operator(Operator.identity).tensor(sigmaz,sigmaz,Operator(Operator.identity)).matrix
c = Operator(Operator.identity).tensor(Operator(Operator.identity),sigmax,sigmax).matrix + Operator(Operator.identity).tensor(Operator(Operator.identity),sigmay,sigmay).matrix + Operator(Operator.identity).tensor(Operator(Operator.identity),sigmaz,sigmaz).matrix
Hs = j * (a + b + c)
# print(Hs)

spectral_decomp = Operator(Hs).spectral_decom()

arbitrary_state = spectral_decomp[int(len(spectral_decomp)/2)][1]
print(arbitrary_state)

# density matrix ρ = |ψ⟩⟨ψ|
rho = Ket(arbitrary_state).outer_product(Bra(arbitrary_state))
print(np.shape(rho))

# reduced density matrix of first two spins by tracing out the third spin and fourth spin
def partial_trace(rho, dim, subsystems_to_trace_out):
    """
    Perform partial trace over specified subsystems.

    Args:
        rho (np.ndarray): The density matrix to trace.
        dim (list): List of dimensions of subsystems.
        subsystems_to_trace_out (list): List of subsystems to trace out (0-indexed).

    Returns:
        np.ndarray: Reduced density matrix.
    """
    total_dim = np.prod(dim)
    reshaped_rho = rho.reshape(dim + dim)
    subsystems_to_keep = [i for i in range(len(dim)) if i not in subsystems_to_trace_out]
    
    # Permute dimensions to bring traced-out subsystems to the end
    permute_order = subsystems_to_keep + subsystems_to_trace_out
    permuted_rho = np.transpose(reshaped_rho, permute_order + [i + len(dim) for i in permute_order])
    
    # Trace out the subsystems
    traced_out_dim = np.prod([dim[i] for i in subsystems_to_trace_out])
    reduced_dim = int(total_dim / traced_out_dim)
    reduced_rho = np.trace(permuted_rho.reshape(reduced_dim, traced_out_dim, reduced_dim, traced_out_dim), axis1=1, axis2=3)
    
    return reduced_rho

# Define dimensions of the subsystems (4 qubits, each of dimension 2)
dim = [2, 2, 2, 2]

# Trace out subsystems 3 and 4
rho_reduced = partial_trace(rho, dim, [2, 3])
print(rho_reduced)



# Hamiltonian for system 1 and 2
H12 = j * (sigmax.tensor(sigmax).matrix + sigmay.tensor(sigmay).matrix + sigmaz.tensor(sigmaz).matrix)
print(H12)
rho_h12 = np.exp(-H12/spectral_decomp[int(len(spectral_decomp)/2)][0]) / np.trace(np.exp(-H12/spectral_decomp[int(len(spectral_decomp)/2)][0]))
print(rho_h12)

# Time evolution of the rho_reduced and rho_h12
t = np.linspace(0, 60, 1000)
rho_reduced_t = [expm(-1j * H12 * time) @ rho_reduced @ expm(1j * H12 * time) for time in t]
rho_h12_t = [expm(-1j * H12 * time) @ rho_h12 @ expm(1j * H12 * time) for time in t]






