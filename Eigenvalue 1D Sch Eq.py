#to find eigenvalue energy 
# assignment (due date christmast)
# @umimaiii


import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0  # Planck's constant
alpha = 1.0
lam = 4
m = 1.0     # Mass of the particle
dx = 0.01   # Step size
x_max = 5.0 # Boundary for x
x = np.arange(-x_max, x_max + dx, dx)  # Spatial grid

# Define the potential
def potential(x):
    return (hbar**2 / (2 * m)) * alpha**2 * lam * (lam - 1) * (1 / 2 - 1 / np.cosh(alpha * x)**2)

V = potential(x)  # Compute potential over spatial domain

# Numerov algorithm
def numerov(E, x, V):
    psi = np.zeros_like(x)  # Initialize wavefunction
    dx2 = dx**2             # Square of step size
    k = 2 * m * (E - V) / hbar**2  # Effective potential term
    
    # Initial conditions
    psi[0] = 0.0           # Wavefunction at the left boundary
    psi[1] = 1e-5          # Small nonzero value to start the integration
    
    for i in range(1, len(x) - 1):
        psi[i + 1] = (2 * (1 - 5/12 * dx2 * k[i]) * psi[i] - (1 + 1/12 * dx2 * k[i-1]) * psi[i-1]) / (1 + 1/12 * dx2 * k[i+1])
    
    return psi

# Shooting method: check if boundary conditions are satisfied
def boundary_condition(E):
    psi = numerov(E, x, V)  # Solve for given energy E
    return psi[-1]          # Return value of wavefunction at the right boundary

# Manual bisection method to find roots (eigenvalues)
def bisection_method(f, E_min, E_max, tol=1e-6, max_iter=100):
    for _ in range(max_iter):
        E_mid = (E_min + E_max) / 2.0
        f_min = f(E_min)
        f_mid = f(E_mid)

        # Check for convergence
        if abs(f_mid) < tol or abs(E_max - E_min) < tol:
            return E_mid
        
        # Update bounds
        if f_min * f_mid < 0:
            E_max = E_mid
        else:
            E_min = E_mid
    
    raise ValueError("Bisection method did not converge.")

# Find eigenvalues
def find_eigenvalues(E_range, tol=1e-6):
    eigenvalues = []
    for i in range(len(E_range) - 1):
        if boundary_condition(E_range[i]) * boundary_condition(E_range[i+1]) < 0:
            E_root = bisection_method(boundary_condition, E_range[i], E_range[i+1], tol=tol)
            eigenvalues.append(E_root)
    return eigenvalues

# Energy range
E_range = np.linspace(-3,3,1000)  # Reasonable range for harmonic oscillator

# Find eigenvalues
eigenvalues = find_eigenvalues(E_range)
print("Eigenvalues (Energy levels):", eigenvalues)

# Plot potential and wavefunctions
plt.figure(figsize=(10, 6))
plt.plot(x, V, label="Potential V(x)")

for i, E in enumerate(eigenvalues[:3]):  # Plot first 3 eigenfunctions
    psi = numerov(E, x, V)
    psi_norm = psi / np.max(np.abs(psi))  # Normalize the wavefunction
    plt.plot(x, psi_norm + E, label=f"Wavefunction for E={E:.2f}")

plt.title("Potential and Wavefunctions for Schrodinger Equation ")
plt.xlabel("x")
plt.ylabel("$\psi(x)$ & $V(x)$")
plt.legend()
plt.grid()
plt.show()



