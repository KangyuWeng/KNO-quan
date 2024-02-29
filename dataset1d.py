import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from FyeldGenerator import generate_field

device = "cpu"
dtype = np.complex128

# Grid size
Nt = 256
Nx = 256

# Subsampling
x_step = 1
t_step = 2*4*512

# Steps
dx = 1/(Nx*x_step - 1)
dt = 1e-3 * dx**2

# Length scale of the Gaussian fluctuations
lscale = 0.1

# Gaussian power spectrum
def Gaussian(l):
    def Pk(p):
        return (l/np.sqrt(2*np.pi)) * np.exp(-l**2 * p**2/2)
    return Pk

def distrib(shape):
    a = np.random.normal(loc=0, scale=1, size=shape)
    b = np.random.normal(loc=0, scale=1, size=shape)
    return a + 1j * b

# Sample random gaussian field
def randomGaussian(Nxx, length_scale = lscale):
    return generate_field(distrib, Gaussian(Nxx*length_scale), (Nxx,))

# Compute the evolution of the initial wave function psi0 with the potential Vp for a given number of steps
def computeSK(psi0, Vp, steps = Nt):
    
    I = np.eye(Nx*x_step)
    Isub = np.roll(I,1,axis=0) + np.roll(I,-1,axis=0)

    A = torch.tensor((1j/dt - 1/(2*dx**2)) * I - 3e3 * np.diagflat(Vp)/2 + 1/(4*dx**2) * Isub, device=device)
    B = torch.tensor((1j/dt + 1/(2*dx**2)) * I + 3e3 * np.diagflat(Vp)/2 - 1/(4*dx**2) * Isub, device=device)

    C = torch.linalg.solve(A, B)
    Cp = torch.linalg.matrix_power(C, t_step)

    psi = np.zeros([steps, Nx*x_step], dtype=np.csingle)
    psi[0] = psi0

    psi = torch.tensor(np.zeros([steps, Nx*x_step], dtype=dtype), device=device)
    psi[0] = torch.tensor(psi0)

    for t in range(steps-1):
        psi[t+1] = torch.matmul(Cp, psi[t])
        norm = torch.sum(torch.abs(psi[t+1])**2) * dx
        psi[t+1] /= torch.sqrt(norm)
        
    psi = psi.cpu().numpy()
    return psi

# Solve the Schrodinger equation with a given potential and boundary conditions
def solveSchrodinger(bc = "random", potential = "random", steps = Nt):

    x = np.linspace(0, 1, Nx*x_step)

    if potential == "random":
        V = 10*randomGaussian(Nx*x_step)
    elif potential == "central":
        mu, sigma = 1/2, 1/10
        V = 30*randomGaussian(Nx*x_step, length_scale = 1) * np.exp(-(x-mu)**2/(2*sigma**2))
    elif potential == "bump":
        V =  (-np.tanh(10*np.pi*(x-0.425)) - np.tanh(-10*np.pi*(x-0.575)))/2
    elif potential == "nbump":
        V =  -(-np.tanh(10*np.pi*(x-0.425)) - np.tanh(-10*np.pi*(x-0.575)))/2
    
    normal = np.sum(np.abs(V)**2) * dx
    V /= np.sqrt(normal)
    
    if bc == "sin":
        psi0 = np.sqrt(2)*np.sin(2*np.pi*x)
    if bc == "random":
        psi0 = randomGaussian(Nx*x_step) + 1j*randomGaussian(Nx*x_step)
    elif bc == "wavepacket":
        sigma = 0.06
        p = 50
        x0 = 0.25
        psi0 = (1/(sigma*(2*np.pi)**0.5))**0.5 * np.exp(-0.25*((x-x0)/sigma)**2) * np.exp(1j*p*x) + (1/(sigma*(2*np.pi)**0.5))**0.5 * np.exp(-0.25*((x-x0-1)/sigma)**2) * np.exp(1j*p*(x-1))
        
    norm = np.sum(np.abs(psi0)**2) * dx
    psi0 /= np.sqrt(norm)
        
    psi_m2 = computeSK(psi0, V, steps = steps)
    return V[::x_step], psi_m2[:,::x_step]


# Example usage
V, sol = solveSchrodinger(bc="wavepacket", potential="bump") # out of distribution

for T in [0, 100]:
    plt.plot(np.real(sol[T]))
    plt.plot(np.imag(sol[T]))
    plt.plot(V)
    plt.title(f"T={T}")
    plt.savefig("example.png")

# Generate the data
from torch.utils.data import TensorDataset, DataLoader

batch_size = 32

train_samples = 4*8
test_samples = 1

class WavefunctionDataset(TensorDataset):
    def __init__(self, samples, bc, potential, sample_times = Nt-1):
        
        # The total number of samples is samples*sample_times
        
        # input: (x, V, Re(psi_in), Im(psi_in))
        # ouput: (Re(psi_out), Im(psi_out))
        
        self.X = torch.Tensor(np.zeros([int(samples*sample_times), 4, Nx]))
        self.Y = torch.Tensor(np.zeros([int(samples*sample_times), 2, Nx]))
        
        idx = 0
        
        for s in np.arange(samples):
            
            # Solution of the Schrodinger equation
            V, sol = solveSchrodinger(bc=bc, potential=potential, steps=sample_times+1)
            
            # Random sample times
            ts = np.arange(sample_times)
            
            for t in ts:
                
                self.X[idx][0] = torch.Tensor(np.linspace(0,1,num=Nx))
                self.X[idx][1] = torch.Tensor(V)
                self.X[idx][2] = torch.Tensor(np.real(sol[t]))
                self.X[idx][3] = torch.Tensor(np.imag(sol[t]))
                
                self.Y[idx][0] = torch.Tensor(np.real(sol[t+1]))
                self.Y[idx][1] = torch.Tensor(np.imag(sol[t+1]))
                
                idx += 1
    
    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return {'x': x, 'y': y}
    
    def __len__(self):
        return len(self.X)

train_loader = DataLoader(WavefunctionDataset(samples=train_samples, bc="random", potential="random", sample_times=(Nt-1)), batch_size=batch_size, shuffle=True)
test_loader = {"central": DataLoader(WavefunctionDataset(samples=test_samples, bc="wavepacket", potential="central"), batch_size=batch_size, shuffle=True),
               "bump": DataLoader(WavefunctionDataset(samples=1, bc="wavepacket", potential="bump"), batch_size=batch_size, shuffle=True),
               "nbump": DataLoader(WavefunctionDataset(samples=1, bc="wavepacket", potential="nbump"), batch_size=batch_size, shuffle=True)}

