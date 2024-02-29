
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from FyeldGenerator import generate_field

device = "cpu"

###########################################################
# Functions solving the 2D Schrodinger equations
###########################################################

# Grid size
Nt = 256
Nx = 64
Ny = Nx

# Subsampling
x_step = 1
y_step = x_step
t_step = 2*128

# Steps
dx = 1/(Nx*x_step - 1)
dy = 1/(Ny*y_step - 1)
dt = 1e-3 * (dx**2 + dy**2)/2 

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

# Sample random gaussian field in 2D
def randomGaussian2D(length_scale = lscale):
    if length_scale == None:
        length_scale = 0.05 + 0.95*np.random.rand() # pick random length_scale if not provided
    return generate_field(distrib, Gaussian(Nx*length_scale), (Nx,Ny))

# Compute the evolution of the initial wave function psi0 with the potential Vp for a given number of steps in 2D
def computeSK2D(psi0, V, steps = Nt):
    
    A, B = np.zeros([2, Nx, Ny, Nx, Ny], dtype=np.csingle)

    for i in range(Nx):
        for j in range(Ny):
        
            A[i, j, i, j] = 1j/dt - 1/(2*dx**2) - 1/(2*dy**2) - 3e3 * V[i,j] /2 # Note the potential is rescaled here
            A[(i-1) % Nx, j, i, j] += 1/(4*dx**2) # Note we impose periodic boundary conditions
            A[(i+1) % Nx, j, i, j] += 1/(4*dx**2)
            A[i, (j-1) % Ny, i, j] += 1/(4*dy**2)
            A[i, (j+1) % Ny, i, j] += 1/(4*dy**2)
            
            B[i, j, i, j] = 1j/dt + 1/(2*dx**2) + 1/(2*dy**2) + 3e3 * V[i,j] /2
            B[(i-1) % Nx, j, i, j] -= 1/(4*dx**2)
            B[(i+1) % Nx, j, i, j] -= 1/(4*dx**2)
            B[i, (j-1) % Ny, i, j] -= 1/(4*dy**2)
            B[i, (j+1) % Ny, i, j] -= 1/(4*dy**2)
            
    A = torch.tensor(A.reshape([Nx*Ny, Nx*Ny]), device=device)
    B = torch.tensor(B.reshape([Nx*Ny, Nx*Ny]), device=device)

    C = torch.linalg.solve(A, B)
    del A, B

    Cp = torch.linalg.matrix_power(C, t_step)
    del C

    psi = torch.tensor(np.zeros([steps, Nx*Ny], dtype=np.csingle), device=device)
    psi[0] = torch.tensor(psi0.reshape([Nx*Ny]), device=device)

    for t in range(steps-1):
        psi[t+1] = torch.matmul(Cp, psi[t])
        normal = torch.sum(torch.abs(psi[t+1])**2)*dx*dy
        psi[t+1] /= torch.sqrt(normal)
        
    psi = psi.cpu().numpy().reshape([steps, Nx, Ny])
    
    return psi

def solveSchrodinger2D(bc = "random", potential = "random", steps = Nt):

    x,y = np.mgrid[0:1+dx:dx, 0:1+dy:dy]

    if potential == "zero":
        V = np.zeros([Nx*x_step, Ny*y_step]) 
    elif potential == "random":
        sigma = 0.05
        V = 100 * randomGaussian2D()
    elif potential == "central":
        sigma = 0.05
        V = 100 * randomGaussian2D() * np.exp(-0.3*(x-0.5)**2/(2*sigma**2))
    elif potential == "slit":
        V = (np.tanh(10*np.pi*(x-0.425)) - np.tanh(10*np.pi*(x-0.575)) ) * (2 - np.tanh(10*np.pi*(y-0.425)) - np.tanh(-10*np.pi*(y-0.575)))/2
    elif potential == "double-slit":
        V = (np.tanh(10*np.pi*(x-0.425)) - np.tanh(10*np.pi*(x-0.575)) ) * (2 - np.tanh(10*np.pi*(y-0.3)) + np.tanh(10*np.pi*(y-0.45)) - np.tanh(10*np.pi*(y-0.55)) - np.tanh(-10*np.pi*(y-0.7)))/2

    normal =np.sum(np.absolute(V)**2)*dx*dy
    V /= np.sqrt(normal)
    
    # Wave packet parameters    
    sigma = 0.06
    p = 50
    x0 = 0.25
        
    if bc == "sin":
        psi0 = np.exp(-10*(x-0.5)**2)* np.exp(-10*(y-0.5)**2) * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
    elif bc == "random":
        sigma = 0.06*2
        psi0 = randomGaussian2D() + 1j*randomGaussian2D()
    elif bc == "wavepacket-previous":
        psi0 =  np.exp(-0.25*((x-x0)/sigma)**2)  * np.exp(1j*p*x)
    elif bc == "wavepacket":
        psi0 =  np.exp(-0.25*((x-x0)/sigma)**2)  * np.exp(1j*p*x) + np.exp(-0.25*((x-x0-1)/sigma)**2)  * np.exp(1j*p*(x-1))
    
    normal = np.sum(np.absolute(psi0)**2)*dx*dy
    psi0 /= np.sqrt(normal)
        
    psi_m1 = computeSK2D(psi0, V, steps = steps)
    return V[::x_step,::y_step], psi_m1[:, ::x_step, ::y_step]

###########################################################
# Generating the data loaders
###########################################################

from torch.utils.data import TensorDataset, DataLoader

batch_size = 32

train_samples = 32
test_samples = 1

saved = False

class WavefunctionDataset2D(TensorDataset):
    def __init__(self, samples, bc, potential, sample_times = Nt-1):
        
        # The total number of samples is samples*sample_times
        
        # input: (x, y, V, Re(psi_in), Im(psi_in))
        # ouput: (Re(psi_out), Im(psi_out))
        
        self.X = torch.Tensor(np.zeros([int(samples*sample_times), 5, Nx, Ny]))
        self.Y = torch.Tensor(np.zeros([int(samples*sample_times), 2, Nx, Ny]))
        
        idx = 0
        
        for s in np.arange(samples):
            
            # Solution of the Schrodinger equation
            V, sol = solveSchrodinger2D(bc=bc, potential=potential, steps=sample_times+1)
            
            # Uniform sample times
            ts = np.arange(sample_times)
            
            for t in ts:
                
                x, y = np.mgrid[0:1+dx:dx, 0:1+dy:dy]
                
                self.X[idx][0] = torch.Tensor(x)
                self.X[idx][1] = torch.Tensor(y)
                self.X[idx][2] = torch.Tensor(V)
                self.X[idx][3] = torch.Tensor(np.real(sol[t]))
                self.X[idx][4] = torch.Tensor(np.imag(sol[t]))
                
                self.Y[idx][0] = torch.Tensor(np.real(sol[t+1]))
                self.Y[idx][1] = torch.Tensor(np.imag(sol[t+1]))
                
                idx += 1
    
    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return {'x': x, 'y': y}
    
    def __len__(self):
        return len(self.X)

train_loader = DataLoader(WavefunctionDataset2D(samples = train_samples, bc="random", potential="random", sample_times=(Nt-1)), batch_size=batch_size, shuffle=True)
test_loader = {"central": DataLoader(WavefunctionDataset2D(samples = test_samples, bc="wavepacket", potential="central"), batch_size=batch_size, shuffle=True),
               "slit":    DataLoader(WavefunctionDataset2D(samples = 1, bc="wavepacket", potential="slit"), batch_size=batch_size, shuffle=True),
               "dslit":   DataLoader(WavefunctionDataset2D(samples = 1, bc="wavepacket", potential="double-slit"), batch_size=batch_size, shuffle=True)}