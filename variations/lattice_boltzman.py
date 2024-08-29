"""
using the lattice boltzman method for this example
"""

import numpy as np
import matplotlib.pyplot as plt


class FluidSim:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.rho = np.ones((height, width))
        self.ux = np.zeros((height, width))
        self.uy = np.zeros((height, width))

        # the lattice velocities
        self.e = np.array([
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
            [1, 1], [-1, 1], [-1, -1], [1, -1]
        ])

        # the weights for each direction of fluid flow
        self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

        self.f = np.zeros((9, height, width))
        self.feq = np.zeros((9, height, width))
        
    def fluid_equilibrium(self):
        for i in range(9):
            eu = self.e[i, 0]* self.ux + self.e[i, 1] * self.uy
            uv = self.ux**2 + self.uy**2
            self.feq[i] = self.rho * self.w[i] * (1 + 3 * eu + 4.5 * eu ** 2 - 1.5 * uv)
            
    def fluid_collision(self):
        # tau = 0.6
        self.f += - (1/0.6) * (self.f - self.feq)
    
    def stream(self):
        for i in range(9):
            self.f[i] = np.roll(self.f[i], self.e[i], axis= (0,1))
    
    def update_marco_properties(self):
        self.rho = np.sum(self.f, axis=0)
        self.uy = np.sum(self.f * self.e[:, 0][:,np.newaxis, np.newaxis], axis=0) / self.rho
        self.ux = np.sum(self.f * self.e[:, 0][:,np.newaxis, np.newaxis], axis=0) / self.rho
    
    def simulate_motion(self, steps):
        for _ in range(steps):
            self.fluid_equilibrium()
            self.fluid_collision()
            self.stream()
            self.update_marco_properties()
    
    def visualize(self):
        plt.imshow(self.rho, cmap='viridis')
        plt.colorbar()
        plt.title('the fluid density')
        plt.show()
        
