import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.sparse.linalg import factorized

from .derivatives import coefficient_difference, operators

# aim: 
# can I simulate fluids of different densities? like water, oil etc
# can I add external forces in the fluid environment
#



class Fluid:
    def __init__(self, shape, *quantities, pressure_order=1, advect_order = 3):
        self.shape = shape
        self.dimensions = len(shape)
        
        self.quantities = quantities
        for i in quantities:
            setattr(self, i, np.zeros(shape))
        
        self.indices = np.indices(shape)
        self.velocity = np.zeros((self.dimensions, *shape))
        
        laplace = operators(shape, coefficient_difference(2, pressure_order))
        self.pressure_solver = factorized(laplace)
        self.advect_order = advect_order
        
    
    def step(self):
        advect_map = self.indices - self.velocity
        
        def advect(field, filter_epilson=10e-2, mode='constant'):
            filtered = spline_filter(field, order=self.advect_order, mode=mode)
            field = filtered * (1 - filter_epilson) + field * filter_epilson
            return map_coordinates(field, advect_map, prefilter=False, order=self.advect_order, mode=mode)
        
        for i in range(self.dimensions):
            self.velocity[i] = advect(self.velocity[i])
        for quantity in self.quantities:
            setattr(self, quantity, advect(getattr(self, quantity)))
            
        # compute jacobian at each point in the velocity field to extract the curl and divergence of the fluid
        
        j_shape = (self.dimensions,) * 2
        partials = tuple(np.gradient(d) for d in self.velocity)
        jacobian = np.stack(partials).reshape(*j_shape, *self.shape)
        
        divergence = jacobian.trace()
        fluid_curl_filter = np.triu(np.ones(j_shape, dtype=bool), k=1)
        fluid_curl = (jacobian[fluid_curl_filter] - jacobian[fluid_curl_filter.T]).squeeze()
        
        # apply pressure auto correction on the fluid velocity field
        pressure = self.pressure_solver(divergence.flatten()).reshape(self.shape)
        self.velocity -= np.gradient(pressure)
        
        return divergence, fluid_curl, pressure
        
        