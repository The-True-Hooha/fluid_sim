import numpy as np
from PIL import Image
from scipy.special import erf
from .fluid_structure import Fluid
import tqdm

# Constants
RESOLUTION = (500, 500)
DURATION = 200
INFLOW_PADDING = 50
INFLOW_DURATION = 60
INFLOW_RADIUS = 8
INFLOW_VELOCITY = 1
INFLOW_COUNT = 5


class SimFlow:
    def __init__(self):
        print('Generating the fluid solver, please be patient....')
        self.fluid = Fluid(RESOLUTION, 'dye')
        self.center = np.floor_divide(RESOLUTION, 2)
        self.r = np.min(self.center) - INFLOW_PADDING
        self._setup_inflow()

    def _setup_inflow(self):
        points = np.linspace(-np.pi, np.pi, INFLOW_COUNT, endpoint=False)
        self.inflow_points = [
            self.r * np.array([np.cos(p), np.sin(p)]) + self.center for p in points]
        self.inflow_normals = [-p for p in self.inflow_points]

        self.inflow_velocity = np.zeros_like(self.fluid.velocity)
        self.inflow_dye = np.zeros(self.fluid.shape)

        for point, normal in zip(self.inflow_points, self.inflow_normals):
            mask = np.linalg.norm(
                self.fluid.indices - point[:, None, None], axis=0) <= INFLOW_RADIUS
            self.inflow_velocity[:, mask] += normal[:, None] * INFLOW_VELOCITY
            self.inflow_dye[mask] = 1

    def run_sim(self):
        frames = []
        for f in tqdm.tqdm(range(DURATION), desc="Generating frames"):
            if f <= INFLOW_DURATION:
                self.fluid.velocity += self.inflow_velocity
                self.fluid.dye += self.inflow_dye

            fluid_curl = self.fluid.step()[1]
            print(f"Frame {f}: fluid_curl shape: {fluid_curl.shape}")
            print(f"Frame {f}: self.fluid.shape: {self.fluid.shape}")
            print(f"Frame {f}: self.fluid.dye shape: {self.fluid.dye.shape}")

            fluid_curl = (erf(fluid_curl * 2) + 1) / 4
            color = np.dstack((fluid_curl, np.ones(self.fluid.shape), self.fluid.dye))
            color = (np.clip(color, 0, 1) * 255).astype('uint8')
            frames.append(Image.fromarray(color, mode='HSV').convert('RGB'))
        print('Saving simulation result.')
        frames[0].save('fluid_simulation.gif', save_all=True,append_images=frames[1:], duration=20, loop=0)
            # print(f"Frame {f}: fluid_curl shape after erf: {fluid_curl.shape}")

            # try:
            #     color = np.dstack((fluid_curl, np.ones(
            #         self.fluid.shape), self.fluid.dye))
            #     print(f"Frame {f}: color shape after dstack: {color.shape}")
            #     color = (np.clip(color, 0, 1) * 255).astype('uint8')
            #     frames.append(Image.fromarray(color, mode='HSV').convert('RGB'))
            # except ValueError as e:
            #     print(f"Error in frame {f}: {e}")
            #     print(f"fluid_curl min: {fluid_curl.min()}, max: {fluid_curl.max()}")
            #     print(f"self.fluid.dye min: {self.fluid.dye.min()}, max: {self.fluid.dye.max()}")
            # raise

