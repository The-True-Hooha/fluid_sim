from variations.lattice_boltzman import FluidSim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from variations.navier_stokes.sim import SimFlow


class Main:
    def __init__(self):
        self.lattice_boltzman = None
        self.navier_stokes = None

    def lattice_boltzman_sim(self):
        self.lattice_boltzman = FluidSim(100, 100)
        print("Running the Lattice Boltzmann simulation")
        self.lattice_boltzman.simulate_motion(1000)

        print("Displaying results")
        fig, ax = plt.subplots()
        im = ax.imshow(self.lattice_boltzman.rho,
                       cmap='viridis', animated=True)
        plt.colorbar(im)
        ax.set_title('Fluid Density (Lattice Boltzmann)')

        def update(frame):
            self.lattice_boltzman.simulate_motion(
                10)  # Run 10 steps between each frame
            im.set_array(self.lattice_boltzman.rho)
            return [im]

        anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
        plt.show()

    def navier_stokes_sim(self):
        print("Running the Navier-Stokes simulation")
        self.navier_stokes = SimFlow()
        self.navier_stokes.run_sim()
        print("Navier-Stokes simulation completed")

    def run_simulation(self):
        while True:
            print("\nWhich simulation do you want to run?")
            print(
                "Select (a) for Navier-Stokes or (b) for Lattice Boltzmann or (q) to quit")
            choice = input("Select option: ").lower()

            if choice == 'a':
                self.navier_stokes_sim()
            elif choice == 'b':
                self.lattice_boltzman_sim()
            elif choice == 'q':
                print("Exiting the program.")
                break
            else:
                print("The option you selected does not exist. Please try again.")


if __name__ == "__main__":
    try:
        main = Main()
        main.run_simulation()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting gracefully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up and closing the program.")
        plt.close('all')
