from variations.lattice_boltzman import FluidSim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main():
    lattice_boltzman = FluidSim(100, 100)
    print("running the boltzman simulation")
    lattice_boltzman.simulate_motion(1000)
    
    print("displaying results")
    fig, ax = plt.subplots()
    im = ax.imshow(lattice_boltzman.rho, cmap='viridis', animated=True)
    lattice_boltzman.visualize()
    plt.colorbar(im)
    ax.set_title('Fluid Density')
    
    # Function to update the plot
    def update(frame):
        simulator.simulate(10)  # Run 10 steps between each frame
        im.set_array(simulator.rho)
        return [im]
    
    anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

    # Show the plot
    plt.show()

    print("Visualization window closed. Program ending.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting gracefully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up and closing the program.")
        plt.close('all')  # 

