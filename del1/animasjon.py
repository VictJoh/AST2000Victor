import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
import ast2000tools.constants as constants
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


seed = 4042
system = SolarSystem(seed) 

N = 10  # Number of particles
dt = 1e-12  # Time increment [s]
L = 1e-6  # Length of the sides of the cube[m]
T = 3000  # Temperature [K]
k = constants.k_B  # Boltzmann constant
m = constants.m_H2  # Mass of an H2 molecule [kg]
iterations = 1000 # Number of timesteps (iterations) 

time_list = np.linspace(0, (iterations-1)*dt, iterations)

def generate_random_positions(N, L):
    """
    Generate random coordinates within a cube for N particles. The particles are distributed uniformly.
 
    Parameters:
    N (int): Number of particles for which to generate coordinates.
    L (float): Length of the sides of the cube [m].

    Returns:
    x, y, z (arrays) = x,y,z coordinates of particles in cube [m]
    """
    return np.random.uniform(low=0, high=L, size=(int(N), 3))    

def generate_maxwell_boltzmann_velocities(N):
    """
    Generates velocities for N particles in each direction according to the Maxwell-Bolzmann distribution.

    Parameteres:
    N (int) = Number of particles

    Returns: 
    abs_v (array) = absolute value of velocity [m/s]
    vx, vy, vz (arrays) = velocity in x,y,z direction [m/s]
    """
    std = np.sqrt(k * T / m) # Standard deviation of the Maxwell-Boltzmann distribution
    return np.random.normal(loc=0, scale=std, size = (int(N), 3))

def check_collision(pos, vel, L):   
    """
    Adjusts the velocity of a particle when it hits a wall of the cube.

    If the particle's position exceeds the boundaries of the cube (either 0 or L),
    it resets the position to the boundary value and reverses the direction of velocity.
    We assume perfect elasiticity
    
    Parameters:
    pos (array): The current positions [x, y, z] of the particle [m].
    vel (array): The current velocities [vx, vy, vz] of the particle [m/s].
    L (float): Length of the sides of the cube [m].

    Returns:
    vel (array): Updated velocities in x,y,z direction [m/s]
    """

    particles_at_bottom = pos <= 0
    pos[particles_at_bottom] = 0
    vel[particles_at_bottom] *= -1

    particles_at_top = pos >= L 
    pos[particles_at_top] = L
    vel[particles_at_top] *= -1

    return pos, vel

positions = generate_random_positions(N, L)
velocities = generate_maxwell_boltzmann_velocities(N)

v_means = np.zeros(iterations)
all_v_norms = np.zeros((iterations, N))

for t in range(iterations):
    positions += velocities * dt #Euler-Forward
    positions, velocities = check_collision(positions, velocities, L)

    v_norms = np.linalg.norm(velocities, axis=1)
    all_v_norms[t] = v_norms
    v_means[t] = np.mean(v_norms)

def validate_mean_speed():
    """
    Validates the mean speed of particles in a simulation by comparing it to the expected 
    mean speed from the Maxwell-Boltzmann distribution.

    Parameters:
    None

    Returns:
    None: Prints a message indicating whether the test passed or failed
    """
    v_mean_exp = np.sqrt(8 * k * T / (np.pi * m))
    v_mean_result = np.mean(v_means)
    alpha = 0.05 * v_mean_exp
    
    if abs(v_mean_exp - v_mean_result) <= alpha:
        print("Test passed: The simulated mean speed is within the expected margin.")
    else:
        print(f"Test failed: The simulated mean speed deviated by {abs(v_mean_exp - v_mean_result)} from the expected.")

def plot_maxwell_boltzmann_comparison(all_v_norms, T, m, k):
    """
    Plots the distribution of particle speeds from a simulation as a histogram. On top is the theoretical
    Maxwell-Boltzmann distribution for a comparison.

    Parameters:
    all_v_norms (array): All the speeds of particles from the simulation [m/s].
    T (float): The temperature of the system[K].
    m (float): Particle-mass [kg].
    k (float): Boltzmann's constant [J/K].

    Returns:
    None: This function displays the plot but does not return any value.
    """
    max_speed = np.max(all_v_norms)
    speeds = np.linspace(0, max_speed, 400)
    factor = 4 * np.pi * (m / (2 * np.pi * k * T))**(3/2)
    exponent = np.exp(-m * speeds**2 / (2 * k * T))
    mb_distribution = factor * speeds**2 * exponent

    plt.figure(figsize=(8, 6), dpi=240)

    plt.hist(all_v_norms.flatten(), bins=50, density=True, alpha=0.6, color='gray', label='Simulated Speeds')

    plt.plot(speeds, mb_distribution, color='black', linewidth=2, linestyle='--', label='Maxwell-Boltzmann Distribution')

    plt.xlabel(r'Speed (m/s)', fontsize=10)
    plt.ylabel(r'Probability Density', fontsize=10)
    plt.title('Simulated Speeds vs Maxwell-Boltzmann Distribution', fontsize=12)

    plt.grid(True, color='gray', linestyle='--', alpha=0.6)

    v_median = np.sqrt(2 * k * T / m)
    plt.axvline(x=v_median, color='black', linestyle=':', linewidth=2)
    plt.text(v_median, max(mb_distribution), r'Most Probable Speed', color='black', fontsize=8)

    plt.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.savefig('maxwell_boltzmann_comparison_grayscale.png', format='png', dpi=240)
    plt.show()


# validate_mean_speed()
plot_maxwell_boltzmann_comparison(all_v_norms, T, m, k)

# Prepare figure for animation
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_zlim(0, L)
scat = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2])

def update(frame):
    global positions, velocities
    positions += velocities * dt  # Update positions
    positions, velocities = check_collision(positions, velocities, L)  # Check for collisions
    scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
    return scat,

# Create animation
ani = FuncAnimation(fig, update, frames=iterations, interval=1, blit=False)

plt.show()
