"""
This code is written without the skeleton-code.
"""

import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools import constants as constants
import numpy as np
from matplotlib import pyplot as plt

N = 10**5  # Number of particles
dt = 1e-12  # Time increment [s]
L = 1e-6  # Length of the sides of the cube[m]
T = 3000  # Temperature [K]
k = constants.k_B  # Boltzmann constant
m = constants.m_H2  # Mass of an H2 molecule [kg]
iterations = 1000    # Number of timesteps (iterations) 
    
seed = 4042
system = SolarSystem(seed) 
np.random.seed(seed)

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

    If the particle's position exceeds the boundaries,
    it resets the positions and reverses the direction.
    We assume perfect elasiticity
    
    Parameters:
    pos (array): The current positions [x, y, z] of the particle [m].  
    vel (array): The current velocities [vx, vy, vz] of the particle [m/s].
    L (float): Length of the sides of the cube [m].

    Returns:
    pos (array): Updated positions in x,y,z coordinates [m]
    vel (array): Updated velocities in x,y,z direction [m/s]
    indice_particles_at_thrust (array): returns the indices of the positions of particles escaping the rocket
    """
    indice_particles_at_bottom = pos <= 0 # Gives the location in the position matrix of values below 0 as True and False otherwise
    pos[indice_particles_at_bottom] = 0 # Sets all values that are True to 0 (back to the edge)
    vel[indice_particles_at_bottom] *= -1 # Changes the velocity in the direction of collision to the other direction
    # The same as above but values above L
    indice_particles_at_top = pos >= L 
    pos[indice_particles_at_top] = L
    vel[indice_particles_at_top] *= -1 

    particles_z0 = pos[:, 2] <= 0 # Gives index of particles with z-axis below 0
    particles_x_thrust = (pos[:, 0] > 0.25 * L) & (pos[:, 0] < 0.75 * L) # Gives index of particles with x position between 0.25L and 0.75L as True (where the hole is)
    particles_y_thrust = (pos[:, 1] > 0.25 * L) & (pos[:, 1] < 0.75 * L) # --||-- y_position
    indice_particles_at_thrust = np.where(particles_z0 & particles_x_thrust & particles_y_thrust)[0] # returns index of particles where all 3 conditions are met, we use [0] as the array is wrapped in a tuple

    return pos, vel, indice_particles_at_thrust


def validate_mean_speed(v_means):
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
    deviation = abs(v_mean_exp - v_mean_result)
    if deviation <= alpha:
        print("The simulated mean speed is within the wanted margin")
    else:
        print(f"The simulated mean speed deviated by {deviation} m/s from the expected.")

def calculate_force_exerted(indice_particles_at_thrust, velocities, m, dt):
    """
    Calculates the force exerted from the particles that have escaped

    Parameters:
    indice_particles_at_thrust (array): Indices of the particles that are at the thrust
    velocities (array) : The velocity-matrix with all velocities in x,y,z
    m (float) : The mass of the H2-particle
    
    Returns:
    dF (float) : The total force exerted on the rocket
    """
    z_velocities_escaping =  abs(velocities[indice_particles_at_thrust,2]) # makes an array of all the z-velocities of the escaping particles
    dp = np.sum(m * z_velocities_escaping) # gives the change of momentum
    dF = dp / dt # calculates the force exerted
    return dF 

def calculate_fuel_consumption(indice_particles_at_thrust, m, dt):
    consumed = (len(indice_particles_at_thrust) * m) / dt 
    return consumed 

print(calculate_fuel_consumption(indice_particles_at_thrust, m, dt))

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

    plt.grid(True, color='gray', linestyle='--')

    v_median = np.sqrt(2 * k * T / m)
    plt.axvline(x=v_median, color='black', linestyle=':', linewidth=2)
    plt.text(v_median, max(mb_distribution), r'Most Probable Speed', color='black', fontsize=8)

    plt.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.savefig('maxwell_boltzmann_comparison_grayscale.png', format='png', dpi=240)
    plt.show()



def main():
    positions = generate_random_positions(N, L) # generates our positions
    velocities = generate_maxwell_boltzmann_velocities(N) # generates our velocities
    time_list = np.linspace(0, (iterations-1)*dt, iterations) # makes an array of every time we look at

    v_means = np.zeros(iterations) # makes an empty array
    all_v_norms = np.zeros((iterations, N)) # makes an array

    for t in range(iterations):
        # Euler-Cromer
        positions, velocities, indice_particles_at_thrust = check_collision(positions, velocities, L) 
        positions += velocities * dt  

        v_norms = np.linalg.norm(velocities, axis=1)
        all_v_norms[t] = v_norms
        v_means[t] = np.mean(v_norms)
        calculate_force_exerted(indice_particles_at_thrust, velocities, m, dt)

    validate_mean_speed(v_means)
    plot_maxwell_boltzmann_comparison(all_v_norms, T, m, k)


if __name__ == "__main__":
    main()
