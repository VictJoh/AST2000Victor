import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools import constants
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from scipy import integrate

seed = 4042
system = SolarSystem(seed)

G = constants.G_sol
star_mass = system.star_mass
au_to_m = constants.AU
year_to_sec = 31556926

def solve_analytic_r(a, e, f):
    p = a * (1-e**2)
    r = p / (1 + e * np.cos(f))
    return r

def xy_pos(f, r):
    r_x = r * np.cos(f)
    r_y = r * np.sin(f)
    return r_x, r_y

@jit
def acceleration(positions, G, star_mass):
    num_of_planets = positions.shape[0]
    accelerations = np.zeros_like(positions)
    for i in range(num_of_planets):
        r_i = np.sqrt(np.sum(positions[i]**2))
        a_i = -G * star_mass * positions[i] / r_i**3
        accelerations[i] = a_i
    return accelerations

@jit
def frog_leap_studios(T, dt, initial_positions, initial_velocities, G, star_mass, num_planets):
    """
    This code is based heavily on the numerical compendium
    """
    N = int(T // dt) # Number of time steps
    positions = np.zeros((N, num_planets, 2)) # makes a position tensor (2 * num_of_planets matrix over N time-steps)
    positions[0] = initial_positions.copy() # makes the "matrix" of the tensor into the original position

    velocities = initial_velocities.copy() # Initial velocities
    a_i = acceleration(positions[0], G, star_mass) # Initial accelerations
    t = 0 # Initial time
    for i in range(N - 1):
        positions[i + 1] = positions[i] + velocities * dt + 0.5 * a_i * dt ** 2 # setting pos at t = i+1
        a_iplus1 = acceleration(positions[i + 1], G, star_mass) #  update acceleration
        velocities += 0.5 * (a_i + a_iplus1) * dt # update velocities
        a_i = a_iplus1 # sets the new acceleration

    return positions


def plot_analytic():
    f_vals = np.linspace(0, 2*np.pi, 360)
    
    plt.figure(figsize=(8,8))
    for i in range(system.number_of_planets):
        e = system.eccentricities[i]
        a = system.semi_major_axes[i]
        r = solve_analytic_r(a, e, f_vals)
        r_x, r_y = xy_pos(f_vals, r)
        plt.plot(r_x, r_y, label=f'Planet {i+1}')
    
    star_color = np.array(system.star_color) / 255
    plt.plot(0, 0, "o", color=star_color, label='Star', markersize=10) # not to scale of course
    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')
    plt.title('Analytical Solution: Elliptical Orbits of Planets')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_numeric(positions_over_time):
    plt.figure(figsize=(8, 8))
    for i in range(system.number_of_planets):
        x_positions = positions_over_time[:, i, 0]  
        y_positions = positions_over_time[:, i, 1]  
        plt.plot(x_positions, y_positions, label= f'Planet {i + 1}')

    star_color = np.array(system.star_color) / 255 # turns into right format for pyplot
    plt.plot(0, 0, "o", color=star_color, label='Star', markersize=10)
    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')
    plt.title('Numerical Simulation of Planetary Orbits')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_combined(positions_over_time):
    f_vals = np.linspace(0, 2*np.pi, 360)
    
    plt.figure(figsize=(10,10))
    for i in range(system.number_of_planets):
        e = system.eccentricities[i]
        a = system.semi_major_axes[i]
        r = solve_analytic_r(a, e, f_vals)
        r_x, r_y = xy_pos(f_vals, r)
        plt.plot(r_x, r_y, label=f'Planet {i+1} (Analytical)', linestyle='--')
        
        plt.plot(positions_over_time[:, i, 0],
                 positions_over_time[:, i, 1],
                 label=f'Planet {i+1} (Numerical)')
    
    star_color = np.array(system.star_color) / 255
    plt.plot(0, 0, "o", color=star_color, label='Star', markersize=10)
    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')
    plt.title('Comparison of Analytical and Numerical Planetary Orbits')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_sim():
    initial_positions = system.initial_positions.T.copy()
    initial_velocities = system.initial_velocities.T.copy()
    num_of_planets = system.number_of_planets
    positions_over_time = frog_leap_studios(T, dt, initial_positions, initial_velocities, G, star_mass, num_of_planets)
    
    np.save('planet_positions.npy', positions_over_time)
    return positions_over_time

@jit
def find_idx_aphelion_perihelion(positions_over_time, planet_idx, aphelion_angle):
    x_positions = positions_over_time[:, planet_idx, 0]
    y_positions = positions_over_time[:, planet_idx, 1]
    r = np.sqrt(x_positions**2 + y_positions**2)

    idx_aphelion = np.argmax(r)
    idx_perihelion = np.argmin(r)
    
    return idx_aphelion, idx_perihelion

def calculate_area(start_idx, steps, positions_over_time, planet_idx):
    x_positions = positions_over_time[start_idx:start_idx+steps, planet_idx, 0]
    y_positions = positions_over_time[start_idx:start_idx+steps, planet_idx, 1]
    
    r = np.sqrt(x_positions**2 + y_positions**2)  
    theta = np.arctan2(y_positions, x_positions)  
    theta = np.unwrap(theta)  # makes theta continous

    area = 0.5 * integrate.trapezoid(r**2, theta)
    return area

def calculate_distance(start_idx, steps,positions_over_time, planet_idx):
    x_positions = positions_over_time[start_idx:start_idx+steps, planet_idx, 0]
    y_positions = positions_over_time[start_idx:start_idx+steps, planet_idx, 1]
    
    distances = np.sqrt(np.diff(x_positions)**2 + np.diff(y_positions)**2) # this is basically forward-euler using np.diff
    total_distance = np.sum(distances)
    return total_distance

def calculate_orbital_periods(positions_over_time, dt, num_of_planets):
    periods = np.zeros(num_of_planets)

    for planet_idx in range(num_of_planets):
        x_positions = positions_over_time[:, planet_idx, 0]
        y_positions = positions_over_time[:, planet_idx, 1]
    
        theta = np.arctan2(y_positions, x_positions)
        theta = np.unwrap(theta) # makes theta continous
        
        d_theta = theta - theta[0] # change in theta
        
        orbit_indices = np.where(d_theta >= 2 * np.pi)[0] # get pos-index from when planet completes full orbit
        
        if len(orbit_indices) == 0:
            print(f"Planet {planet_idx + 1} did not complete a full orbit")
        else:
            # The orbital period is the time at which the first full orbit is completed
            period_idx = orbit_indices[0]
            period = period_idx * dt
            periods[planet_idx] = period
            print(f"Planet {planet_idx + 1} completed an orbit in {period:.2f} years.")
    return periods  

def compare_Kepler_orbits(periods, semi_major_axes, planet_masses, star_mass):
    kepler = periods**2 / semi_major_axes**3

    newton = np.sqrt((4 * np.pi**2 * semi_major_axes**3) / (G * (star_mass + planet_masses)))
    
    for i in range(len(periods)):
        print(f"Planet {i+1}:")
        print(f"Kepler's Period (T^2 / a^3): {kepler[i]:.6f}")
        print(f"Difference between Simulated and Newton's period: {abs(periods[i] - newton[i]):.6f} years")
        


if __name__ == "__main__":   
    plt.style.use('grayscale')
    T = 50
    dt = 1e-4
    run_sim()
    positions_over_time = np.load('planet_positions.npy')
    
    # plot_analytic()
    # plot_numeric(positions_over_time)
    plot_combined(positions_over_time)
    planet_idx = 0
    aphelion_angle = system.aphelion_angles[planet_idx]
    start_aphelion, start_perihelion = find_idx_aphelion_perihelion(positions_over_time, planet_idx)
    steps = 100
    num_of_planets = system.number_of_planets
    delta_t = steps * dt

    area_aphelion = calculate_area(start_aphelion, steps, positions_over_time, planet_idx)
    area_perihelion = calculate_area(start_perihelion, steps, positions_over_time, planet_idx)

    print(f"Area at aphelion {area_aphelion}")
    print(f"Area at perihelion  {area_perihelion}")

    distance_aphelion = calculate_distance(start_aphelion, steps, positions_over_time, planet_idx)
    distance_perihelion = calculate_distance(start_perihelion, steps, positions_over_time, planet_idx)

    print(f"Distance traveled near aphelion: {distance_aphelion} AU")
    print(f"Distance traveled near perihelion: {distance_perihelion} AU")

    mean_velocity_aphelion = distance_aphelion / delta_t
    mean_velocity_perihelion = distance_perihelion / delta_t

    print(f"Mean velocity near aphelion: {mean_velocity_aphelion} AU/yr")
    print(f"Mean velocity near perihelion: {mean_velocity_perihelion} AU/yr")

    periods = calculate_orbital_periods(positions_over_time, dt, num_of_planets)
    semi_major_axes = system.semi_major_axes
    planet_masses = system.masses
    star_mass = system.star_mass
    compare_Kepler_orbits(periods, semi_major_axes, planet_masses, star_mass)