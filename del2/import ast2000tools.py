"""Vi har ikke brukt kodemal"""

import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools import constants
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('grayscale')
from numba import njit, prange
from scipy import integrate
import time
import os

seed = 4042
system = SolarSystem(seed)
np.random.seed(4042)

# Constants
G = constants.G_sol
star_mass = system.star_mass
au_to_m = constants.AU
year_to_sec = 31556926

"""Analytical and numerical simulations"""

def solve_analytic_r(a, e, f):
    """
    Solves the radial distance in an elliptical orbit analyticaly

    Parameters:
    a (float): Semi-major axis [AU].
    e (float): Eccentricity of the orbit
    f (array): Degrees [radians]

    Returns:
    r (array): Distance from the star [AU]
    """
    p = a * (1-e**2)
    r = p / (1 + e * np.cos(f))
    return r

def xy_pos(f, r):
    """
    Converts polar coordinates to Cartesian coordinates.

    Parameters:
    f (array): Degree [radian]
    r (array): Distance from the star [AU]

    Returns:
    r_x (array): x-coordinate [AU]
    r_y (array): y-coordinate [AU]
    """
    r_x = r * np.cos(f)
    r_y = r * np.sin(f)
    return r_x, r_y


def rot_xy(rx, ry, angle): # https://stackoverflow.com/questions/20840692/rotation-of-a-2d-array-over-an-angle-using-rotation-matrix heavily based on this, MAT1120 and Jonas' critique of our original plot :(
    rotMatrix = np.array([[np.cos(angle), -np.sin(angle)], 
                          [np.sin(angle),  np.cos(angle)]])
    positions = np.vstack((rx.flatten(), ry.flatten()))
    rotated_positions = rotMatrix @ positions
    return rotated_positions[0], rotated_positions[1] 




@njit
def acceleration(positions, G, star_mass):
    """
    Calculates the gravitational acceleration.

    Parameters:
    positions (array): Planet positions in (x,y) [AU]
    G (float): Gravitational constant
    star_mass (float): Mass of the star [Solar mass]

    Returns:
    accelerations (array): Gravitational accelerations of each planet [AU/year^2]
    """
    num_of_planets = positions.shape[0]
    accelerations = np.zeros_like(positions)
    for i in range(num_of_planets):
        r_i = np.sqrt(np.sum(positions[i]**2))
        a_i = -G * star_mass * positions[i] / r_i**3
        accelerations[i] = a_i
    return accelerations



@njit
def frog_leap_studios(T, dt, initial_positions, initial_velocities, G, star_mass, num_planets):
    """
    Leapfrog integrator - This is heavily based on the numerical compendium

    Parameters:
    T (float): Total simulation time [yrs]
    dt (float): Time step [yrs]
    initial_positions (array): Initial positions of the planets [AU]
    initial_velocities (array): Initial velocities of the planets [AU]
    G (float): Gravitational constant 
    star_mass (float): Mass of the star [Solar mass]
    num_planets (int): Number of planets.

    Returns:
    positions (array): Simulated positions of planets [AU]
    """
    N = int(T // dt) # Number of time steps
    positions = np.zeros((N, num_planets, 2)) # makes a position 3D array
    position = initial_positions.copy()
    positions[0] =  position 

    velocities = initial_velocities.copy() # Initial velocities
    a_i = acceleration(positions[0], G, star_mass)

    for i in range(N-1):
        position += velocities * dt + 0.5 * a_i * dt ** 2
        positions[i + 1] = position
        a_iplus1 = acceleration(positions[i + 1], G, star_mass) #  update acceleration
        velocities += 0.5 * (a_i + a_iplus1) * dt # update velocities
        a_i = a_iplus1 # sets the new acceleration
    return positions


def gen_positions(T, dt, initial_positions, initial_velocities, G, star_mass, num_of_planets):
    """
    Generates and saves the positions of the planets over time.

    Parameters:
    T (float): Total simulation time [yrs]
    dt (float): Time step [yrs]
    initial_positions (array): Initial positions of the planets [AU]
    initial_velocities (array): Initial velocities of the planets [AU/yr]
    G (float): Gravitational constant
    star_mass (float): Mass of the star [Solar mass]
    num_of_planets (int): Number of planets in the system

    Returns:
    positions_over_time (array): Simulated positions of planets over time [AU].
    """
    positions_over_time = frog_leap_studios(T, dt, initial_positions, initial_velocities, G, star_mass, num_of_planets)
    np.savez_compressed('planet_positions.npz', positions_over_time=positions_over_time)
    return positions_over_time


@njit
def find_idx_aphelion_perihelion(positions_over_time, planet_idx):
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

def compare_Kepler_orbits(periods, semi_major_axes, planet_masses, star_mass, G, corrected = False):
    """
    Plots the difference between Kepler's and Newton's versions of orbital periods compared to the simulated ones.

    Parameters:
    periods (array): Simulated orbital periods [years]
    semi_major_axes (array): Semi-major axes of the planets [AU]
    planet_masses (array): Masses of the planets [Solar masses]
    star_mass (float): Mass of the star [Solar masses]
    G (float): Gravitational constant
    """
    if corrected == True:
        kepler_periods = np.sqrt(semi_major_axes**3 / star_mass)
    if corrected == False:
        kepler_periods = np.sqrt(semi_major_axes**3)

    newton_periods = np.sqrt((4 * np.pi**2 * semi_major_axes**3) / (G * (star_mass + planet_masses)))

    diff_kepler = periods - kepler_periods
    diff_newton = periods - newton_periods

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    if corrected == True:
         ax1.plot(range(1, len(periods) + 1), diff_kepler, label="Difference from Kepler (adjusted for star mass)")
    if corrected == False:
         ax1.plot(range(1, len(periods) + 1), diff_kepler, label="Difference from Kepler")
    ax1.set_xlabel("Planet")
    ax1.set_ylabel("Difference in Orbital Periods (years)")
    ax1.set_title("Difference from Kepler's Prediction")
    ax1.grid(True)
    ax1.legend(loc="best")

    ax2.plot(range(1, len(periods) + 1), diff_newton, label="Difference from Newton")
    ax2.set_xlabel("Planet")
    ax2.set_ylabel("Difference in Orbital Periods (years)")
    ax2.set_title("Difference from Newton's Prediction")
    ax2.grid(True)
    ax2.legend(loc="best")

    plt.tight_layout()
    
    if corrected == True:
        plt.savefig('kepler_newton_deviation_corrected.png')
    if corrected == False:
        plt.savefig('kepler_newton_deviations.png')
    

def plot_combined(positions_over_time):
    plt.figure(figsize=(10, 10))
    
    # Plot the numerical orbits
    for i in range(system.number_of_planets):
        x = positions_over_time[:, i, 0] # takes all times steps of planet i at coordinate 0 (x-axis)
        y = positions_over_time[:, i, 1] # -- || --                                        1 (y-axis)
        plt.plot(x, y, alpha=0.8)

    # Plot the analytical
    for i in range(system.number_of_planets):
        e = system.eccentricities[i]
        a = system.semi_major_axes[i]
        aphelion_angle = system.aphelion_angles[i] + np.pi  # Adding pi if needed to adjust alignment
        f_vals = np.linspace(0, 2 * np.pi, 1000) 
        r = solve_analytic_r(a, e, f_vals)
        r_x, r_y = xy_pos(f_vals, r)
        r_x, r_y = rot_xy(r_x, r_y, aphelion_angle)
        plt.plot(r_x, r_y, linestyle='--', alpha=1)

    plt.plot(0,0, label = "numerical", alpha=0.8)
    plt.plot(0,0 , linestyle='--', label = "analytical", alpha=1)

    # Plot the star in the center
    star_color = np.array(system.star_color) / 255
    plt.plot(0, 0, "o", color=star_color, label='Star', markersize=10)

    # Formatting the plot
    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')
    plt.title('Comparison of Analytical and Numerical Planetary Orbits')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('combined_plot.png')


"""Two body simulation"""
@njit
def gravitational_force(G, m_1, m_2, r):
    r_norm = np.sqrt(r[0]**2 + r[1]**2)
    F = - G * m_1 * m_2 / r_norm**2
    direction = r / r_norm
    return F * direction

@njit
def frog_leap_studios_two_body(T, dt, pos_star, pos_planet, v_star, v_planet, star_mass, planet_mass, G):
    N = int(T // dt)

    positions_star = np.zeros((N, 2))
    positions_planet = np.zeros((N, 2))
    velocities_star = np.zeros((N, 2))
    velocities_planet = np.zeros((N, 2))

    pos_CM = (star_mass * pos_star + planet_mass * pos_planet) / (star_mass + planet_mass)
    v_CM = (star_mass * v_star + planet_mass * v_planet) / (star_mass + planet_mass) # use the total momentum

    pos_star -= pos_CM
    pos_planet -= pos_CM
    v_star -= v_CM
    v_planet -= v_CM

    positions_star[0] = pos_star
    positions_planet[0] = pos_planet
    velocities_star[0] = v_star
    velocities_planet[0] = v_planet

    for i in range(1, N):
        r = positions_planet[i - 1] - positions_star[i - 1]
        force_planet = gravitational_force(G, star_mass, planet_mass, r)
        force_star = -force_planet

        v_planet += 0.5 * dt * force_planet / planet_mass
        v_star += 0.5 * dt * force_star / star_mass

        positions_planet[i] = positions_planet[i - 1] + dt * v_planet
        positions_star[i] = positions_star[i - 1] + dt * v_star

        r = positions_planet[i] - positions_star[i]
        force_planet = gravitational_force(G, star_mass, planet_mass, r)
        force_star = -force_planet

        v_planet += 0.5 * dt * force_planet / planet_mass
        v_star += 0.5 * dt * force_star / star_mass

        velocities_planet[i] = v_planet
        velocities_star[i] = v_star

    return positions_star, positions_planet, velocities_star, velocities_planet

def radial_vel(v_star, inclination, T, dt, line_of_sight, peculiar_vel):
    N = int(T // dt)
    times = np.linspace(0, T, N)
    
    radial_velocity_star = np.dot(v_star, line_of_sight) * np.sin(inclination)
    radial_velocity_peculiar = np.dot(peculiar_vel, line_of_sight)
    radial_velocities = radial_velocity_star + radial_velocity_peculiar
    
    noise_std = 0.2 * np.max(abs(radial_velocity_star))
    noise = np.random.normal(0, noise_std, size=N)
    noise_radial_velocities = radial_velocities + noise

    return times, noise_radial_velocities

@njit
def compute_flux(times, positions_planet, positions_star, R_planet, R_star):
    flux_min = 1 - (R_planet / R_star) ** 2

    N = len(times)
    flux = np.ones(N) 
    has_eclipse = False
    for i in range(N):
        dy = abs(positions_planet[i, 1] - positions_star[i, 1])
        if dy <= abs(R_star):  
            flux[i] = flux_min
            if i >= 10000 and has_eclipse == False: # this was a desperate attempt to make another eclipse the main focus as we begin our simulation with half an eclipse
                has_eclipse == True
                eclipse_start_idx = i  

    noise_std = 1e-5 * flux_min
    noise = np.random.normal(0, noise_std, size=N)  
    flux = flux + noise

    return flux, eclipse_start_idx

@njit(parallel=True)
def calc_energy(m_star, m_planet, positions_star, positions_planet, velocities_star, velocities_planet, G):
    N = positions_star.shape[0]
    total_energy = np.zeros(N)

    mu = (m_star * m_planet) / (m_star + m_planet)
    
    for i in prange(N):
        r_vec = positions_planet[i] - positions_star[i]
        r = np.sqrt(r_vec[0]**2 + r_vec[1]**2) # numba didnt accept linalg.norm :(

        v_vec = velocities_planet[i] - velocities_star[i]
        v_relative = np.sqrt(v_vec[0]**2 + v_vec[1]**2)

        kinetic_energy = 0.5 * mu * v_relative**2
        potential_energy = -G * m_star * m_planet / r
        total_energy[i] = kinetic_energy + potential_energy
    return total_energy

def convert_energy(energy_in_au2_yr2):
    au_to_m = utils.AU_to_m(1)  
    year_to_sec = utils.yr_to_s(1)  

    conversion_factor = au_to_m**2 / year_to_sec**2

    energy_in_joules = energy_in_au2_yr2 * conversion_factor
    return energy_in_joules



def plot_radial(times, noise_radial_velocities):
    plt.figure(figsize=(10, 6))
    plt.plot(times, noise_radial_velocities, label="Observed Radial Velocity", alpha=0.6)
    plt.xlabel("Time (years)")
    plt.ylabel("Radial Velocity (AU/year)")
    plt.title("Observed Radial Velocity of Star")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('radial_velocity_plot.png')

def plot_orbit(positions_star, positions_planet):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Full orbit plot
    ax1.plot(positions_star[:, 0], positions_star[:, 1], label="Star")
    ax1.plot(positions_planet[:, 0], positions_planet[:, 1], label="Planet")
    ax1.set_xlabel("X (AU)")
    ax1.set_ylabel("Y (AU)")
    ax1.set_title("Full Orbit of Star and Planet")
    ax1.legend(loc="best")
    ax1.grid(True)
    ax1.axis('equal')

    # Plot of the stars orbit
    ax2.plot(positions_star[:, 0], positions_star[:, 1], label="Star")
    ax2.set_xlabel("X (AU)")
    ax2.set_ylabel("Y (AU)")
    ax2.set_title("Zoomed-in View of Star's Orbit")
    ax2.legend(loc="best")
    ax2.grid(True)
    
    # Adjusts the zoom
    star_max = np.max(np.abs(positions_star)) * 1.1
    ax2.set_xlim(-star_max, star_max)
    ax2.set_ylim(-star_max, star_max)
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig('zoomed_orbit_plot.png')


def plot_flux(times, flux, eclipse_start_idx, buffer_steps):
    start_idx = eclipse_start_idx - buffer_steps
    end_idx = eclipse_start_idx + buffer_steps
    start_time = times[start_idx]
    times = times[start_idx:end_idx] * 365.25 * 24 * 3600
    times = times - times[0]
    plt.figure(figsize=(10, 6))

    plt.plot(times, flux[start_idx:end_idx], label="Flux around Eclipse", alpha=0.6)
    
    plt.xlabel(f"Time after {start_time} years (seconds)")
    plt.ylabel("Relative Flux")
    plt.title(f"Light Curve around the eclipse")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('flux_plot.png')
    
def plot_energy(times, energy):
    energy = convert_energy(energy) / 1000 # into kJ
    mean_energy = np.mean(energy)
    delta_energy = mean_energy - energy

    plt.figure(figsize=(10, 6))
    plt.plot(times, delta_energy, label=f"Deviation of energy from the mean: {mean_energy:.2f} kJ", alpha=0.6)
    plt.xlabel("Time (years)")
    plt.ylabel(f"Energy (kJ)")
    plt.ylim(min(delta_energy)*1.8, max(delta_energy)*1.8)
    plt.title("Deviation of energy from mean")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('energy_plot.png')



def main():
    """First part"""
    
    T = 100  
    dt = 1e-5  
    num_of_planets = system.number_of_planets
    G = constants.G_sol

    
    initial_positions = system.initial_positions.T.copy()
    initial_velocities = system.initial_velocities.T.copy()

    
    if os.path.exists('planet_positions.npz'):
        positions_over_time = np.load('planet_positions.npz')['positions_over_time']
    else:
        positions_over_time = gen_positions(T, dt, initial_positions, initial_velocities, G, star_mass, num_of_planets)

    
    plot_combined(positions_over_time)

    
    periods = calculate_orbital_periods(positions_over_time, dt, num_of_planets)
    semi_major_axes = system.semi_major_axes
    planet_masses = system.masses

    compare_Kepler_orbits(periods, semi_major_axes, planet_masses, star_mass, G)
    compare_Kepler_orbits(periods, semi_major_axes, planet_masses, star_mass, G, corrected=True)

    """Second part"""
   
    G = constants.G_sol
    T = 14  
    dt = 1e-5  

    planet_idx = 0
    pos_planet = np.copy(system.initial_positions.T[planet_idx])
    pos_star = np.array([0.0, 0.0])
    v_star = np.array([0.0, 0.0])
    v_planet = np.copy(system.initial_velocities.T[planet_idx])
    m_star = system.star_mass
    m_planet = system.masses[planet_idx]
    R_star = utils.km_to_AU(system.star_radius)
    R_planet = utils.km_to_AU(system.radii[planet_idx])

    peculiar_vel = np.array([0.001, 0.004])

    positions_star, positions_planet, velocities_star, velocities_planet = frog_leap_studios_two_body(T, dt, pos_star, pos_planet, v_star, v_planet, m_star, m_planet, G)

    line_of_sight = np.array([1, 0])   
    inclination = np.radians(85)  

    times, noise_radial_velocities = radial_vel(velocities_star, inclination, T, dt, line_of_sight, peculiar_vel)
    flux, eclipse_start_idx = compute_flux(times, positions_planet, positions_star, R_planet, R_star)

    
    buffer_steps = 600

    
    plot_flux(times, flux, eclipse_start_idx, buffer_steps)

    
    energies = calc_energy(m_star, m_planet, positions_star, positions_planet, velocities_star, velocities_planet, G)
    plot_radial(times, noise_radial_velocities)
    plot_orbit(positions_star, positions_planet)
    plot_energy(times, energies)

    plt.show()

if __name__ == "__main__":
    main()