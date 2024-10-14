"""Vi har ikke brukt kodemal og er 
    veldig leie oss for at vi ikke fikk brukt klasser pga. vi ikke forsto hvordan vi skulle gjøre det med numba"""

import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools import constants
import numpy as np
from matplotlib import pyplot as plt
from numba import njit, prange
from scipy import integrate
import time
import os

import matplotlib as mpl # https://pythonforthelab.com/blog/python-tip-ready-publish-matplotlib-figures/ inspiration from this
mpl.rcParams['font.size'] = 16
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.format'] = 'png'  
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['axes.formatter.use_mathtext'] = False
plt.style.use('grayscale')

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
    Solves the radial distance in the orbit

    Parameters:
    a (float): Semi-major axis [AU]
    e (float): Eccentricity of the orbit
    f (array): Angle [radians]

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
    """
    Rotates coordinates by a given angle

    Parameters:
    x (array): x-coordinates [AU]
    y (array): y-coordinates [AU]
    angle (float): Rotation angle [radians]

    Returns:
    x_rot (array): Rotated x-coordinates [AU]
    y_rot (array): Rotated y-coordinates [AU]
    """
    rotMatrix = np.array([[np.cos(angle), -np.sin(angle)],  [np.sin(angle),  np.cos(angle)]])
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
    positions = np.zeros((N, num_planets, 2), dtype=np.float32) # makes a position 3D array
    velocities_over_time = np.zeros((N, num_planets, 2), dtype=np.float32)
    position = initial_positions.copy()
    positions[0] =  position # sets the face of the matrix

    velocities = initial_velocities.copy() # Initial velocities
    a_i = acceleration(positions[0], G, star_mass)

    velocities = initial_velocities.copy()
    velocities_over_time[0] = velocities

    for i in range(N-1):
        position += velocities * dt + 0.5 * a_i * dt ** 2
        positions[i + 1] = position
        a_iplus1 = acceleration(positions[i + 1], G, star_mass) #  update acceleration
        velocities += 0.5 * (a_i + a_iplus1) * dt # update velocities
        velocities_over_time[i + 1] = velocities
        a_i = a_iplus1 # sets the new acceleration
    return positions, velocities_over_time

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
    positions_over_time (array): Simulated positions of planets over time [AU]
    """
    positions_over_time, velocities_over_time = frog_leap_studios(T, dt, initial_positions, initial_velocities, G, star_mass, num_of_planets)
    N = int(T//dt)
    times = np.linspace(0,T,N)
    np.savez_compressed('planet_positions.npz', positions_over_time=positions_over_time, times = times)
    np.savez_compressed('planet_velocities.npz', velocities_over_time=velocities_over_time)
    return positions_over_time, velocities_over_time

def calculate_orbital_periods(positions_over_time, dt, num_of_planets):
    """
    Calculates the orbital periods of the planets.

    Parameters:
    positions_over_time (array): Positions of the planets over time [AU]
    dt (float): Time step [yr]

    Returns:
    periods (array): Orbital periods of the planets [yr]
    """
    periods = np.zeros(num_of_planets)

    for planet_idx in range(num_of_planets):
        x_positions = positions_over_time[:, planet_idx, 0]
        y_positions = positions_over_time[:, planet_idx, 1]
        theta = np.arctan2(y_positions, x_positions)
        theta = np.unwrap(theta) # makes theta continous in a sense
        
        d_theta = theta - theta[0] # change in theta
        
        orbit_indices = np.where(d_theta >= 2 * np.pi)[0] # get pos-index from when planet completes full orbit
        
        if len(orbit_indices) == 0:
            print(f"Planet {planet_idx + 1} did not complete a full orbit")
        else:
            period_idx = orbit_indices[0]
            period = period_idx * dt
            periods[planet_idx] = period
            print(f"Planet {planet_idx + 1} completed an orbit in {period:.2f} years.")
    return periods  

def compare_Kepler_orbits(periods, semi_major_axes, planet_masses, star_mass, G, corrected = False):
    """
    Compares the simulated orbital periods with Kepler and Newton

    Parameters:
    periods (array): Simulated orbital periods [yr]
    semi_major_axes (array): Semi-major axes of the planets [AU]
    star_mass (float): Mass of the star [solar masses]
    corrected (bool): If we should account for star mass

    Returns:
    None
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
    ax1.legend(loc="upper right")

    ax2.plot(range(1, len(periods) + 1), diff_newton, label="Difference from Newton")
    ax2.set_xlabel("Planet")
    ax2.set_ylabel("Difference in Orbital Periods (years)")
    ax2.set_title("Difference from Newton's Prediction")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    
    if corrected == True:
        plt.savefig('kepler_newton_deviation_corrected.png')
    if corrected == False:
        plt.savefig('kepler_newton_deviations.png')

def plot_combined(positions_over_time):
    """
    Plots the numerical and analytical orbits of the planets

    Parameters:
    positions_over_time (array): Self explanatory [AU]

    Returns:
    None
    """

    # Plot Numerical
    plt.figure()
    for i in range(system.number_of_planets):
        x = positions_over_time[:, i, 0] # takes all times steps of planet i at coordinate 0 (x-axis)
        y = positions_over_time[:, i, 1] # -- || --                                        1 (y-axis)
        plt.plot(x, y, alpha=0.8)
    # Plot analytical
    for i in range(system.number_of_planets):
        e = system.eccentricities[i]
        a = system.semi_major_axes[i]
        perihelion = system.aphelion_angles[i] + np.pi  
        f_vals = np.linspace(0, 2 * np.pi, 1000) 
        r = solve_analytic_r(a, e, f_vals)
        r_x, r_y = xy_pos(f_vals, r)
        r_x, r_y = rot_xy(r_x, r_y, perihelion)
        plt.plot(r_x, r_y, linestyle='--', alpha=1)
    
    # Make two plots at the center to give names
    plt.plot(0,0, label = "numerical", alpha=0.8) 
    plt.plot(0,0 , linestyle='--', label = "analytical", alpha=1) 

    star_color = np.array(system.star_color) / 255
    plt.plot(0, 0, "o", color=star_color, label='Star', markersize=10)

    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')
    plt.title('Comparison of Analytical and Numerical Orbits')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('combined_plot.png')


@njit
def find_idx_aphelion_perihelion(positions_over_time, planet_idx):
    """
    Finds the index of the aphelion and perihelion

    Parameters:
    positions_over_time (array): Positions of the planets over time [AU]
    planet_idx (int): Index of the planet

    Returns:
    idx_aphelion (int): Index of aphelion
    idx_perihelion (int): Index of periheleon
    """
    x_positions = positions_over_time[:, planet_idx, 0]
    y_positions = positions_over_time[:, planet_idx, 1]
    r = np.sqrt(x_positions**2 + y_positions**2)

    idx_aphelion = np.argmax(r)
    idx_perihelion = np.argmin(r)
    
    return idx_aphelion, idx_perihelion

def calculate_area(start_idx, steps, positions_over_time, planet_idx):
    """
    Calculates the area of the planet's radius vector over a given number of steps

    Parameters:
    start_idx (int): Starting index
    steps (int): Number of time_steps
    positions_over_time (array): Planet positions over time [AU]
    planet_idx (int): Index of the planet

    Returns:
    area (float): Area swept by the planet [AU²]
    """
    x_positions = positions_over_time[start_idx:start_idx+steps, planet_idx, 0]
    y_positions = positions_over_time[start_idx:start_idx+steps, planet_idx, 1]
    
    r = np.sqrt(x_positions**2 + y_positions**2)  
    theta = np.arctan2(y_positions, x_positions)  
    theta = np.unwrap(theta)

    area = 0.5 * integrate.trapezoid(r**2, theta)
    return area

def calculate_distance(start_idx, steps,positions_over_time, planet_idx):
    """
    Calculates the total distance traveled by the planet over a given number of steps.

    Parameters:
    start_idx (int): Starting index
    steps (int): Number of steps to include in the calculation
    positions_over_time (array): Planet positions over time [AU]
    planet_idx (int): Index of the planet

    Returns:
    total_distance (float): Total distance traveled by the planet [AU]
    """
    x_positions = positions_over_time[start_idx:start_idx+steps, planet_idx, 0]
    y_positions = positions_over_time[start_idx:start_idx+steps, planet_idx, 1]
    
    distances = np.sqrt(np.diff(x_positions)**2 + np.diff(y_positions)**2)
    total_distance = np.sum(distances)
    return total_distance




"""Two body simulation"""
@njit
def gravitational_force(G, m_1, m_2, r):
    """
    Computes the gravitational force between two masses based on their positions

    Parameters:
    G (float): Gravitational constant
    m_1 (float): Mass of the first [solar masses]
    m_2 (float): Mass of the second [solar masses]
    r (array): Position vector [AU]

    Returns:
    array: Gravitational force vector [AU/yr²]
    """
    r_norm = np.sqrt(r[0]**2 + r[1]**2)
    F = - G * m_1 * m_2 / r_norm**2
    direction = r / r_norm
    return F * direction

@njit     
def frog_leap_studios_two_body(T, dt, pos_star, pos_planet, v_star, v_planet, star_mass, planet_mass, G):
    """
    Leapfrog integrator - This is also based on the numerical compendium, but a bit less elegant

    Parameters:
    T (float): Total time [yrs]
    dt (float): Timesteps [yrs]
    pos_star (array): Initial position of the star [AU]
    pos_planet (array): Initial position of the planet [AU]
    v_star (array): Initial velocity of the star [AU/yr]
    v_planet (array): Initial velocity of the planet [AU/yr]
    star_mass (float): Mass of the star [solar masses]
    planet_mass (float): Mass of the planet [solar masses]
    G (float): Gravitational constant

    Returns:
    positions_star (array): Positions of the star [AU]
    positions_planet (array): Positions of the planet [AU]
    velocities_star (array): Velocities of the star [AU/yr]
    velocities_planet (array): Velocities of the planet [AU/yr]
    """
    N = int(T // dt)

    positions_star = np.zeros((N, 2))
    positions_planet = np.zeros((N, 2))
    velocities_star = np.zeros((N, 2))
    velocities_planet = np.zeros((N, 2))

    # to remove drifting we must account for v_CM
    pos_CM = (star_mass * pos_star + planet_mass * pos_planet) / (star_mass + planet_mass)
    v_CM = (star_mass * v_star + planet_mass * v_planet) / (star_mass + planet_mass)

    # account for CM
    pos_star -= pos_CM
    pos_planet -= pos_CM
    v_star -= v_CM
    v_planet -= v_CM

    # set the initial_conditions into the array
    positions_star[0] = pos_star
    positions_planet[0] = pos_planet
    velocities_star[0] = v_star
    velocities_planet[0] = v_planet
    
    r = pos_planet - pos_star
    f_planet = gravitational_force(G, star_mass, planet_mass, r) 
    f_star = -f_planet
    a_i_planet = f_planet / planet_mass
    a_i_star = f_star / star_mass
    for i in range(N - 1):
        # Update position
        pos_star += v_star * dt + 0.5 * a_i_star * dt **2
        pos_planet += v_planet * dt + 0.5 * a_i_planet * dt **2
        positions_star[i+1] = pos_star
        positions_planet[i+1] = pos_planet

        # Calculate new forces and accelerations
        r = pos_planet - pos_star
        f_planet = gravitational_force(G, star_mass, planet_mass, r)
        f_star = -f_planet

        a_plus1_planet = f_planet / planet_mass
        a_plus1_star = f_star / star_mass

        # Update velocities using correct shape
        v_planet += 0.5 * dt * (a_plus1_planet + a_i_planet)
        v_star += 0.5 * dt * (a_plus1_star + a_i_star)

        # Save updated velocities
        velocities_star[i+1] = v_star
        velocities_planet[i+1] = v_planet

        # Set new accelerations for next iteration
        a_i_planet = a_plus1_planet
        a_i_star = a_plus1_star
    return positions_star, positions_planet, velocities_star, velocities_planet

def radial_vel(v_star, inclination, T, dt, line_of_sight, peculiar_vel):
    """
    Computes the observed radial velocity of the star 

    Parameters:
    v_star (array): Velocities of the star [AU/yr]
    inclination (float): Inclination angle [radians]
    T (float): Total time [yrs]
    dt (float): Timesteps [yrs]
    line_of_sight (array): Line of sight - must be unit vector
    peculiar_vel (array): Peculiar velocity [AU/yr]

    Returns:
    times (array): Time steps [yrs]
    noise_radial_velocities (array): Radial velocities with added noise [AU/yr]
    """
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
    """
    Computes the relative flux

    Parameters:
    times (array): Timesteps [yrs]
    positions_planet (array): Positions of the planet [AU]
    positions_star (array): Positions of the star [AU]
    R_planet (float): Radius of the planet [AU]
    R_star (float): Radius of the star [AU]

    Returns:
    flux (array): Relative flux
    eclipse_start_idx (int): Index where the eclipse begins
    """
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
    """
    Calculates the total energy

    Parameters:
    m_star (float): Mass of the star [solar masses]
    m_planet (float): Mass of the planet [solar masses]
    positions_star (array): Positions of the star [AU]
    positions_planet (array): Positions of the planet[AU]
    velocities_star (array): Velocities of the star [AU/yr]
    velocities_planet (array): Velocities of the planet [AU/yr]
    G (float): Gravitational constant

    Returns:
    total_energy (array): Total energy of the system over time [AU²/yr²]
    """
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

def convert_energy(energy):
    """
    Converts energy from AU²/yr² to Joule

    Parameters:
    energy (float): Energy [AU²/yr²]

    Returns:
    energy (float): Energy [J]
    """
    au_to_m = utils.AU_to_m(1)  
    year_to_sec = utils.yr_to_s(1)  

    factor = au_to_m**2 / year_to_sec**2
    energy = factor * energy
    return energy



def plot_radial(times, noise_radial_velocities, N_body = False):
    """
    Plots the observed radial velocity of the star over time.

    Parameters:
    times (array): Time steps [yrs]
    noise_radial_velocities (array): Radial velocities with noise [AU/yr]

    Returns:
    None
    """
    plt.figure()
    plt.plot(times, noise_radial_velocities, label="Observed Radial Velocity", alpha=0.6)
    plt.xlabel("Time (years)")
    plt.ylabel("Radial Velocity (AU/year)")
    plt.legend(loc="upper right")
    plt.grid(True)
    if N_body == True:
        plt.title("Observed Radial Velocity of Star (N-planets)")
        plt.tight_layout()
        plt.savefig('N_body_radial_velocity_plot.png')
    if N_body == False:
        plt.title("Observed Radial Velocity of Star")
        plt.tight_layout()
        plt.savefig('radial_velocity_plot.png')

def plot_orbit(positions_star, positions_planet):
    """
    Plots the orbits of the star and planet and a zoomed in version

    Parameters:
    positions_star (array): Positions of the star [AU]
    positions_planet (array): Positions of the planet [AU]

    Returns:
    None
    """
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Full orbit plot
    ax1.plot(positions_star[:, 0], positions_star[:, 1], label="Star")
    ax1.plot(positions_planet[:, 0], positions_planet[:, 1], label="Planet")
    ax1.set_xlabel("x (AU)")
    ax1.set_ylabel("y (AU)")
    ax1.set_title("Full Orbit of Star and Planet")
    ax1.legend(loc="upper right")
    ax1.grid(True)
    ax1.axis('equal')

    # Plot of the stars orbit
    ax2.plot(positions_star[:, 0], positions_star[:, 1], label="Star")
    ax2.set_xlabel("x (AU)")
    ax2.set_ylabel("y (AU)")
    ax2.set_title("Zoomed View of Star's Orbit")
    ax2.legend(loc="upper right")
    ax2.grid(True)
    
    # Adjusts the zoom
    star_max = np.max(np.abs(positions_star)) * 1.1
    ax2.set_xlim(-star_max, star_max)
    ax2.set_ylim(-star_max, star_max)
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig('zoomed_orbit_plot.png')


def plot_flux(times, flux, eclipse_start_idx, buffer_steps):
    """
    Plots the light curve around an eclipse 

    Parameters:
    times (array): Timesteps [yrs]
    flux (array): Relative flux 
    eclipse_start_idx (int): Index where the eclipse begins
    buffer_steps (int): Number of steps to include around the eclipse

    Returns:
    NoneX
    """
    start_idx = eclipse_start_idx - buffer_steps
    end_idx = eclipse_start_idx + buffer_steps
    start_time = times[start_idx]
    times = times[start_idx:end_idx] * 365.25 * 24 * 3600
    times = times - times[0]
    plt.figure()
    plt.plot(times, flux[start_idx:end_idx], label="Flux around Eclipse", alpha=0.6)
    plt.xlabel(f"Time after {start_time:.2f} years (seconds)")
    plt.ylabel("Relative Flux")
    plt.title(f"Light Curve around the eclipse")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('flux_plot.png')
    
def plot_energy(times, energy):
    """
    Plots the deviation of the system's energy from the mean

    Parameters:
    times (array): Timesteps [yrs]
    energy (array): Total energy of the system over time [J]

    Returns:
    None
    """
    plt.figure()
    energy = convert_energy(energy) / 1000 # into kJ
    mean_energy = np.mean(energy)
    delta_energy = mean_energy - energy

    plt.plot(times, delta_energy, label=f"Deviation of energy from the mean: {mean_energy:.2f} kJ", alpha=0.6)
    plt.xlabel("Time (years)")
    plt.ylabel(f"Energy (kJ)")
    plt.ylim(min(delta_energy)*1.8, max(delta_energy)*1.8)
    plt.title("Deviation of Energy from Mean")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('energy_plot.png')

"""Part three"""
@njit
def acceleration_N(positions, masses, G):
    """
    Calculates the gravitational accelerations on each planet (+sun) from all other

    Parameters:
    positions (array): Positions [AU]
    masses (array): Masses [solar masses]
    G (float): Gravitational constant

    Returns:
    accelerations (array): Gravitational accelerations [AU/yr²]
    """
    num_objects = positions.shape[0]
    accelerations = np.zeros_like(positions)
    for i in range(num_objects):
        a_i = np.array([0.0, 0.0])
        for j in range(num_objects):
            if i != j: # sum over all elements apart from itself
                r_ij = positions[j] - positions[i] # direction (with a magnitude)
                r_norm = np.sqrt(np.sum(r_ij**2)) # magnitude
                a_i += G * masses[j] * r_ij / r_norm**3 # contribution from planet i
        accelerations[i] = a_i
    return accelerations

def frog_leap_studios_N_body(T, dt, initial_positions, initial_velocities, masses, G):
    """
    Leapfrog integrator 

    Parameters:
    T (float): Total simulation time [yrs]
    dt (float): Time step [yrs]
    initial_positions (array): Initial positions [AU]
    initial_velocities (array): Initial velocities [AU/yr]
    masses (array): Masses [solar masses]
    G (float): Gravitational constant

    Returns:
    positions (array): Simulated positions[AU]
    velocities (array): Simulated velocities  [AU/yr]
    """
    N = int(T // dt)
    num_objects = masses.shape[0]

    positions = np.zeros((N, num_objects, 2))
    velocities = np.zeros((N, num_objects, 2))

    total_mass = np.sum(masses)
    pos_CM = masses @ initial_positions / total_mass
    vel_CM = masses @ initial_velocities / total_mass

    initial_positions -= pos_CM
    initial_velocities -= vel_CM
    
    pos = initial_positions.copy()
    vel = initial_velocities.copy()

    positions[0] = pos
    velocities[0] = vel
    a_i = acceleration_N(pos, masses, G)  
    for i in range(N - 1):
        pos += vel * dt + 0.5 * a_i * dt ** 2
        positions[i + 1] = pos

        a_iplus1 = acceleration_N(pos, masses, G)
        vel += 0.5 * (a_i + a_iplus1) * dt
        velocities[i + 1] = vel

        a_i = a_iplus1
    return positions, velocities

def plot_all_orbits(positions_star, positions_planets):
    """
    Plots the orbits of the star and multiple planets, and a zoomed-in version of the star's orbit.

    Parameters:
    positions_star (array): Positions of the star [AU]
    positions_planets (array): Positions of the planets [AU, num_planets, 2]

    Returns:
    None
    """
    num_planets = positions_planets.shape[1]
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.plot(positions_star[:, 0], positions_star[:, 1], label="Star", linestyle='--')
    for i in range(num_planets):
        ax1.plot(positions_planets[:, i, 0], positions_planets[:, i, 1], label=f"Planet {i+1}")
    
    ax1.set_xlabel("x (AU)")
    ax1.set_ylabel("y (AU)")
    ax1.set_title("Orbits of the Star and Planets")
    ax1.legend(loc="upper right")
    ax1.grid(True)
    ax1.axis('equal')

    ax2.plot(positions_star[:, 0], positions_star[:, 1], label="Star", linestyle='--')
    ax2.set_xlabel("x (AU)")
    ax2.set_ylabel("y (AU)")
    ax2.set_title("Zoomed View of Star's Orbit")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    star_max = np.max(np.abs(positions_star)) * 1.1
    ax2.set_xlim(-star_max, star_max)
    ax2.set_ylim(-star_max, star_max)
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig('zoomed_orbit_multiple_planets.png')

def main():
    """First part"""
    T = 100  
    dt = 1e-5
    N = int (T//dt)
    t_list = np.linspace(0,T,N)
    num_of_planets = system.number_of_planets
    G = constants.G_sol
    initial_positions = system.initial_positions.copy().T
    initial_velocities = system.initial_velocities.copy().T

    # positions_over_time, velocities_over_time = gen_positions(
    #         T, dt, initial_positions, initial_velocities, G, star_mass, num_of_planets
    #     )
    if os.path.exists('planet_positions.npz') and os.path.exists('planet_velocities.npz'):
        positions_over_time = np.load('planet_positions.npz')['positions_over_time']
        velocities_over_time = np.load('planet_velocities.npz')['velocities_over_time']
    else:
        positions_over_time, velocities_over_time = gen_positions(
            T, dt, initial_positions, initial_velocities, G, star_mass, num_of_planets
        )
    transposed_pos = np.transpose(positions_over_time, axes=(2, 1, 0))
    system.verify_planet_positions(simulation_duration=T, planet_positions=transposed_pos, filename=None, number_of_output_points=None)
    system.generate_orbit_video(t_list, transposed_pos, number_of_frames=None, reduce_other_periods=True, filename='orbit_video.xml')

    plot_combined(positions_over_time)

    periods = calculate_orbital_periods(positions_over_time, dt, num_of_planets)
    semi_major_axes = system.semi_major_axes
    planet_masses = system.masses

    compare_Kepler_orbits(periods, semi_major_axes, planet_masses, star_mass, G)
    compare_Kepler_orbits(periods, semi_major_axes, planet_masses, star_mass, G, corrected=True)

    planet_idx = 0
    steps = 100
    delta_t = steps * dt    

    aphelion_angle = system.aphelion_angles[planet_idx]
    start_aphelion, start_perihelion = find_idx_aphelion_perihelion(positions_over_time, planet_idx)

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


    """Second part"""
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

    """Part three"""
    T = 1000
    dt = 1e-3
    G = constants.G_sol

    planet_indices = [0, 2, 4, 5]
    planet_masses = system.masses[planet_indices]
    planet_positions = system.initial_positions.T[planet_indices]
    planet_velocities = system.initial_velocities.T[planet_indices]

    masses = np.insert(planet_masses, 0, system.star_mass)
    initial_positions = np.insert(planet_positions, 0, [0.0, 0.0], axis=0)
    initial_velocities = np.insert(planet_velocities, 0, [0.0, 0.0], axis=0)

    positions_over_time, velocities_over_time = frog_leap_studios_N_body(
        T, dt, initial_positions, initial_velocities, masses, G)

    positions_star = positions_over_time[:, 0]
    velocities_star = velocities_over_time[:, 0]
    positions_planets = positions_over_time[:, 1:] 

    times, noise_radial_velocities = radial_vel(velocities_star, inclination, T, dt, line_of_sight, peculiar_vel)

    plot_all_orbits(positions_star, positions_planets)
    plot_radial(times, noise_radial_velocities, N_body = True)
    plt.show()


if __name__ == "__main__":
    main()