import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools import constants
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
import time

seed = 4042
system = SolarSystem(seed)

@jit
def gravitational_force(G, m_1, m_2, r):
    r_norm = np.sqrt(r[0]**2 + r[1]**2)
    F = - G * m_1 * m_2 / r_norm**2
    direction = r / r_norm
    return F * direction

@jit
def frog_leap_studios_two_body(T, dt, pos_star, pos_planet, v_star, v_planet, star_mass, planet_mass, G):
    N = int(T // dt)

    positions_star = np.zeros((N,2))
    positions_planet = np.zeros((N,2))
    velocities_star = np.zeros((N,2))
    velocities_planet = np.zeros((N,2))
    positions_CM = np.zeros((N,2))
    velocities_CM = np.zeros((N,2))
    
    pos_CM = (star_mass * pos_star + planet_mass * pos_planet) / (star_mass + planet_mass)
    v_CM = (star_mass * v_star + planet_mass * v_planet) / (star_mass + planet_mass)


    #sets to the CM-reference
    pos_star -= pos_CM
    pos_planet -= pos_CM
    v_star -= v_CM
    v_planet -= v_CM

    positions_star[0] = pos_star
    positions_planet[0] = pos_planet
    velocities_star[0] = v_star
    velocities_planet[0] = v_planet
    velocities_CM[0] = (star_mass * v_star + planet_mass * v_planet) / (star_mass + planet_mass)
    positions_CM[0] = (star_mass * pos_star + planet_mass * pos_planet) / (star_mass + planet_mass)
    

    
    for i in range(1, N):
        r = positions_planet[i-1] - positions_star[i-1]

        force_planet = gravitational_force(G, star_mass, planet_mass, r)
        force_star = -force_planet

        v_planet += 0.5 * dt * force_planet / planet_mass
        v_star += 0.5 * dt * force_star / star_mass

        positions_planet[i] = positions_planet[i-1] + dt * v_planet
        positions_star[i] = positions_star[i-1] + dt * v_star
        
        r = positions_planet[i] - positions_star[i]
        force_on_planet = gravitational_force(G,star_mass , planet_mass, r)
        force_on_star = -force_on_planet

        v_planet += 0.5 * dt * force_on_planet / planet_mass
        v_star += 0.5 * dt * force_on_star / star_mass

        velocities_planet[i] = v_planet
        velocities_star[i] = v_star

        velocities_CM[i] = (star_mass * velocities_star[i] + planet_mass * velocities_planet[i]) / (star_mass + planet_mass)
        positions_CM[i] = (star_mass * positions_star[i] + planet_mass * positions_planet[i]) / (star_mass + planet_mass)
    return positions_star, positions_planet, positions_CM, velocities_star, velocities_CM

def radial_vel(v_star, v_CM, inclination, T, dt, line_of_sight):
    N = int(T // dt)
    times = np.linspace(0, T, N)
    radial_velocities = np.zeros(N)

    for i in range(N):
        radial_velocity_star = np.dot(v_star[i], line_of_sight) * np.sin(inclination)
        radial_velocity_cm = np.dot(v_CM[i], line_of_sight)
        radial_velocities[i] = radial_velocity_star + radial_velocity_cm
    
    noise_std = 0.2 * np.max(abs(radial_velocities))
    noise = np.random.normal(0, noise_std, size=N)
    noise_radial_velocities = radial_velocities + noise

    return times, noise_radial_velocities

def compute_flux(times, positions_planet, positions_star, system, planet_idx):
    R_star = utils.km_to_AU(system.star_radius)
    R_planet = utils.km_to_AU(system.radii[planet_idx])

    flux_min = 1 - (R_planet / R_star) ** 2

    N = len(times)
    flux = np.ones(N)  

    for i in range(N):
        dy = abs(positions_planet[i, 1] - positions_star[i, 1])

        if dy >= (R_star + R_planet):
            None
        elif dy <= abs(R_star - R_planet):
            flux[i] = flux_min

    noise_std = 1e-4 * flux_min
    noise = np.random.normal(0, noise_std, size=N)
    flux = flux + noise
    return flux


def run_sim():
    G = constants.G_sol
    T = 10
    dt = 1e-4
    
    planet_idx = 0
    pos_planet = np.copy(system.initial_positions.T[planet_idx])
    pos_star = np.array([0.0,0.0])
    v_star = np.array([0.0,0.0])
    v_planet = np.copy(system.initial_velocities.T[planet_idx])
    m_star = system.star_mass
    m_planet = system.masses[planet_idx]

    positions_star, positions_planet, positions_CM, velocities_star, velocities_CM = frog_leap_studios_two_body(T, dt, pos_star, pos_planet, v_star, v_planet, m_star, m_planet, G)

    line_of_sight = np.array([1,0]) # can just be along the x-axis as changing it would be the same as changing inclination
    inclination = np.radians(9) # degrees to radians

    times, noise_radial_velocities = radial_vel(velocities_star, velocities_CM, inclination, T, dt, line_of_sight)
    plt.style.use('grayscale')

    
    Flux = compute_flux(times, positions_planet, positions_star, system, planet_idx)

    plt.figure(figsize=(10, 6))
    plt.plot(times, noise_radial_velocities, label="Observed Radial Velocity", alpha = 0.6)
    plt.xlabel("Time (years)")
    plt.ylabel("Radial Velocity (AU/year)")
    plt.title("Observed Radial Velocity of Star")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(positions_star[:, 0], positions_star[:, 1], label="Star")
    plt.plot(positions_planet[:, 0], positions_planet[:, 1], label="Planet")
    plt.plot(positions_CM[:, 0], positions_CM[:, 1], label="Center of Mass", linestyle="--",alpha=0.6)
    plt.xlabel("X (AU)")
    plt.ylabel("Y (AU)")
    plt.title("Star and Planet Orbit")
    plt.legend()
    plt.grid(True)
    plt.show()  

    plt.figure(figsize=(10, 6))
    plt.plot(times, Flux, label="Relative Flux", alpha=0.6)
    plt.xlabel("Time (years)")
    plt.ylabel("Relative Flux")
    plt.title("Light Curve of the Star")
    plt.legend()
    plt.grid(True)
    plt.show()

    return

if __name__ == "__main__":   
    run_sim()

