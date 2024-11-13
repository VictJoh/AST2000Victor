"""
This code is written without the skeleton-code
"""
from PIL import Image
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission, LandingSequence
from ast2000tools.shortcuts import SpaceMissionShortcuts

from ast2000tools import constants
import numpy as np
from matplotlib import pyplot as plt
from numba import njit, prange
from scipy import integrate
from scipy.stats import chi2
import time
import os

import matplotlib as mpl # https://pythonforthelab.com/blog/python-tip-ready-publish-matplotlib-figures/ inspiration from this
mpl.rcParams["font.size"] = 16
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.titlesize"] = 20
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["legend.fontsize"] = 15
mpl.rcParams["figure.figsize"] = (10, 6)
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.format"] = "png"  
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.formatter.useoffset"] = False
plt.rcParams["axes.formatter.use_mathtext"] = False
mpl.rcParams["axes.grid"] = True 
mpl.rcParams["grid.alpha"] = 0.3
mpl.rcParams["legend.loc"] = "upper right"
plt.style.use("grayscale")

colors = ["b", "m", "c", "y", "g", "orange", "purple"]

seed = 4042
system = SolarSystem(seed)
mission = SpaceMission(seed)
shortcut = SpaceMissionShortcuts(mission, [78257, 21784])


def calc_orbit(position, velocity, origin = None, origin_vel = None, mass = None, calc_all = False):
    """
    Calculate orbital parameters of the rocket.

    Parameters:
    - position (array): Position of rocket [AU]
    - velocity (array): Velocity of rocket [AU/yr]
    - origin (array): Origin position [AU]
    - origin_vel (array): Origin velocity [AU/yr]
    - mass (float): Mass of central body [Solar masses]
    - calc_all (bool): Whether to calculate all

    Returns:
    - a (float): Semi-major axis [AU]
    - e (float): Eccentricity
    - b (float): Semi-minor axis [AU]
    - T (float): Orbital period [yr]
    - apoapsis (float): Apoapsis distance [AU]
    - periapsis (float): Periapsis distance [AU]
    """

    if origin is None:
        origin = np.zeros_like(position)
        origin_vel = np.zeros_like(velocity)
        
    if mass is None:
        mass = system.star_mass
    mass *= constants.m_sun
    mu = constants.G * mass

    rel_pos = position - origin # position relative to origin
    rel_vel = velocity - origin_vel # vel relative to origin

    r = np.linalg.norm(rel_pos) 

    v_r = np.dot(rel_vel, rel_pos) / r # radial velocity 
    v_theta = np.linalg.norm(np.cross(rel_pos, rel_vel)) / r  # angular velocity

    h = r * v_theta # angular momentum 
    energy = (v_r**2 + v_theta**2) / 2 - (mu / r)

    a = -mu / (2 * energy) # semi-major axis
    e = np.sqrt(1 + (2 * energy * h**2) / mu**2) # eccentricity

    if calc_all == True:
        b = a * np.sqrt(1 - e**2)  # semi-minor axis
        T = 2 * np.pi * np.sqrt(a**3 / mu)  # orbital time
        apoapsis = a * (1 + e) 
        periapsis = a * (1 - e)

        return a, e, b, T, apoapsis, periapsis
    else:
        return a, e

def initiate_launch():
    mission.set_launch_parameters(thrust = 1950414.2360053714, mass_loss_rate = 313.3328015733722, initial_fuel_mass = 100000, estimated_launch_duration = 1200, launch_position = [2.79291596, 0.50094891], time_of_launch=6.23095)
    mission.launch_rocket(time_step = 0.001)

def verify_launch():
    mission.verify_launch_result(position_after_launch =  [2.79291222, 0.50100744])

def verify_orientation():
    mission.verify_manual_orientation(position_after_launch = [2.79291222, 0.50100744] , velocity_after_launch = [2.15886864, 6.07093967] , angle_after_launch = 105)

def begin_interplanetary():
    interplanetary_travel_data = np.load("interplanetary_travel_data.npz")

    t_list = interplanetary_travel_data["t_list"]
    boost_list = interplanetary_travel_data["boost_list"]

    InterplanetaryTravel = mission.begin_interplanetary_travel()
    print("Begun interplanetary travel")

    InterplanetaryTravel.verbose = False
    t = t_list[0]

    for i in range(1, len(t_list)):
        coast_time = t_list[i] - t
        InterplanetaryTravel.coast(coast_time)

        t += coast_time  

        delta_v = boost_list[i]
        InterplanetaryTravel.boost(delta_v)

    InterplanetaryTravel.verbose = True
    time_after_launch, _, _ = InterplanetaryTravel.orient()
    print(time_after_launch)
    InterplanetaryTravel.record_destination(1)

def place_in_orbit():
    time_after_launch = 11.9264
    shortcut.place_spacecraft_in_stable_orbit(time = time_after_launch, orbital_height = 5e5, orbital_angle = 0, planet_idx = 1)

def descend():
    landing_sequence = mission.begin_landing_sequence()
    print("Landing Begun")

    # t0, pos0, vel0 = landing_sequence.orient()
    # vel0_magnitude = np.linalg.norm(vel0)

    # dvel_magnitude = vel0_magnitude * 0.2 # reduce by 10 percent
    # dvel_dir = -vel0 / vel0_magnitude
    # dvel = dvel_magnitude * dvel_dir

    # delta_v = np.array([dvel[0], dvel[1], 0])

    # landing_sequence.boost(delta_v)

    total_duration = 3600 * 24  # 
    dt = 100 # every hour [s]
    times = []
    distances = []
    landing_sequence.fall(dt*10) # fall a bit to stabilize first
    landing_sequence.look_in_direction_of_planet()
    landing_sequence.start_video()
    num_steps = int(total_duration / dt)
    landing_sequence.verbose = False
    for i in range(num_steps):
        landing_sequence.fall(dt)
        t, position, velocity = landing_sequence.orient()
        distance = np.linalg.norm(position)
        times.append(t)
        distances.append(distance)
    times = np.array(times) 
    np.array(distances)
    landing_sequence.finish_video(number_of_frames = 50000)
    landing_sequence.verbose = True
    t0, pos0, v0 = landing_sequence.orient()

    plt.figure()
    plt.plot(np.array(times) / (3600*24), np.array(distances) / 1000, label='Distance to planet', color='blue', alpha=0.6)

    plt.title(f"Distance to planet")
    plt.xlabel('Time (days)')
    plt.ylabel('Distance to planet (km)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Del 6/Distance to planet.jpeg") 

    return t0, pos0, v0, landing_sequence

def cartesian_to_spherical(coords):
    x,y,z = coords
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r) if r != 0 else 0
    phi = np.arctan2(y, x) % (2 * np.pi)
    return np.array([r, theta, phi])

def spherical_to_cartesian(coords):
    r, theta, phi = coords
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def calc_coords(current_coords, t0, t1):
    time_elapsed = t1 - t0 

    current_spherical = cartesian_to_spherical(current_coords)
    r, theta, phi = current_spherical

    omega = 2 * np.pi / system.periods[1]
    phi_rotated = (phi + omega * time_elapsed) % (2 * np.pi) # rotate along phi
    new_coords_spherical = np.array([r, theta, phi_rotated]) # new coords

    new_coords_cartesian = spherical_to_cartesian(new_coords_spherical)

def scout(t0, pos0, v0, landing_sequence, num_pictures=10):
    a, e, b, T, apoapsis, periapsis = calc_orbit(position=pos0, velocity=v0, origin = None, origin_vel=None, mass = system.masses[1], calc_all=True) # origin is now planet 1

    dt_pictures = T / num_pictures
    t_list = np.linspace(0, T, num_pictures)
    if dt_pictures <= 0:
        print("dt_pictures is less than 0")
        quit()

    potential_sites_spherical = []
    landing_sequence.verbose = False
    for i, t in enumerate(t_list):
        t, pos, vel = landing_sequence.orient()
        
        potential_sites_spherical.append(cartesian_to_spherical(pos))

        landing_sequence.look_in_direction_of_planet()
        landing_sequence.take_picture(filename=f'scout_pic{i}.xml')
        landing_sequence.fall(dt_pictures)
    return potential_sites_spherical

def load_data():
    data = np.loadtxt("Del 6/spectrum_seed42_600nm_3000nm.txt")
    wavelengths = data[:, 0]
    flux = data[:, 1]
    noise_data = np.loadtxt("Del 6/sigma_noise.txt")
    noise = noise_data[:,1]
    return wavelengths, flux, noise

def max_Doppler(lambd0):
    v_max = 10 * 1e3 # m/s
    lambd0_m = lambd0 * 1e-9 # convert to m
    c = constants.c
    delta_lambd_max = (v_max / c) * lambd0_m
    return delta_lambd_max * 1e9 # back to nm

def calc_sigma(lambd0, T, m):
    lambd0_m = lambd0 * 1e-9 # convert to m
    m_kg = m * 1.66053906660e-27 # kg
    sigma = (lambd0_m / constants.c) * np.sqrt(constants.k_B * T / m_kg) # nm
    return sigma * 1e9 # back to nm

def Gaussian_line(lambd, F_min, sigma, lambd0):
    flux = 1 + (F_min - 1) * np.exp(-((lambd - lambd0)**2) / (2 * sigma**2)) # F(λ) = Fcont(λ) + (Fmin − Fcont(λ))e−(λ−λ0)2/(2σ2) where Fcont is 1 for all lambda
    return flux

def calc_chi(observed_flux, expected_flux, sigma_noise):
    chi = np.sum(((observed_flux - expected_flux) / sigma_noise) ** 2)
    return chi

def fit_spectral_line(wavelengths, noise, flux, lambd0, m):
    n = 200 # is used
    F_min_vals = np.linspace(0.60, 1, n)
    T_vals = np.linspace(100, 1000, n)

    max_dlambd = max_Doppler(lambd0)
    delta_lambd_vals = np.linspace(-max_dlambd, max_dlambd, n)

    
    max_sigma = 0.01 # nm, just an empirical estimate to get some leeway
    valid_mask = (wavelengths >= lambd0 - max_dlambd - max_sigma) & (wavelengths <= lambd0 + max_dlambd + max_sigma)
    valid_lambds = wavelengths[valid_mask]
    valid_flux = flux[valid_mask]
    valid_noise = noise[valid_mask]
    
    min_chi = np.inf
    best_params = None

    for F_min in F_min_vals:
        for T in T_vals:
            sigma = calc_sigma(lambd0, T, m)
            for delta_lambd in delta_lambd_vals:
                shifted_lambd0 = lambd0 + delta_lambd
                expected_flux = Gaussian_line(valid_lambds, F_min, sigma, shifted_lambd0)
                chi = calc_chi(valid_flux, expected_flux, valid_noise)
                if chi < min_chi:
                    min_chi = chi
                    dof = len(valid_flux) - 3  # degrees of freedom (data points - parameters)
                    chi_reduced = chi / dof
                    p_value = chi2.sf(min_chi, dof)
                    best_params = {
                        "F_min": F_min,
                        "T": T,
                        "delta_lambd": delta_lambd,
                        "chi": chi,
                        "chi_reduced": chi_reduced,
                        "p_value": p_value,
                        "sigma": sigma,
                        "shifted_lambd0": shifted_lambd0,
                        "valid_lambds": valid_lambds,
                        "expected_flux": expected_flux,
                        "dof": dof
                    }
    return best_params

def find_fits(molecules, wavelengths, flux, noise):
    results = []

    for i in molecules:
        m = molecules[i]["mass"]
        for lambd0 in molecules[i]["spectral_lines"]:
            fit_result = fit_spectral_line(wavelengths, noise, flux, lambd0, m)
            fit_result.update({"molecule": i, "lambd0": lambd0})
            results.append(fit_result)
    return results

def find_molecules_in_atmosphere(fits):
    """Maybe use"""
    tol = 0.1

    T_min, T_max = 150, 450  
    p_threshold = 0.05
    chi_reduced_min, chi_reduced_max = 0.8, 1.2
    F_min_threshold = 0.95  

    return # molecules_in_atmosphere

def plot_fit(wavelengths, flux, noise, result):
    molecule = result["molecule"]
    lambd0 = result["lambd0"]
    shifted_lambd0 = result["shifted_lambd0"]
    F_min = result["F_min"]
    T = result["T"]
    sigma = result["sigma"]
    chi = result["chi"]
    valid_lambds = result["valid_lambds"]
    expected_flux = result["expected_flux"] 

    plt.figure()
    plt.plot(wavelengths, flux, label='Observed Flux', color='blue', alpha=0.6)
    plt.plot(valid_lambds, expected_flux, label='Model', color='red', linestyle='--')
    plt.axvline(x=lambd0, color='green', linestyle=':', label='Original lambda0')
    plt.axvline(x=shifted_lambd0, color='purple', linestyle=':', label='Shifted lambda0')

    plt.title(f"{molecule}, {lambd0} nm, Chi^2: {chi:.2f}")
    plt.xlim(valid_lambds[0], valid_lambds[-1])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Del 6/fit{molecule}_{(lambd0)}nm.jpeg")
    plt.close()
    
def plot_atmosphere(molecules, molecules_in_atmosphere, plot_T = True, plot_rho = True):
    mu = np.mean([molecules[molecule]["mass"] for molecule in molecules_in_atmosphere])

    mass = system.masses[1]
    R = system.radii[1]
    G = constants.G
    g = (G*mass) / R**2

    gamma = 1.4    
    m_H = constants.m_H2
    k = constants.k_B  

    T0 = 291.95 # K surface temperature on planet 1         
    rho0 = system.atmospheric_densities[1] #kg/m^3

    a = gamma / (gamma - 1)  

    b = (mu * m_H * g * (gamma - 1)) / (gamma * k)  
    c = T0 / 2                                   
    d = (2 * mu * m_H * g) / (k * T0)            


    h_a = (gamma * k * T0) / (2 * mu * m_H * g * (gamma - 1))  
    h_a = h_a  
    
    h_list = np.linspace(0, h_a*2, 1000) 
    T_list = np.zeros_like(h_list)
    rho_list = np.zeros_like(h_list)

    adiabatic_mask = (h_list <= h_a)
    isothermal_mask = (h_list >h_a)

    T_list[adiabatic_mask] = T0 - b * h_list[adiabatic_mask]
    T_list[isothermal_mask] = c

    rho_list[adiabatic_mask] = rho0 * (T0 - b * h_list[adiabatic_mask])**a
    rho_list[isothermal_mask] = rho0 * c**a * np.exp(-d * (h_list[isothermal_mask]   - h_a))
    
    h_list /= 1000 # turn into km
    h_a /= 1000 # --||--
    if plot_T == True:
        plt.figure()
        plt.plot(h_list / 1000, T_list, label='Temperature', color='red')
        plt.axvline(x=h_a / 1000, color='blue', linestyle='--', label='h_a (Adiabatic end)')
        plt.title('Temperature vs Height')
        plt.xlabel('Height (km)')
        plt.ylabel('Temperature (K)')
        plt.legend()
        plt.tight_layout()
        plt.savefig("Temperature_Profile.png")
        plt.show()
    if plot_rho == True:
        plt.figure()
        plt.plot(h_list / 1000, rho_list, label='Density', color='green')
        plt.axvline(x=h_a / 1000, color='blue', linestyle='--', label='h_a (Adiabatic end)')
        plt.title('Density Profile vs Height')
        plt.xlabel('Height (km)')
        plt.ylabel('Density (kg/m³)')
        plt.legend()
        plt.tight_layout()
        plt.savefig("Density_Profile.png")
        plt.show()

def main():
    
    # initiate_launch()
    # verify_launch()
    # verify_orientation()
    # # begin_interplanetary() # not used as I got the nescessary info
    # InterplanetaryTravel = mission.begin_interplanetary_travel()
    # InterplanetaryTravel.record_destination(1)
    # place_in_orbit()
    # t0, pos0, v0, landing_sequence = descend()
    # scout(t0, pos0, v0, landing_sequence, num_pictures=10)
    
    molecules = {
    "O2": {"spectral_lines": [632, 690, 760], "mass": 32},
    "H2O": {"spectral_lines": [720, 820, 940], "mass": 18},
    "CO2": {"spectral_lines": [1400, 1600], "mass": 44},
    "CH4": {"spectral_lines": [1660, 2200], "mass": 16},
    "CO": {"spectral_lines": [2340], "mass": 28},
    "N2O": {"spectral_lines": [2870], "mass": 44}
    }

    wavelengths, flux, noise = load_data()
    fits = find_fits(molecules, wavelengths, flux, noise)


    for result in fits:
        print(
            f"{result['molecule']}, Original lambd0: {result['lambd0']} nm, "
            f"Shifted lambd0: {result['shifted_lambd0']:.3f} nm, "
            f"Doppler Shift = {result['lambd0'] - result['shifted_lambd0']:.3f} nm, "
            f"F_min: {result['F_min']:.3f}, T: {result['T']} K, "
            f"Sigma: {result['sigma']:.3f} nm, "
            f"Chi^2: {result['chi']:.3f} "
            f"Reduced Chi^2: {result['chi_reduced']:.3f}, "
            f"p-value: {result['p_value']:.3f}"
        )
        plot_fit(wavelengths, flux, noise, result)
    molecules_in_atmosphere = ["O2", "CO"]
    plot_atmosphere(molecules, molecules_in_atmosphere)

if __name__ == "__main__":
    main()
    