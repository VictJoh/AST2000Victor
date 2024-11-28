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
    InterplanetaryTravel.record_destination(1)

def place_in_orbit():
    time_after_launch = 11.9264
    shortcut.place_spacecraft_in_stable_orbit(time = time_after_launch, orbital_height = 5e5, orbital_angle = 0, planet_idx = 1) # even though my orbit was circular as shown in part 5 it wasn"t close enough :(

def descend():
    global landing_sequence 
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
    # landing_sequence.start_video()
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
    # landing_sequence.finish_video(number_of_frames = 50000)
    landing_sequence.verbose = True
    t0, pos0, v0 = landing_sequence.orient()

    # plt.figure()
    # plt.plot(np.array(times) / (3600), np.array(distances) / 1000, label="Distance to planet", color="blue", alpha=0.6)

    # plt.title(f"Distance to planet")
    # plt.xlabel("Time (hours)")
    # plt.ylabel("Distance to planet (km)")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"Del 6/Distance to planet.jpeg") 

    return t0, pos0, v0

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

def calc_coords(coords0, t0, t1):
    time_elapsed = t1 - t0 

    coords0_spherical = cartesian_to_spherical(coords0)
    r, theta, phi = coords0_spherical

    omega = 2 * np.pi / (system.rotational_periods[1] * (24 * 3600)) # [1/s]
    phi_rotated = (phi + omega * time_elapsed) % (2 * np.pi) # rotate along phi
    new_coords_spherical = np.array([r, theta, phi_rotated]) # new coords

    new_coords_cartesian = spherical_to_cartesian(new_coords_spherical)
    if np.abs(np.linalg.norm(new_coords_cartesian) - np.linalg.norm(coords0)) > 10:
        print(f"coord transformation deviated by {np.linalg.norm(new_coords_cartesian) - np.linalg.norm(coords0)} m")
    return new_coords_cartesian

def scout(t0, pos0, v0, num_pictures, take_pictures = False):
    global landing_sequence
    a, e, b, T, apoapsis, periapsis = calc_orbit(position=pos0, velocity=v0, origin = None, origin_vel=None, mass = system.masses[1], calc_all=True) # origin is now planet 1
    dt_pictures = T / num_pictures
    t_list = np.linspace(0, T, num_pictures)
    if dt_pictures <= 0:
        print("dt_pictures is less than 0")
        quit()

    potential_sites_cartesian = np.zeros((len(t_list), 3))
    landing_sequence.verbose = False
    for i, t in enumerate(t_list):
        t, pos, vel = landing_sequence.orient()
        potential_sites_cartesian[i] = pos

        landing_sequence.look_in_direction_of_planet()
        if take_pictures:   
            landing_sequence.take_picture(filename=f"scout_pic{i}.xml")
        landing_sequence.fall(dt_pictures)
    return t_list, potential_sites_cartesian

def calc_density(h):
    mu = 30.0 # CO2 and CH4 [u]

    M = system.masses[1] * constants.m_sun
    R = system.radii[1] * 1e3 # m
    G = constants.G # m^3kg^-1s^-2
    g = (G*M) / R**2 # [m/s^2]

    gamma = 1.4    
    m_H = constants.m_p # [kg]
    k = constants.k_B  # [(m^2 kg) / (s^2 K)]

    T0 = 291.95 # [K] surface temperature on planet 1         
    rho0 = system.atmospheric_densities[1] # [kg/m^3]

    a = 1 / (gamma - 1)  

    b = (mu * m_H * g * (gamma - 1)) / (gamma * k) # K/m                               
    c = (2 * mu * m_H * g) / (T0 * k)  # 1/m

    rho_a = rho0 * 0.5**(a)

    h_a = T0 / (2*b) # adiabatic end
    
    if h < h_a:
        rho = rho0 * (1 - (b/T0) * h)**a
    else:
        rho = rho_a * np.exp(-c * (h - h_a))
    return rho

def calc_parachute_area(C_d, m, g, rho, v_t):
    area = (2*m*g) / (C_d * rho*(v_t**2))
    return area

def landing(landing_pos0, picture_time, simulation_time, dt = 1e-3):
    global landing_sequence
    global t_launch_lander
    global F_L
    global delta_v
    


    N = int(simulation_time // dt)

    G = constants.G
    C_d = 1
    v_safe = 3 # m/s

    M = system.masses[1] * constants.m_sun # [kg]  
    R = system.radii[1] * 1e3  # [m]
    rho0 = system.atmospheric_densities[1]
    Omega = 2 * np.pi / system.rotational_periods[1] / (24 * 3600) # [s^-1]

    t_launch_lander = 2500 # [s] tested to hit location
    h_parachute = 10000 # [m]

    t0, pos0, vel0 = landing_sequence.orient()
    m = mission.spacecraft_mass
    lander_area = mission.spacecraft_area

    
    pos = np.array(pos0)
    v = np.array(vel0)
    t = t0.copy()

    times = []
    positions = []
    velocities = []
    heights = []
    accelerations = []


    F_L = np.zeros(3)
    A = lander_area
    parachute_deployed = False
    parachute_broken = False
    lander_launched = False
    landing_thruster_activated = 0
    for i in range(N):
        r = np.linalg.norm(pos)
        h = r - R
        r_dir = pos / np.linalg.norm(pos)
        v_r = np.dot(v, r_dir)

        if h <= 0:
            if np.abs(v_r) >= v_safe:
                print(f"lander has crashed at time {t} s with radial velocity {v_r} m /s")
            else:
                print(f"lander has landed at time {t} with radial velocity {v_r} m/s")
            break

        if not parachute_deployed and h <= h_parachute:
            g = np.linalg.norm(F_g) / m
            v_t = 5 # something that worked
            parachute_area = calc_parachute_area(C_d, m, g, rho0, v_t)
            print(f"deployed parachute with area: {parachute_area}")
            t_parachute = t
            parachute_deployed = True
            A += parachute_area


        rho = calc_density(h)
        w = np.array([-Omega * pos[1], Omega * pos[0], 0.0])
        v_drag = v - w

        v_drag_dir = v_drag / np.linalg.norm(v_drag)
        F_d = 0.5 * rho * C_d * A * np.linalg.norm(v_drag)**2 * (-v_drag_dir)

        # print(F_d)
        F_g = (G * M * m) / r**2 * (-pos / r) 

        if lander_launched and np.linalg.norm(F_d) > 250000 and parachute_deployed and not parachute_broken:
            parachute_broken = True
            A = lander_area
            print(f"parachute broke after being deployed for {t - t_parachute} s at a F_d = {np.linalg.norm(F_d)} N ")
        
        if not landing_thruster_activated and h <= 100:
            v_t = np.sqrt((2 * np.linalg.norm(F_g)) / (rho0 * A * C_d))
            F_L_dir = r_dir
            F_L = 0.5 * rho0 * C_d * A * (v_t**2 - v_safe**2) * F_L_dir
            landing_thruster_activated = True
            print(f"performed boost F_L = {np.linalg.norm(F_L)} at time {t} s")
            
        F_tot = F_g + F_d + F_L

        a = F_tot / m

        v += a*dt
        pos += v*dt
        
        P_d = np.linalg.norm(F_d / A)
        if P_d > 1e7 and lander_launched:
            print(f"lander has burnt up after {t - t0} s at height {h} m")
            break

        if not lander_launched and t - t0 > t_launch_lander:
            # boost against velocity direction to slow down

            t_lander = t
            lander_launched = True
            A = mission.lander_area
            m = mission.lander_mass

            delta_v = -v / 40 # reduce by 1/40
            print(v, delta_v)
            v += delta_v
            print(f"launched lander after {i} steps with velocity {delta_v} m/s")

        # if i % 10000 == 0:
        #     print(f"Step {i}: pos = {pos}, vel = {v}, acc = {a}, rho = {rho}, F_g = {F_g}, F_d = {F_d}")
        t += dt
        times.append(t)
        positions.append(pos.copy())
   
        velocities.append(v.copy())
        accelerations.append(a.copy())
        heights.append(h)


        # if np.linalg.norm(a) >= 30:
        #     print(f"acceleration bigger than 30 at {h} m")
    

    accelerations = np.array(accelerations.copy())
    positions = np.array(positions.copy())
    velocities = np.array(velocities.copy())
    times = np.array(times.copy())
    heights = np.array(heights.copy())


    end_time = times[-1]
    
    end_landing_site_pos = calc_coords(landing_pos0, picture_time, end_time) 

    end_pos, end_vel, end_t = positions[-1], velocities[-1], times[-1]
    print(end_pos, end_vel, end_t)

    print(f"deviated by {np.linalg.norm(end_landing_site_pos - end_pos)} m from landing site")

    return positions, velocities, times, accelerations, heights, t_lander, end_landing_site_pos

def plot_positions(positions, end_landing_site_pos, times):
    global t_launch_lander
    idx_launch_lander = np.argmin(np.abs(t_launch_lander - times))
    end_landing_site_pos /= 1000 # [km]
    x = positions[:, 0] / 1000   # [km]
    y = positions[:, 1] / 1000 # [km]
    
    x_launch = x[idx_launch_lander]
    y_launch = y[idx_launch_lander]

    R = system.radii[1]

    plt.figure()

    planet = plt.Circle((0, 0), R, color="green", alpha=0.8, label="Planet")
    plt.gca().add_artist(planet)

    plt.scatter(end_landing_site_pos[0], end_landing_site_pos[1], label = "Landing Site", color = "purple", s = 3)
    plt.scatter(x_launch, y_launch, label = "Launched Lander", color = "black", s = 1)

    plt.plot(x, y, label="Lander Trajectory", color = "red", linestyle = "--")
    plt.title("Lander Trajectory")
    plt.xlabel("x [km]")
    plt.ylabel("y [km]")
    plt.legend(loc="upper left")
    plt.axis("equal")
    plt.savefig("Del7/new_Lander_trajectory.png")

def plot_velocities(velocities, positions, times):
    velocities = np.array(velocities)
    positions = np.array(positions)

    radial_velocities = np.sum(velocities * positions, axis=1) / np.linalg.norm(positions, axis=1)
    tangential_velocities = np.sqrt(np.linalg.norm(velocities, axis=1)**2 - radial_velocities**2)

    # Radial velocity
    plt.figure()
    plt.plot(times, radial_velocities, label="Radial Velocity", color="blue")
    plt.title("Radial Velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("Radial Velocity [m/s]")
    plt.savefig("Del7/new_radial_velocity.png")


    # Tangential velocity
    plt.figure()
    plt.plot(times, tangential_velocities, label="Tangential Velocity", color="green")
    plt.title("Tangential Velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("Tangential Velocity [m/s]")
    plt.savefig("Del7/new_tangential_velocity.png")

def plot_acceleration(accelerations, times):
    accelerations = np.linalg.norm(accelerations, axis=1)
    plt.figure()
    plt.plot(times, accelerations, label="Acceleration over time", color="red")

    plt.title("Acceleration over time")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s^2]")
    plt.legend()
    plt.savefig("Del7/new_Acceleration_over_time.png")

def plot_height(heights,times):
    heights /= 1000 #[km]
    plt.figure()
    plt.plot(times, heights, label="Height over time", color="red")

    plt.title("Height over time")
    plt.xlabel("Time [s]")
    plt.ylabel("Height [km]")
    plt.legend()
    plt.savefig("Del7/new_heght_over_time.png")

def landing_ast(landing_pos0, picture_time, simulation_time, dt = 1.0, video = True):
    global landing_sequence
    global t_launch_lander
    global F_L
    global delta_v

    t0, pos0, v0 = landing_sequence.orient()

    N = int(simulation_time // dt)

    R = system.radii[1]* 1000

    h_parachute = 10000
    
    times = []
    positions = []
    velocities = []
    accelerations = []

    landing_sequence.verbose = False

    landing_sequence.adjust_parachute_area(area = 70.46758715956261)
    landing_sequence.adjust_landing_thruster(np.linalg.norm(F_L), 90) # adjusted force as I got stuck in the air, actual: 


    for i in range(N):
        t, pos, vel = landing_sequence.orient()
        h = np.linalg.norm(pos) - R        

        if not landing_sequence.lander_launched and t - t0 >= t_launch_lander:
            landing_sequence.verbose = True
            landing_sequence.launch_lander(delta_v)
            landing_sequence.verbose = False
            if video:
                landing_sequence.start_video() 

        if not landing_sequence.parachute_deployed and h <= h_parachute:
            landing_sequence.verbose = True
            landing_sequence.deploy_parachute()
            landing_sequence.verbose = False

        if not landing_sequence.landing_thruster_activated and h <= 100:
                landing_sequence.verbose = True
                landing_sequence.activate_landing_thruster()
                landing_sequence.verbose = False
                break

        times.append(t)
        positions.append(pos.copy())
        velocities.append(vel.copy())

        landing_sequence.fall(dt*10) 

    if landing_sequence.landing_thruster_activated:
        landing_sequence.verbose = True
        while landing_sequence.reached_surface == False and i < 10000:
            t, pos, vel = landing_sequence.orient()
            landing_sequence.fall(dt*10)
            times.append(t)
            positions.append(pos.copy())
            velocities.append(vel.copy())

    t_end, pos_end, vel_end = landing_sequence.orient()
    end_landing_site_pos = calc_coords(landing_pos0, picture_time, t_end) 
    print(f"deviated by {np.linalg.norm(end_landing_site_pos - pos_end)} m from landing site")
    print(t_end, pos_end, vel_end)
    if video:
        landing_sequence.finish_video()

def main():
    initiate_launch()
    verify_launch()
    verify_orientation()
    # begin_interplanetary() # not used as I got the nescessary info
    InterplanetaryTravel = mission.begin_interplanetary_travel()
    InterplanetaryTravel.record_destination(1)
    place_in_orbit()
    t0, pos0, v0 = descend()
    t_list, pos_list = scout(t0, pos0, v0, num_pictures=20, take_pictures = False)
    landing_site = pos_list[13]
    picture_time = t_list[13]

    # projection to the surface
    landing_site_spherical = cartesian_to_spherical(landing_site)
    landing_site_spherical[0] = system.radii[1] * 1000 # [m]
    landing_site = spherical_to_cartesian(landing_site_spherical)

    simulation_time = 3600*3 # [s]
    positions, velocities, times, accelerations, heights, t_lander, end_landing_site_pos = landing(landing_site, picture_time, simulation_time)
    plot_positions(positions, end_landing_site_pos, times)
    plot_height(heights, times)
    plot_velocities(velocities, positions, times)
    plot_acceleration(accelerations, times)
    landing_ast(landing_site, picture_time, simulation_time, dt = 1.0, video = True)
    plt.show()



if __name__ == "__main__":
    main()
    