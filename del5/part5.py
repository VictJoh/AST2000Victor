"""
This code is written without the skeleton-code
"""

from PIL import Image
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools.shortcuts import SpaceMissionShortcuts
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
mpl.rcParams['axes.grid'] = True 
mpl.rcParams['grid.alpha'] = 0.3
plt.style.use('grayscale')
colors = ['b', 'm', 'c', 'y', 'g', 'orange', 'purple']

seed = 4042
system = SolarSystem(seed)
mission = SpaceMission(seed)
shortcut = SpaceMissionShortcuts(mission, [78257])

file_path = "C:/Users/victo/Documents/GitHub/AST2000Victor"



class RocketEngine:
    def __init__(self, seed, N, engine_simulation_time, dt, L, T):
        self.seed = seed
        self.N = N
        self.dt = dt
        self.L = L
        self.T = T
        self.engine_simulation_time = engine_simulation_time
        self.k = constants.k_B
        self.m = constants.m_H2
    
        np.random.seed(seed)

    def gen_rand_pos(self):
        """
        Generate random coordinates within a cube for N particles. The particles are distributed uniformly.
    

        Returns:
        (x, y, z) (array) = x,y,z coordinates of particles in cube as a N*3 array [m]
        """
        return np.random.uniform(low=0, high=self.L, size=(int(self.N), 3))   
     
    def gen_mb_vel(self):
        """
        Generates velocities for N particles in each direction according to the Maxwell-Bolzmann distribution.

        Returns: 
        abs_v (array) = absolute value of velocity [m/s]
        vx, vy, vz (arrays) = velocity in x,y,z direction as N*3 array [m/s]
        """
        std = np.sqrt(self.k * self.T / self.m) # Standard deviation of the Maxwell-Boltzmann distribution
        return np.random.normal(loc=0, scale=std, size = (int(self.N), 3))
    
    def check_collision(self, pos, vel):   
        """
        Adjusts the velocity of a particle when it hits a wall of the cube.

        If the particle's position exceeds the boundaries, it resets the positions and reverses the direction.
        We assume perfect elasiticity
        
        Parameters:
        pos (array): The current positions [x, y, z] of the particle [m].  
        vel (array): The current velocities [vx, vy, vz] of the particle [m/s].

        Returns:
        pos (array): Updated positions in x,y,z coordinates [m]
        vel (array): Updated velocities in x,y,z direction [m/s]
        indice_particles_at_thrust (array): returns the indices of the positions of particles escaping the rocket as True
        """
        indice_particles_at_bottom = pos <= 0 # Gives the location in the position matrix of values below 0 as True and False otherwise
        pos[indice_particles_at_bottom] = 0 # Sets all values that are True to 0 (back to the edge)
        vel[indice_particles_at_bottom] *= -1 # Changes the velocity in the direction of collision to the other direction
        # The same as above but values above L
        indice_particles_at_top = pos >= self.L 
        pos[indice_particles_at_top] = self.L
        vel[indice_particles_at_top] *= -1 

        particles_z0 = pos[:, 2] <= 0 # Gives index of particles with z-axis below 0 as True
        particles_x_thrust = (pos[:, 0] > 0.25 * self.L) & (pos[:, 0] < 0.75 * self.L) # Gives index of particles with x position between 0.25L and 0.75L as True (where the hole is)
        particles_y_thrust = (pos[:, 1] > 0.25 * self.L) & (pos[:, 1] < 0.75 * self.L) # --||-- y_position

        indice_particles_at_thrust = (particles_z0 & particles_x_thrust & particles_y_thrust) # checks iff all three are true for each particle

        return pos, vel, indice_particles_at_thrust
    
    def calc_force_exerted(self, indice_particles_at_thrust, velocities):
        """
        Calculates the force exerted from the particles that have escaped

        Parameters:
        indice_particles_at_thrust (array): Indices of the particles that are at the thrust
        velocities (array) : The velocity-matrix with all velocities in x,y,z
        
        Returns:
        dF (float) : The total force exerted on the rocket
        """
        z_velocities_escaping =  abs(velocities[indice_particles_at_thrust, 2]) # makes a boolean array of all the z-velocities of the escaping particles
        dp = np.sum(self.m * z_velocities_escaping) # gives the change of momentum
        dF = dp / self.dt # calculates the force exerted
        return dF 
    
    def calc_fuel_consumption(self, indice_particles_at_thrust):
        """
        Calculates the fuel consumption of the rocket

        Parameters:
        indice_particles_at_thrust (array): Indices of the particles that are at the thrust
        m (float): The mass of the H2-particle [kg]

        Returns: 
        consumption (float) : The total amount of fuel consumed by the rocket
        """
        consumption = (np.count_nonzero(indice_particles_at_thrust) * self.m) # amount of particles that are at the the thrust multiplied with their mass
        return consumption

    def validate_mean_speed(self, v_means):
        """
        Compares simulated speed to the expected 
        mean speed from the Maxwell-Boltzmann distribution.

        Parameters:
        v_means (array) : The mean velocities of the system during each timestep 

        Returns:
        None
        """
        v_mean_exp = np.sqrt(8 * self.k * self.T / (np.pi * self.m)) #the expected mean from MB-distribution
        total_v_mean = np.mean(v_means) # Takes the mean of all the means from each time step
        alpha = 0.05 * v_mean_exp # The relative deviation we accept
        deviation = abs(v_mean_exp - total_v_mean) # Our absolut deviation
        if deviation <= alpha:
            print("The simulated mean speed is within the wanted margin")
        else:
            print(f"The simulated mean speed deviated by {deviation} m/s from the expected.")

    def plot_maxwell_boltzmann_comparison(self, all_v_norms):
        """
        Plots the distribution of particle speeds as a histogram with Maxwell-Boltzmann.

        Parameters:
        all_v_norms (array): All the speeds of particles from the simulation [m/s].

        Returns:
        None
        """
        max_speed = np.max(all_v_norms) #takes the max speed
        speeds = np.linspace(0, max_speed, 10000) # makes the x-axis from 0 to max speed with 10000 spaces
        factor = 4 * np.pi * (self.m / (2 * np.pi * self.k * self.T))**(3/2) # constants in the equation
        exponent = np.exp(-self.m * speeds**2 / (2 * self.k * self.T)) # the exp-part
        mb_distribution = factor * speeds**2 * exponent # Maxwell-Boltzmann distribution gives expected number per velocity

        plt.figure(figsize=(8, 6), dpi=240) # We set the size to 8,6 as this best fits in our latex-file and resolution to 240 as this looked good

        plt.hist(all_v_norms.flatten(), bins=50, density=True, color='gray', label='Simulated Speeds') #Makes histogram with 50 blocks from our v_norms we use .flatten as we want it to be a 1D array

        plt.plot(speeds, mb_distribution, color='black', linewidth=2, linestyle='--', label='Maxwell-Boltzmann Distribution')

        plt.xlabel('Speed (m/s)', fontsize=14)
        plt.ylabel('Probability Density', fontsize=14)
        plt.title('Simulated Speeds vs Maxwell-Boltzmann Distribution', fontsize=18)

        plt.grid(True, color='gray', linestyle='--')

        v_median = np.sqrt(2 * self.k * self.T / self.m) # the expected mean speed
        plt.axvline(x=v_median, color='black', linestyle=':', linewidth=2) # makes a dotted line at the mean speed
        plt.text(v_median, max(mb_distribution), 'Most Probable Speed', color='black', fontsize=12) # makes text at the dotted line

        plt.legend(fontsize=12, loc='upper right')  

        plt.tight_layout() # makes it look better we think. To be honest we just learned to do this no matter what
        plt.savefig('maxwell_boltzmann_comparison_grayscale.png', format='png', dpi=240)
        plt.show()

    def run_engine(self):
        """
        Runs the engine for the given duration. It gives us the consumption and the average force we can expect. 

        Returns:
        avg_F (float) = The average force exerted [N]
        consumption (float) = How much fuel we use [kg/s]
        """
        positions = self.gen_rand_pos()
        velocities = self.gen_mb_vel()
        t_list = np.arange(0,self.engine_simulation_time, self.dt)
        F = 0
        consumed = 0
        v_means = []
        all_v_norms = np.zeros((len(t_list), self.N))
        for t in range(len(t_list)):
            positions, velocities, indice_particles_at_thrust = self.check_collision(positions, velocities) 
            positions += velocities * self.dt  

            v_norms = np.linalg.norm(velocities, axis=1)
            v_means.append(np.mean(v_norms))
            all_v_norms[t] = v_norms
            
            dF = self.calc_force_exerted(indice_particles_at_thrust, velocities)
            consumed += self.calc_fuel_consumption(indice_particles_at_thrust)
            F += dF

        # self.plot_maxwell_boltzmann_comparison(all_v_norms)
        self.validate_mean_speed(v_means)
        avg_F = F / len(t_list)
        consumption = consumed / self.engine_simulation_time
        return avg_F, consumption   

def interpolate(times, positions_over_time, planet_idx, t, velocities_over_time = None):
    """
    Interpolates the position (and optionally velocity) at a given time. Works for all units.

    Parameters:
    times (array): Time points corresponding to positions
    positions_over_time (array): Positions of planets over time (N, num_of_planets,2) 
    planet_idx (int): Index of the planet to interpolate 
    t (float): The time to interpolate at
    velocities_over_time (array): Velocities of planets over time (N, num_of_planets,2) 

    Returns:
    position (array): Interpolated position (x,y) 
    velocity (array): Interpolated velocity (vx, vy) 
    """
    x_positions = positions_over_time[:, planet_idx, 0]
    y_positions = positions_over_time[:, planet_idx, 1]


    x = np.interp(t, times, x_positions)
    y = np.interp(t, times, y_positions)
    if velocities_over_time is not None:
        vx_over_time = velocities_over_time[:, planet_idx, 0]
        vy_over_time = velocities_over_time[:, planet_idx, 1]
        vx = np.interp(t, times, vx_over_time)
        vy = np.interp(t, times, vy_over_time)
        return np.array([x, y]), np.array([vx, vy])
    else:
        return np.array([x, y])

def interpolate_vectorized(times, positions_over_time, new_times, velocities_over_time = None):
    """
    Interpolate planet positions and optionally velocities to new time list new_times. Works for all units.

    Parameters:
    times (array): time list for the positions_over_time
    positions_over_time (array): array of size (N,num_of_planets,2) with original positions
    new_times (array): array containing new times
    velocities_over_time (array): Velocities of planets over time (N, num_of_planets,2) 

    Returns:
    interpolated_positions (array): Interpolated positions
    interpolated_velocities (array, optional): Interpolated velocities 
    """

    x_positions = positions_over_time[:, :, 0].T  # (num_of_planets, N)
    y_positions = positions_over_time[:, :, 1].T  # (num_of_planets, N)


    interpolated_x = np.array([np.interp(new_times, times, x) for x in x_positions])
    interpolated_y = np.array([np.interp(new_times, times, y) for y in y_positions])

    interpolated_x = interpolated_x.T  
    interpolated_y = interpolated_y.T  

    interpolated_positions = np.stack([interpolated_x, interpolated_y], axis=-1) # (N, num_of_planets, 2)

    if velocities_over_time is not None:
        vx_over_time = velocities_over_time[:, :, 0].T
        vy_over_time = velocities_over_time[:, :, 1].T

        interpolated_vx = np.array([np.interp(new_times, times, vx) for vx in vx_over_time])
        interpolated_vy = np.array([np.interp(new_times, times, vy) for vy in vy_over_time])

        interpolated_vx = interpolated_vx.T  
        interpolated_vy = interpolated_vy.T  

        interpolated_velocities = np.stack([interpolated_vx, interpolated_vy], axis=-1) # (N, num_of_planets, 2)

        return interpolated_positions, interpolated_velocities
    else:
        return interpolated_positions

def rotate_vector(vec, angle):
    """
    Rotates a 2D vector counter-clockwise by an angle.

    Parameters:
    vec (array): The vector to rotate.
    angle (float): The angle (radians)

    Returns:
    rotated_vec (array): Rotated vector
    """
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)]])
    return rotation_matrix @ vec
    
class Rocket:
    def __init__(self, seed, F, consumption, fuel_mass, number_of_engines, rocket_duration, dt, planet_idx, t_launch = 0.0, angle_launch = 0.0):
        # rocket engine
        self.seed = seed
        self.dt = dt
        self.initial_fuel_mass = fuel_mass
        self.fuel_mass = fuel_mass
        self.rocket_duration = rocket_duration
        self.F = F * number_of_engines
        self.consumption = consumption * number_of_engines
        self.G = constants.G
        self.total_fuel_consumed = 0

        # planet
        self.planet_idx = planet_idx
        self.planet_mass = system.masses[planet_idx] * constants.m_sun # [kg]
        self.planet_radius = system.radii[planet_idx] * 1000 # [m]
        self.planet_rotation_period = system.rotational_periods[planet_idx] * constants.day # [s]
        
        pos_data = np.load('planet_positions.npz')
        self.positions_over_time = pos_data['positions_over_time']
        self.times = pos_data['times']

        self.t_launch = t_launch * constants.yr # [s]
        self.t_launch_years = t_launch # [yrs]


        self.planet_pos = interpolate(self.times, self.positions_over_time, planet_idx, self.t_launch_years) * constants.AU  # [m]
       
        planet_angle = np.arctan2(self.planet_pos[1], self.planet_pos[0]) # radians
        self.planet_direction= np.array([np.cos(planet_angle), np.sin(planet_angle)]) # planet's direction according to satr
        self.relative_direction = rotate_vector(self.planet_direction, angle_launch) # rocket's direction relative to planet
    
        self.rocket_pos = self.planet_pos + self.planet_radius * self.relative_direction # [m]
        planet_rotation_speed = 2 * np.pi * self.planet_radius / self.planet_rotation_period # [m/s]

        v_rot_direction = np.array([-self.relative_direction[1], self.relative_direction[0]]) # rotational direction (-y,x)
        v_rot = planet_rotation_speed * v_rot_direction # [m/s]

        self.rocket_v = v_rot 

        self.m_rocket = self.fuel_mass + mission.spacecraft_mass #[kg]
        
        self.rocket_initial_pos = np.copy(self.rocket_pos)


    def gravity(self, r_vec): 
        """
        Calculates the gravity force on the rocket

        Parameters:
        r_vec (array): Vector from the planet to the rocket [m]

        Returns:
        a_g (array) : the gravity force on the rocket [m/s^2]
        """

        r_norm = np.linalg.norm(r_vec)
        a_g = -self.G * self.planet_mass / r_norm**3 * r_vec
        return a_g
    
    def U_r(self, r_vec):
        """
        Calculates the potential energy of the rocket 

        Returns:
        The potential energy of the rocket (float) [J] 
        """
        return - (constants.G * self.planet_mass * self.m_rocket) / np.linalg.norm(r_vec)
    
    def T(self):
        """
        Calculates the kinetic energy of the rocket 

        Parameters:
        m_rocket (float) : the mass of the rocket [kg]

        Returns:
        The kinetic energy (float) [J]
        """
        return 0.5 * self.m_rocket * np.linalg.norm(self.rocket_v)**2


    def run(self):
        """
        Runs the rocket and calculates all the end values after the rocket launch

        Returns:
            rocket_positions (array): The positions of the rocket over time (m).
            rocket_velocity (array): The velocities of the rocket over time (m/s).
            total_fuel_consumed (float): Total fuel consumed (kg).
            reached_escape_vel (float or None): Time when escape velocity was reached (s), if achieved.
        
        """
        time_list = np.arange(0, self.rocket_duration, self.dt)
        fuel_consumed = self.consumption * self.dt
        a_r = 0
        fuel_complete_time = None
        self.reached_escape_vel = None
        self.rocket_positions = np.zeros((len(time_list), 2)) 
        self.rocket_velocities = np.zeros((len(time_list), 2)) 
        self.rocket_positions[0] = self.rocket_pos
        self.rocket_velocities[0] = self.rocket_v

        
        total_system_times = self.t_launch + time_list
        total_times_system_yrs = total_system_times / constants.yr
        for i, t in enumerate(time_list[1:], start=1):
            r_vec = self.rocket_pos - self.planet_pos
            r_dir = r_vec / np.linalg.norm(r_vec)
            a_g = self.gravity(r_vec)
            
            E_tot = self.U_r(r_vec) + self.T()
            if self.fuel_mass > 0.01*self.initial_fuel_mass and E_tot <= 0:
                self.total_fuel_consumed += fuel_consumed
                self.fuel_mass -= fuel_consumed
                self.m_rocket = mission.spacecraft_mass + self.fuel_mass
                a_r = (self.F / self.m_rocket) * r_dir
            else:
                a_r = 0
                if fuel_complete_time is None and self.fuel_mass < 0.01 * self.initial_fuel_mass:
                    fuel_complete_time = t
                    print(f"fuel complete after {fuel_complete_time}")
                if E_tot >= 0 and self.reached_escape_vel is None:
                    self.reached_escape_vel = t
                    print(f"remaining fuel: {self.fuel_mass} kg")
                    print(f"reached escape velocity at {self.reached_escape_vel}")
                    break
            a = a_r + a_g
            if np.linalg.norm(r_vec) <= self.planet_radius:
                a = np.array([0.0, 0.0])
            self.rocket_v += a * self.dt
            self.rocket_pos += self.rocket_v * self.dt
            self.rocket_positions[i] = self.rocket_pos
            self.rocket_velocities[i] = self.rocket_v
        print(f"Distance from planet after escaping: {np.linalg.norm(self.rocket_pos - self.planet_pos)} m")
        return self.rocket_pos, self.rocket_v, self.total_fuel_consumed, self.reached_escape_vel
    

    # after rocketsystem
    def initiate_launch(self):
        """
        Use the ASt2000tools package to launch
        """

        launch_position = (self.rocket_initial_pos) / constants.AU
        mission.set_launch_parameters(self.F, self.consumption, self.initial_fuel_mass, self.rocket_duration, launch_position=launch_position, time_of_launch=self.t_launch_years)
        mission.launch_rocket(time_step = self.dt)
        return
    
    def take_picture(self):
        self.picture_file_name = 'sky_picture.png'
        mission.take_picture(filename=self.picture_file_name, full_sky_image_path=f'{file_path}/himmelkule.npy')

    def find_phi(self, num_images=360):
        """
        Find the azimuthal angle phi that best matches an input image by comparing it to reference images.

        Parameters:
        input_image (str): File path to input image
        image_path (str): File path to all images
        num_images (int): Number of reference images (one for each phi)

        Return:
        best_phi (int): The best phi angle [deg]
        """
        image_path = f"{file_path}/Del4/pictures"
        img = np.array(Image.open(self.picture_file_name))
        min_diff = np.inf
        best_phi = None
        for i in range(num_images):
            ref_img = np.array(Image.open(f"{image_path}/himmelkule_image{i}.png"))
            diff = np.sum((img - ref_img) ** 2) # sums up all differences in rgb values and squares it

            if diff < min_diff:
                min_diff = diff
                best_phi = i
        return best_phi

    def verify_launch(self, position_after_launch):
        mission.verify_launch_result(position_after_launch)

    def verify_orientation(self, position_after_launch, velocity_after_launch, angle_after_launch):
        """
        Parameters
        position_after_launch (1-D array_like) Array of shape (2,) containing your inferred values for the x and y-position of the spacecraft, in astronomical units relative to the star.

        velocity_after_launch (1-D array_like)  Array of shape (2,) containing your inferred values for the x and y-velocity of the spacecraft, in astronomical units per year relative to the star.

        angle_after_launch (float)  Your inferred value for the azimuthal angle of the spacecraft's pointing, in degrees.

        Raises
        RuntimeError  When called before verify_launch_result().

        RuntimeError When any of the inputted values are too far from the correct values.
        """
        mission.verify_manual_orientation(position_after_launch, velocity_after_launch, angle_after_launch)
    

class RocketSystem:
    def __init__(self, rocket, simulation_time, dt, destination_idx):
        self.rocket = rocket
        self.simulation_time = simulation_time  # [yr]
        self.dt = dt  # [yr]
        self.destination_idx = destination_idx

        self.G = constants.G_sol
        self.star_mass = system.star_mass  # [Solar mass]
        self.masses = system.masses

        # Load positions and velocities over time
        pos_data = np.load('planet_positions.npz')
        vel_data = np.load('planet_velocities.npz')
        self.positions_over_time = pos_data['positions_over_time']
        self.times = pos_data['times']
        self.velocities_over_time = vel_data['velocities_over_time']

        planet_idx = rocket.planet_idx

        t_launch_years = rocket.t_launch_years # [yr]
        rocket_duration_years = simulation_time # [yr]
        t_end = t_launch_years + rocket_duration_years # [yr]
        t_reached_escape_vel = t_launch_years + rocket.reached_escape_vel / constants.yr #  [yr]
        self.N = int(simulation_time // self.dt) # num of time-steps
        self.new_times = np.linspace(t_reached_escape_vel, t_end, self.N)

        # Pos and vel at end and start of reaching escape vel
        planet_pos_start, planet_vel_start = interpolate(self.times, self.positions_over_time , planet_idx, t_launch_years, velocities_over_time=self.velocities_over_time)
        planet_pos_escape, planet_vel_ecape = interpolate(self.times, self.positions_over_time, planet_idx, t_reached_escape_vel, velocities_over_time=self.velocities_over_time)
        angle_start = np.arctan2(planet_pos_start[1],planet_pos_start[0])
        angle_end = np.arctan2(planet_pos_escape[1],planet_pos_escape[0])
        d_angle = angle_end-angle_start
        print(f"angle :{d_angle}")

        # Pos and vel relative to planet
        self.rocket_pos = rocket.rocket_pos / constants.AU + planet_vel_start * (rocket.reached_escape_vel / constants.yr)
        self.rocket_initial_pos = self.rocket_pos.copy()

        rocket_relative_vel = rocket.rocket_v * constants.yr / constants.AU  # [AU/yr]
        # rocket_relative_vel = rotate_vector(rocket_relative_vel,d_angle) # rotate to star's reference system
        self.rocket_vel = rocket_relative_vel + planet_vel_start  # [AU/yr]
        print(planet_vel_start, 'planet vel start')
        self.rocket_initial_vel = self.rocket_vel.copy()
        print(f"init vel in system: {self.rocket_initial_vel}")

        # make new times and velocities
        self.interpolated_positions, self.interpolated_velocities = interpolate_vectorized(self.times, self.positions_over_time, self.new_times, self.velocities_over_time) # (N, num_of_planets, 2)

    def acceleration(self, rocket_pos, planet_positions, planet_masses, star_mass):
        """
        Calculates the gravitational acceleration on the rocket from all planets (+star)

        Parameters:
        rocket_pos (array): Position of the rocket [AU]
        planet_positions (array): Positions of the planets [AU]
        planet_masses (array): Masses of the planets [Solar masses]
        star_mass (float): Mass of the star [Solar mass]

        Returns:
        a_total (array): Total gravitational acceleration on the rocket [AU/yr²]
        """
        r_star = rocket_pos - np.array([0.0, 0.0])
        r_star_norm = np.linalg.norm(r_star)
        a_star = -self.G * star_mass * r_star / r_star_norm**3
        a_planets = np.array([0.0, 0.0])
        if self.calc_a == True:
            for i in range(system.number_of_planets):
                r_planet = rocket_pos - planet_positions[i]
                r_planet_norm = np.linalg.norm(r_planet) 
                a_planet = -self.G * planet_masses[i] * r_planet / r_planet_norm**3
                a_planets += a_planet
                # if r_planet_norm < 1e-5:
                #     print(f"Rocket is only {r_planet_norm} AU away from planet")
                #     quit()
        else:
            None
        a_total = a_star + a_planets
        return a_total
    
    def U(self, rocket_pos, planet_positions, planet_masses, star_mass):
        """
        Calculates the potential energy of the rocket.

        Parameters:
        rocket_pos (array): Current position of the rocket [AU]
        planet_positions (array): Positions of all planets [AU]
        planet_masses (array): Masses of all planets [Solar mass]
        star_mass (float): Mass of the star [Solar mass]

        Returns:
        U_total (float): Total potential energy [AU^2/yr^2].
        """
        r_star = np.linalg.norm(rocket_pos)
        U_star = -self.G * star_mass * (self.rocket.m_rocket / constants.m_sun) / r_star

        r_planets = np.linalg.norm(rocket_pos - planet_positions, axis=1)
        U_planets = -self.G * planet_masses * (self.rocket.m_rocket / constants.m_sun) / r_planets

        U_total = U_star + np.sum(U_planets)
        return U_total

    
    def calc_orbit(self, position, velocity, origin = None, origin_vel = None, mass = None, calc_all = False):
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
            origin = np.array([0, 0])
            origin_vel = np.array([0, 0])
            mass = self.star_mass

        mu = self.G * mass

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

    def Hohmann(self, rocket_pos, rocket_vel, t):
        """
        Perform a Hohmann transfer from the current orbit to the target orbit. Assumes that the perapsis of first orbit is within the destination orbit

        Parameters:
        rocket_pos (array): Current position of the rocket [AU]
        rocket_vel (array): Current velocity of the rocket [AU/yr]
        t (float): Time relative to start [yr]

        Returns:
        dv_1 (float): Delta-v for the first burn [AU/yr]
        dv_2 (float): Delta-v for the second burn [AU/yr]
        t_1 (float): Time for the first boost [yr]
        t_2 (float): Time for the second boost [yr]
        """
        # current orbital parameters
        a1, e1 = self.calc_orbit(rocket_pos, rocket_vel)
        mu = self.G * self.star_mass

        r_p = a1 * (1 - e1) # periapsis radius of first orbit
        r_a = a1 * (1 + e1) # apoapsis - || -
        v_p = np.sqrt(mu * (2 / r_p - 1 / a1)) # velocity at perapsis of first orbit
        # v_p =  # actual velocity before boost 1 ( if needed )
        v_circ_1 = np.sqrt(mu / r_p)
        print(v_p, 'v_p')
        
        # destination parameters
        r2 = np.mean([np.linalg.norm(pos) for pos in self.interpolated_positions[:, self.destination_idx]]) + 0.03725 # radius of destination planet added approximate deviation from planet read from graph due to being elliptical )
        v2 = np.sqrt(mu / r2)

        # transfer orbit parameters
        a_t = (r_p + r2) / 2
        v_tp = np.sqrt(mu * (2 / r_p - 1 / a_t)) # velocity at periapsis of transfer orbit
        v_ta_calc = np.sqrt(mu * (2 / r2 - 1 / a_t)) # --||--apoapsis
        v_ta = v_ta_calc # 4.564912602984917  # actual velocity before boost 2 (most likely needed if close to destination)


        print(v_tp, 'v_tp')
        print(f'v_ta_calc: {v_ta_calc}, v_ta_actual: {v_ta}')
        # first burn (the same as v_tp - v_p, but just to make it clear)
        dv_1 = v_circ_1 - v_p # boost into circular orbit
        dv_1 += v_tp - v_circ_1 # boost into transfer orbit

        # second burn
        dv_2 = v2 - v_ta
        t_12 = np.pi * np.sqrt(a_t**3 / mu) # time from burn 1 to 2
        

        # Find anomaly of rocket in elliptical orbit to find time of boosts
        r_vec = rocket_pos
        v_vec = rocket_vel
        r = np.linalg.norm(r_vec)
        e_vec = (1/mu) * ((np.linalg.norm(v_vec)**2 - mu/r)*r_vec - np.dot(r_vec, v_vec)*v_vec)
        e = np.linalg.norm(e_vec)
        cos_theta = np.dot(e_vec, r_vec) / (e * r)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        if np.dot(r_vec, v_vec) < 0:
            theta = 2 * np.pi - theta
        
        if e != 0:
            e_anomaly = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(theta / 2), np.sqrt(1 + e) * np.cos(theta / 2))
            mean_anomaly = e_anomaly - e * np.sin(e_anomaly) 
            mean_anomaly = mean_anomaly % (2 * np.pi) # adjust to be within range
        else:
            print("error with eccentricity")

        T = 2 * np.pi * np.sqrt(a1**3 / mu) # orbit time

        t_since_periapsis = T * (mean_anomaly / (2 * np.pi)) 
        
        # time of boosts 
        t_1 = T - t_since_periapsis + t # (adds current time t (relative to start time))
        t_2 = t_1 + t_12  # 

        print(f"dv1: {dv_1}, dv2: {dv_2}, Time to transfer: {t_2}, Time to first boost: {t_1}")
        # for manual testing
        # dv_1 = -0.2
        # dv_2 = 0
        return dv_1, dv_2, t_1, t_2
        

    def calc_orbit_boost(self, rocket_pos, rocket_vel, planet_pos, planet_vel, planet_mass):
        """
        Calculate the delta-v vector required to enter orbit around the planet.

        Parameters:
        rocket_pos (array): Position of the rocket [AU]
        rocket_vel (array): Velocity of the rocket [AU/yr]
        planet_pos (array): Position of the planet [AU]
        planet_vel (array): Velocity of the planet [AU/yr]
        planet_mass (float): Mass of the planet [Solar masses]

        Returns:
        delta_v (array): The delta-v vector required [AU/yr]
        """

        r_vec = rocket_pos - planet_pos  # [AU]
        relative_vel = rocket_vel - planet_vel

        r = np.linalg.norm(r_vec)
    
        v_orbit = np.sqrt(self.G * planet_mass / r)  # [AU/yr]

        e_r = r_vec / r  
        e_theta = np.array([-e_r[1], e_r[0]])  

        v_orbit = v_orbit * e_theta 

        dv = v_orbit - relative_vel  # [AU/yr]

        return dv

    
    def run(self, boosts = False, calc_energy = False, calc_a = True):
        """
        Simulates the rocket's journey through the solar system.

        Parameters:
        boosts (bool): Whether to perform boosts
        calc_energy (bool): Whether to calculate the energy
        calc_a (bool): Whether to include acceleration from planets

        Returns:
        None
        """
        self.calc_a = calc_a
        dt = self.dt
        N = self.N
        masses = self.masses
        star_mass = self.star_mass

        
        m = self.rocket.m_rocket / constants.m_sun # rocket mass
        F = self.rocket.F * (constants.yr**2 / constants.AU) / (constants.m_sun) # engine force
        consumption = (self.rocket.consumption / constants.m_sun) * constants.yr # engine consumption
        fuel_consumed = 0

        rocket_pos = self.rocket_pos
        rocket_vel = self.rocket_vel

        planet_positions = self.interpolated_positions[0]
        a_i = self.acceleration(rocket_pos, planet_positions, masses, star_mass)

        # perfom first boost to really get out of planet orbit
        dv0 = 0.5 # a random number that I get by testing, really [AU/yr] (higher number will get more accuracy as it is further from planet, but use more fuel)
        dt_boost0 = m / F

        v_dir = (rocket_vel / np.linalg.norm(rocket_vel))
        boost0 = dv0 * v_dir
        self.boost0 = boost0

        rocket_vel += boost0

        fuel_consumed += consumption * dt_boost0
        m -= consumption * dt_boost0
        print(f"performing boost 0: boosted (AU/yr): {boost0}")

        rocket_pos_list = np.zeros((N,2))
        rocket_pos_list[0] = rocket_pos
        rocket_vel_list = np.zeros((N,2))
        rocket_vel_list[0] = rocket_vel


        energies = []

        t = 0
        t_list = np.linspace(0, self.simulation_time, N) # time 0 is when t_reached_escape_vel
        idx_boost0 = 0

        l_destination_list = np.zeros(N-1)
        reached_proximity_length = False
        calculated_Hohmann = False
        performed_orbit_burn = False
        for i in range(N-1):
            if calculated_Hohmann == False and t > 2.5: # calculate after a while so the rocket is closer to periapsis to cause less error (min value must be manually adjusted per testing)
                dv1, dv2, t_1, t_12 = self.Hohmann(rocket_pos, rocket_vel, t)

                self.t_1 = t_1
                self.t_12 = t_12

                idx_boost1 = np.argmin(abs(t_list - t_1)) 
                idx_boost2 = np.argmin(abs(t_list - t_12))
                calculated_Hohmann = True
            if boosts and calculated_Hohmann == True:
                v_dir = (rocket_vel / np.linalg.norm(rocket_vel))
                if i == idx_boost1:
                    print(np.linalg.norm(rocket_vel), 'actual velocity when boost 1')
                    dt_boost1 = abs(dv1) * m / F 
                    boost1 = dv1 * v_dir
                    self.boost1 = boost1

                    rocket_vel += boost1
                    fuel_consumed += consumption * dt_boost1
                    m -= consumption * dt_boost1
                    print(f"performing boost 1: boosted (AU/yr): {boost1} at time {t_1} years")
                if i == idx_boost2:
                    print(np.linalg.norm(rocket_vel), 'actual velocity when boost2')
                    dt_boost2 = abs(dv2) * m / F 
                    boost2 = dv2 * v_dir
                    self.boost2 = boost2

                    rocket_vel += boost2
                    fuel_consumed += consumption * dt_boost2
                    m -= consumption * dt_boost2
                    print(f"performing boost 2: boosted (AU/yr): {boost2} at time {t_12} years")

                # if performed_orbit_burn == False and reached_proximity_length == True:
                #     planet_pos = self.interpolated_positions[i][self.destination_idx]
                #     planet_vel = self.interpolated_velocities[i][self.destination_idx]
                #     planet_mass = masses[self.destination_idx]

                #     dv_orbit = self.calc_orbit_boost(rocket_pos, rocket_vel, planet_pos, planet_vel, planet_mass)
                #     boost_orbit = dv_orbit
                #     self.boost_orbit = boost_orbit
                #     dt_boost_orbit = np.linalg.norm(boost_orbit) * m / F

                #     rocket_vel += boost_orbit
                #     print(rocket_vel)
                #     fuel_consumed += consumption * dt_boost_orbit
                #     m -= consumption * dt_boost_orbit
                #     print(f"Performing orbit boost: delta_v {boost_orbit} at time {t} years")
                #     performed_orbit_burn = True

                

            if m < mission.spacecraft_mass / constants.m_sun:
                print('ran out of fuel filled 1000 kg')
                m = (mission.spacecraft_mass + 1000) / constants.m_sun

            t += dt
            rocket_pos += rocket_vel*dt + 0.5*a_i*dt**2
            rocket_pos_list[i+1] = rocket_pos

            # update acceleration
            planet_positions = self.interpolated_positions[i+1]
            a_iplus1 = self.acceleration(rocket_pos, planet_positions, masses, star_mass)

            # update velocity
            rocket_vel += 0.5*(a_iplus1 + a_i) * dt
            rocket_vel_list[i+1] = rocket_vel

            # ready for next
            a_i = a_iplus1

            # add energy to list
            if calc_energy == True:
                T = 0.5 * (self.rocket.m_rocket/constants.m_sun) * np.linalg.norm(rocket_vel)**2
                U = self.U(rocket_pos, planet_positions, masses, star_mass)
                energies.append(T+U)

            # check if we are within proximity length
            l_destination = np.linalg.norm(rocket_pos - planet_positions[self.destination_idx])
            proximity_length = np.linalg.norm(rocket_pos)*np.sqrt(masses[self.destination_idx])/(10*star_mass)
            l_destination_list[i] = l_destination - proximity_length
            if l_destination <= proximity_length and reached_proximity_length == False:
                print(f"reached proximity length at {t}") 
                reached_proximity_length = True
                self.t_prox_length = t
                

        self.rocket_pos = rocket_pos
        self.rocket_pos_list = rocket_pos_list
        self.rocket_vel = rocket_vel
        self.rocket_vel_list = rocket_vel_list
        self.l_destination_list = l_destination_list
        if reached_proximity_length == False:
            idx_prox = np.argmin(l_destination_list)
            print(f"deviated by {l_destination_list[idx_prox]} AU from the desired proximity-length after {t_list[idx_prox]} years")

        print(f"fuel consumed {fuel_consumed * constants.m_sun}(kg)")
        print(f"final position in space: {rocket_pos} AU")
        
        self.energies = np.array(energies)
    
    def plot_combined(self, boosts):
        plt.figure()
        for i in range(system.number_of_planets):
            x = self.interpolated_positions[:, i, 0]
            y = self.interpolated_positions[:, i, 1]
            planet_name = f"Planet {i+1}"
            plt.plot(x, y, alpha=0.8, label=planet_name, color = colors[i])

        star_color = np.array(system.star_color) / 255
        plt.plot(0, 0, "o", color=star_color, label='Star', markersize=10)

        initial_planet1 = self.interpolated_positions[0, self. rocket.planet_idx, :]  
        end_planet2 = self.interpolated_positions[-1, self.destination_idx, :] 
        plt.plot(initial_planet1[0], initial_planet1[1], 'ro', markersize=3, label="Initial Position of Planet 1")
        plt.plot(end_planet2[0], end_planet2[1], 'bo', markersize=3, label="End Position of Planet 2")
        
        rocket_x = self.rocket_pos_list[:, 0] 
        rocket_y = self.rocket_pos_list[:, 1] 

        plt.plot(rocket_x, rocket_y, color='r', linestyle='--', label="Rocket", alpha = 1, markersize=1)

        # plt.scatter(rocket_x, rocket_y, s = 1)
        # plt.scatter(self.interpolated_positions[:,0,0], self.interpolated_positions[:,0,1], s = 1)

        plt.xlim(-30, 30)
        plt.ylim(-30, 30)
        plt.autoscale(False) 
        plt.xlabel('x (AU)')
        plt.ylabel('y (AU)')
        plt.title(f'Rocket Orbit inside Solar System from {self.new_times[0]:.1f} to {self.new_times[-1]:.1f} years')
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if boosts == True:
            plt.savefig(f'rocket_and_planets_boosts_destination_p{self.destination_idx}_calc_a={self.calc_a}.png')
        else:
            plt.savefig(f'rocket_and_planets_calc_a={self.calc_a}.png')
        plt.show()

    def plot_energy(self, boosts = False):
        """
        Plots the deviation of the system's energy from the mean
        """
        plt.figure()
        energy = np.array(self.energies) * constants.m_sun * (constants.AU / constants.yr)**2 / 1000  # into kJ
        energy_dt = self.dt 
        times_energies = np.arange(len(energy)) * energy_dt


        plt.plot(times_energies, energy, label=f"Energy over time (kJ)", alpha=0.6)
        plt.xlabel("Time (years)")
        plt.ylabel(f"Energy (kJ)")
        plt.title("Energy over time")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        if boosts == False:
            plt.savefig(f'energy_plot_part5_calc_a={self.calc_a}.png')
        else:
            plt.savefig(f'energy_plot_part5_boosts_calc_a={self.calc_a}.png')
        plt.show()

    def plot_l_distances(self):
        plt.figure()
        l_destination_list = self.l_destination_list
        times = self.new_times[:-1]
        plt.plot(times, l_destination_list, label=f"Distance over time", alpha=0.6)
        plt.xlabel(f"Time from {times[0]:.2f} to {times[-1]:.2f}  years")
        plt.ylabel(f"Distance to destination (AU)")
        plt.title("Distance from Spaceship to Destination")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'L_destination_calc_a={self.calc_a}.png')
        plt.show()

    def interplanetary_travel(self, take_picture = False):
        InterplanetaryTravel = mission.begin_interplanetary_travel()
        print("Begun Interplanetary Travel")
        InterplanetaryTravel.verbose = False

        coast_time = 0.001

        times = self.new_times
        t_start = times[0]
        t_end = t_start + self.t_prox_length # times[-1]

        t_1 = self.t_1 + t_start
        t_12 = self.t_12 + t_start

        boost0 = self.boost0
        boost0_performed = False

        boost1 = self.boost1
        boost1_performed = False

        boost2 = 0 # not performed as we reach prox-length before self.boost2 
        boost2_performed = False

        # boost_orbit = self.boost_orbit
        
        t = t_start

        while t < t_end - coast_time:
            if boost0_performed == False:
                InterplanetaryTravel.boost(boost0)

            _, pos, vel = InterplanetaryTravel.orient()

            i = np.argmin(np.abs(times - t))

            simulated_pos = self.rocket_pos_list[i]
            simulated_vel = self.rocket_vel_list[i]

            tol = 1e-10

            dev_pos = np.linalg.norm(simulated_pos - pos)
            dev_vel = np.linalg.norm(simulated_vel - vel)

            if dev_vel > tol:
                delta_v = simulated_vel - vel
                InterplanetaryTravel.boost(delta_v)

            if boost1_performed == False and t + coast_time >= t_1:
                InterplanetaryTravel.coast(t_1 - t)
                t = t_1
                InterplanetaryTravel.boost(boost1)
                boost1_performed = True
            elif boost2_performed == False and t + coast_time >= t_12:
                InterplanetaryTravel.coast(t_12 - t)
                t = t_12
                InterplanetaryTravel.boost(boost2)
                boost2_performed = True
            else:
                InterplanetaryTravel.coast(coast_time)
                t += coast_time

            if t > t_end - coast_time:
                InterplanetaryTravel.coast(t_end - t)
                t += t_end - t

        InterplanetaryTravel.verbose = True
        print(f"Position deviated by {dev_pos} AU")

        if take_picture == True:
            InterplanetaryTravel.look_in_direction_of_planet(self.destination_idx)
            InterplanetaryTravel.take_picture(filename = "picture_prox.xml")
        for i in range(2): # boost two times to make sure we are orbiting
            time_of_orbit_boost, pos, vel = InterplanetaryTravel.orient()
            destination_pos, destination_vel = interpolate(self.times, self.positions_over_time, self.destination_idx, time_of_orbit_boost, velocities_over_time = self.velocities_over_time)

            boost_orbit = self.calc_orbit_boost(pos, vel, destination_pos, destination_vel, self.masses[self.destination_idx])

            InterplanetaryTravel.boost(boost_orbit) 
            InterplanetaryTravel.coast(0.05)


        t0, pos0, v0 = InterplanetaryTravel.orient()
        destination_pos0, destination_vel0 = interpolate(self.times, self.positions_over_time, self.destination_idx, t0, velocities_over_time = self.velocities_over_time)

        a0, e0, b0, T0, apoapsis0, periapsis0 = self.calc_orbit(pos0, v0, origin = destination_pos0, origin_vel = destination_vel0, mass = self.masses[self.destination_idx], calc_all = True)

        comparison_time = 20
        InterplanetaryTravel.coast(comparison_time)

        t1, pos1, v1 = InterplanetaryTravel.orient()
        destination_pos1, destination_vel1 = interpolate(self.times, self.positions_over_time, self.destination_idx, t1, velocities_over_time = self.velocities_over_time)

        a1, e1, b1, T1, apoapsis1, periapsis1 = self.calc_orbit(pos1, v1, origin = destination_pos1, origin_vel = destination_vel1, mass = self.masses[self.destination_idx], calc_all = True)

        au_to_km = constants.AU / 1e3  
        year_to_days = 365.25        

        # should have used another function making a table with a dictonary. 
        a_diff = abs(a1 - a0)
        a_mean = (a1 + a0) / 2
        e_diff = abs(e1 - e0)
        e_mean = (e1 + e0) / 2
        b_diff = abs(b1 - b0)
        b_mean = (b1 + b0) / 2
        T_diff = abs(T1 - T0)
        T_mean = (T1 + T0) / 2
        apoapsis_diff = abs(apoapsis1 - apoapsis0)
        apoapsis_mean = (apoapsis1 + apoapsis0) / 2
        periapsis_diff = abs(periapsis1 - periapsis0)
        periapsis_mean = (periapsis1 + periapsis0) / 2

        a0_km = a0 * au_to_km
        a1_km = a1 * au_to_km
        b0_km = b0 * au_to_km
        b1_km = b1 * au_to_km
        apoapsis0_km = apoapsis0 * au_to_km
        apoapsis1_km = apoapsis1 * au_to_km
        periapsis0_km = periapsis0 * au_to_km
        periapsis1_km = periapsis1 * au_to_km
        T0_days = T0 * year_to_days
        T1_days = T1 * year_to_days

        a_diff_km = abs(a1_km - a0_km)
        a_mean_km = (a1_km + a0_km) / 2
        b_diff_km = abs(b1_km - b0_km)
        b_mean_km = (b1_km + b0_km) / 2
        apoapsis_diff_km = abs(apoapsis1_km - apoapsis0_km)
        apoapsis_mean_km = (apoapsis1_km + apoapsis0_km) / 2
        periapsis_diff_km = abs(periapsis1_km - periapsis0_km)
        periapsis_mean_km = (periapsis1_km + periapsis0_km) / 2
        T_diff_days = abs(T1_days - T0_days)
        T_mean_days = (T1_days + T0_days) / 2

        print(f"Comparison of orbit after {comparison_time} years")
        print(f"Semi-major-axis start: {a0} AU, end: {a1} AU, difference: {a_diff} AU, mean: {a_mean} AU")
        print(f"Eccentricity start: {e0}, end: {e1}, difference: {e_diff}, mean: {e_mean}")
        print(f"Semi-minor-axis start: {b0} AU, end: {b1} AU, difference: {b_diff} AU, mean: {b_mean} AU")
        print(f"Orbital period start: {T0} years, end: {T1} years, difference: {T_diff} years, mean: {T_mean} years")
        print(f"Apoapsis start: {apoapsis0} AU, end: {apoapsis1} AU, difference: {apoapsis_diff} AU, mean: {apoapsis_mean} AU")
        print(f"Periapsis start: {periapsis0} AU, end: {periapsis1} AU, difference: {periapsis_diff} AU, mean: {periapsis_mean} AU")

        print("--------")
        
        print(f"Semi-major-axis start: {a0_km:.2f} km, end: {a1_km:.2f} km, difference: {a_diff_km:.2f} km, mean: {a_mean_km:.2f} km")
        print(f"Eccentricity start: {e0}, end: {e1}, difference: {e_diff}, mean: {e_mean}")
        print(f"Semi-minor-axis start: {b0_km:.2f} km, end: {b1_km:.2f} km, difference: {b_diff_km:.2f} km, mean: {b_mean_km:.2f} km")
        print(f"Orbital period start: {T0_days:.2f} days, end: {T1_days:.2f} days, difference: {T_diff_days:.2f} days, mean: {T_mean_days:.2f} days")
        print(f"Apoapsis start: {apoapsis0_km:.2f} km, end: {apoapsis1_km:.2f} km, difference: {apoapsis_diff_km:.2f} km, mean: {apoapsis_mean_km:.2f} km")
        print(f"Periapsis start: {periapsis0_km:.2f} km, end: {periapsis1_km:.2f} km, difference: {periapsis_diff_km:.2f} km, mean: {periapsis_mean_km:.2f} km")

        if take_picture == True:
            InterplanetaryTravel.look_in_direction_of_planet(self.destination_idx)
            InterplanetaryTravel.take_picture(filename = "picture_orbit.xml")
        



def main():
    seed = 4042
    N = 10**5
    dt_engine = 10**-12 
    dt_rocket = 1e-3
    L = 1e-6    
    T = 6000
    number_of_engines = int(1.9e15)
    engine_simulation_time = 10**-9  
    rocket_duration = 20 * 60   
    fuel_mass = 100000
    planet_idx = 0

    engine = RocketEngine(seed, N, engine_simulation_time, dt_engine, L, T)
    F, consumption = engine.run_engine() 
    t_launch = 6.23095 # [yrs] brute forced as I can't use Hohmann launch time because of angle = 0 and the boost dv0 :( 
    angle_launch = 0 # 0.59 #0.59 # in radians
    rocket = Rocket(seed, F, consumption, fuel_mass, number_of_engines, rocket_duration, dt_rocket, planet_idx, t_launch, angle_launch)
    rocket.initiate_launch()
    rocket_xy, rocket_v, total_fuel_consumed, launch_duration = rocket.run()

    print(f"Final Position: {rocket_xy} m")
    print(f"Final Velocity: {rocket_v} m/s")
    print(f"Total Fuel Consumed: {total_fuel_consumed} kg")


    # Initialize and run the rocket system simulation after launch
    simulation_time = 10 # 5.655657171280385 # time at reached prox length # total simulation time in [yrs]

    dt_simulation = 1e-5  # time step in [yrs]
    destination_idx = 1 # planet 2
    rocket_system = RocketSystem(rocket, simulation_time, dt_simulation, destination_idx)

    # verify launch
    rocket.verify_launch(rocket_system.rocket_initial_pos)
    

    # find and verify orientation
    rocket.take_picture()
    orientation = rocket.find_phi()
    print(orientation, "orientation")
    boosts = True
    _, vel, _ = shortcut.get_orientation_data()
    rocket_system.rocket_vel = np.array(vel)
    rocket_system.rocket_initial_vel = np.array(vel)
    rocket.verify_orientation(rocket_system.rocket_initial_pos, rocket_system.rocket_initial_vel, orientation)

    rocket_system.run(boosts = boosts, calc_energy = True, calc_a = True)

    rocket_system.plot_combined(boosts)
    rocket_system.plot_energy(boosts)
    rocket_system.plot_l_distances()
    rocket_system.interplanetary_travel(take_picture = True)
    
if __name__ == "__main__":
    main()