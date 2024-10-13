"""
This code is written without the skeleton-code
"""
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
mpl.rcParams['axes.grid'] = True 
mpl.rcParams['grid.alpha'] = 0.3
plt.style.use('grayscale')
colors = ['b', 'm', 'c', 'y', 'g', 'orange', 'purple']


seed = 4042
system = SolarSystem(seed)
mission = SpaceMission(seed)


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
      
        self.system = SolarSystem(seed)
        self.mission = SpaceMission(seed)  
       
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

class Rocket:
    def __init__(self, seed, F, consumption, fuel_mass, number_of_engines, rocket_duration, dt, planet_idx, t_launch = 0.0, angle_launch = 0.0):
        # system
        self.system = SolarSystem(seed)
        self.mission = SpaceMission(seed)

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
        self.planet_mass = self.system.masses[planet_idx] * constants.m_sun # turns into kg
        self.planet_radius = self.system.radii[planet_idx] * 1e3 # turns into m
        self.planet_rotation_period = self.system.rotational_periods[planet_idx] * constants.day
        
        pos_data = np.load('planet_positions.npz')
        self.positions_over_time = pos_data['positions_over_time']
        self.times = pos_data['times']

        self.t_launch = t_launch
        self.t_launch_years = t_launch / constants.yr
        self.idx_launch = np.argmin(np.abs(self.times - self.t_launch_years))

        self.planet_pos = self.positions_over_time[self.idx_launch, planet_idx, :] * constants.AU  # [m/s]

        total_angle = np.arctan2(self.planet_pos[1], self.planet_pos[0]) + angle_launch
        planet_surface_direction = np.array([np.cos(total_angle), np.sin(total_angle)])
        self.rocket_pos = self.planet_pos + self.planet_radius * planet_surface_direction

        planet_rotation_speed = 2 * np.pi * self.planet_radius / self.planet_rotation_period 
        v_rot_direction = np.array([-planet_surface_direction[1], planet_surface_direction[0]]) 
        v_rot = planet_rotation_speed * v_rot_direction

        self.rocket_v = v_rot

        self.rocket_initial_pos = self.rocket_pos.copy()
        self.planet_initial_pos = self.planet_pos.copy()

        self.m_rocket = self.fuel_mass + self.mission.spacecraft_mass
        np.random.seed(seed)


    def gravity(self, r_vec): 
        """
        Calculates the gravity force on the rocket

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
        [self.rocket_x, self.rocket_y] (array) : The position of the rocket (m)
        self.rocket_v (float) : The absolute speed of the rocket (m/s)
        self.total_fuel_consumed (float) : Total fuel consumed (kg)
        
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
            r_vec = self.rocket_pos - self.planet_initial_pos
            r_dir = r_vec / np.linalg.norm(r_vec)
            a_g = self.gravity(r_vec)
            
            E_tot = self.U_r(r_vec) + self.T()
            if self.fuel_mass > 0.01*self.initial_fuel_mass and E_tot <= 0:
                self.total_fuel_consumed += fuel_consumed
                self.fuel_mass -= fuel_consumed
                self.m_rocket = self.mission.spacecraft_mass + self.fuel_mass
                a_r = (self.F / self.m_rocket) * r_dir
            else:
                a_r = 0
                if fuel_complete_time == None and self.fuel_mass < 0.01 * self.initial_fuel_mass:
                    fuel_complete_time = t
                    print(f"fuel complete after {fuel_complete_time}")
                if E_tot >= 0 and self.reached_escape_vel == None:
                    self.reached_escape_vel = t
                    print(f"remaining fuel: {self.fuel_mass} kg")
                    print(f"reached escape velocity at {self.reached_escape_vel}")
            a = a_r + a_g
            if np.linalg.norm(a) < 0 and np.linalg.norm(r_vec) <= self.planet_radius:
                a = np.array([0.0,0.0])
            self.rocket_v += a * self.dt
            self.rocket_pos += self.rocket_v * self.dt
            self.rocket_positions[i] = self.rocket_pos
            self.rocket_velocities[i] = self.rocket_v
        print(f"Distance from planet after escaping: {np.linalg.norm(self.rocket_pos[-1] - self.planet_pos)} m")
        return self.rocket_pos, self.rocket_v, self.total_fuel_consumed, self.reached_escape_vel
    
    def initiate_launch(self):
        """
        Use the ASt2000tools package to launch
        """

        launch_position = (self.rocket_initial_pos) / constants.AU
        self.mission.set_launch_parameters(self.F, self.consumption, self.initial_fuel_mass, self.rocket_duration, launch_position=launch_position, time_of_launch=self.t_launch_years)
        self.mission.launch_rocket(time_step = self.dt)
        return

class RocketSolarSystem:
    def __init__(self, seed, rocket, tot_sim_time, t_launch, planet_idx, launch_duration, dt):
        self.seed = seed
        self.system = SolarSystem(seed)
        self.mission = SpaceMission(seed)
        self.G = constants.G
        self.tot_sim_time = tot_sim_time
        self.launch_duration = launch_duration
        # Load planetary data
        positions_data = np.load('planet_positions.npz')
        v_data = np.load('planet_velocities.npz')
        self.positions_over_time = positions_data['positions_over_time']
        self.velocities_over_time = v_data['velocities_over_time']
        self.times = positions_data['times']  # in years
        self.times_seconds = self.times * constants.yr  # in seconds
        self.N = len(self.times)
        self.rocket = rocket
        self.dt = dt

        # Launch time
        self.t_launch = t_launch
        self.t_launch_years = t_launch / constants.yr
        self.idx_launch = np.argmin(np.abs(self.times - self.t_launch_years))

        # Convert positions and velocities to SI units
        self.positions_over_time_SI = self.positions_over_time * constants.AU
        self.velocities_over_time_SI = self.velocities_over_time * (constants.AU / constants.yr)

        self.dt = self.times_seconds[1] - self.times_seconds[0]

        # Initial rocket position and velocity
        rocket_pos_SI = rocket.rocket_pos + self.velocities_over_time_SI[self.idx_launch][planet_idx] * 597.059
        rocket_v_SI = rocket.rocket_v + self.velocities_over_time_SI[self.idx_launch][planet_idx]

        self.rocket_pos = rocket_pos_SI
        self.rocket_v = rocket_v_SI

        self.rocket_positions = np.zeros((self.N, 2))
        self.rocket_positions[0] = self.rocket_pos

        self.star_mass = self.system.star_mass * constants.m_sun

    def interpolate(self, t):
        idx = np.searchsorted(self.times_seconds, t) - 1
        if self.times_seconds[idx] >= t:
            idx += -1
        t0 = self.times_seconds[idx]
        t1 = self.times_seconds[idx + 1]    
        dt = t1-t0
        
        interpolated_pos = np.zeros((self.system.number_of_planets, 2))
        for i in range(self.system.number_of_planets)
            pos0 = self.positions_over_time_SI[idx, i]
            pos1 = self.positions_over_time_SI[idx + 1, i]
            interpolated_pos[i] = pos0 + ((t - t0) / dt) * (pos1 - pos0)
        return interpolated_pos

    @staticmethod
    @njit
    def run_simulation(rocket_pos, rocket_v, rocket_positions, N, dt, G, star_mass):
        """
        """
        for i in range(1, N):
            r_vec = rocket_pos
            r_norm = np.sqrt(r_vec[0]**2 + r_vec[1]**2)
            a_g = -(G * star_mass / r_norm**3) * r_vec
            
            rocket_v += a_g * dt
            rocket_pos += rocket_v * dt
            rocket_positions[i] = rocket_pos
        return rocket_positions

    def run(self):
        """Runs the run_simulation, but inputs the class-
        variables so we can use numba"""
        self.rocket_positions = self.run_simulation(
            self.rocket_pos, self.rocket_v, self.rocket_positions,
            self.N, self.dt, self.G, self.star_mass)
        return self.rocket_positions

    def plot_combined(self):
        t_end = (self.t_launch_years + (self.tot_sim_time / constants.yr))
        idx_end = np.argmin(np.abs(self.times - t_end))
        plt.figure()
        for i in range(self.system.number_of_planets):
            x = self.positions_over_time[self.idx_launch:idx_end, i, 0]
            y = self.positions_over_time[self.idx_launch:idx_end, i, 1]
            planet_name = f"Planet {i+1}"
            plt.plot(x, y, alpha=0.8, label=planet_name, color = colors[i])

        star_color = np.array(self.system.star_color) / 255
        plt.plot(0, 0, "o", color=star_color, label='Star', markersize=10)

        rocket_x = self.rocket_positions[:, 0] / constants.AU
        rocket_y = self.rocket_positions[:, 1] / constants.AU
        plt.plot(rocket_x, rocket_y, 'r-', label="Rocket")
        plt.xlim(-30, 30)
        plt.ylim(-30, 30)
        plt.autoscale(False) 
        plt.xlabel('x (AU)')
        plt.ylabel('y (AU)')
        plt.title('Rocket Orbit inside Solar System over 3 years')
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig('rocket_and_planets.png')
        plt.show()

        

def main():
    seed = 4042
    N = 10**5
    dt_engine = 10**-12 
    dt_rocket = 0.01
    L = 1e-6    
    T = 6000
    number_of_engines = 10**15  
    engine_simulation_time = 10**-9  
    rocket_duration = 20 * 60   
    fuel_mass = 100000
    planet_idx = 0

    engine = RocketEngine(seed, N, engine_simulation_time, dt_engine, L, T)
    F, consumption = engine.run_engine() 
    t_launch = 3e4 * (1e-5 * constants.yr) # in seconds
    angle_launch = 0 # in radians
    rocket = Rocket(seed, F, consumption, fuel_mass, number_of_engines, rocket_duration, dt_rocket, planet_idx, t_launch, angle_launch)
    rocket_xy, rocket_v, total_fuel_consumed, launch_duration = rocket.run()

    print(f"Final Position: {rocket_xy} m")
    print(f"Final Velocity: {rocket_v} m/s")
    print(f"Total Fuel Consumed: {total_fuel_consumed} kg")
    rocket.initiate_launch()

    tot_sim_time = 3 * constants.yr
    RocketSystem = RocketSolarSystem(seed, rocket, tot_sim_time, t_launch, planet_idx, launch_duration)
    rocket_positions = RocketSystem.run()
    RocketSystem.plot_combined()
    plt.show()

    # for part 4
    # RocketSystem.rocket.initiate_launch()
    # RocketSystem.rocket.mission.verify_launch_result((RocketSystem.rocket_positions[0])/constants.AU)
    # print(RocketSystem.rocket.mission.measure_distances())

if __name__ == "__main__":
    main()

