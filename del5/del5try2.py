from PIL import Image
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

import matplotlib as mpl  # https://pythonforthelab.com/blog/python-tip-ready-publish-matplotlib-figures/ inspiration from this
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
        std = np.sqrt(self.k * self.T / self.m)  # Standard deviation of the Maxwell-Boltzmann distribution
        return np.random.normal(loc=0, scale=std, size=(int(self.N), 3))

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
        indice_particles_at_bottom = pos <= 0  # Gives the location in the position matrix of values below 0 as True and False otherwise
        pos[indice_particles_at_bottom] = 0  # Sets all values that are True to 0 (back to the edge)
        vel[indice_particles_at_bottom] *= -1  # Changes the velocity in the direction of collision to the other direction
        # The same as above but values above L
        indice_particles_at_top = pos >= self.L
        pos[indice_particles_at_top] = self.L
        vel[indice_particles_at_top] *= -1

        particles_z0 = pos[:, 2] <= 0  # Gives index of particles with z-axis below 0 as True
        particles_x_thrust = (pos[:, 0] > 0.25 * self.L) & (pos[:, 0] < 0.75 * self.L)  # Gives index of particles with x position between 0.25L and 0.75L as True (where the hole is)
        particles_y_thrust = (pos[:, 1] > 0.25 * self.L) & (pos[:, 1] < 0.75 * self.L)  # --||-- y_position

        indice_particles_at_thrust = (particles_z0 & particles_x_thrust & particles_y_thrust)  # checks iff all three are true for each particle

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
        z_velocities_escaping = abs(velocities[indice_particles_at_thrust, 2])  # makes a boolean array of all the z-velocities of the escaping particles
        dp = np.sum(self.m * z_velocities_escaping)  # gives the change of momentum
        dF = dp / self.dt  # calculates the force exerted
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
        consumption = (np.count_nonzero(indice_particles_at_thrust) * self.m)  # amount of particles that are at the the thrust multiplied with their mass
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
        v_mean_exp = np.sqrt(8 * self.k * self.T / (np.pi * self.m))  # the expected mean from MB-distribution
        total_v_mean = np.mean(v_means)  # Takes the mean of all the means from each time step
        alpha = 0.05 * v_mean_exp  # The relative deviation we accept
        deviation = abs(v_mean_exp - total_v_mean)  # Our absolute deviation
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
        max_speed = np.max(all_v_norms)  # takes the max speed
        speeds = np.linspace(0, max_speed, 10000)  # makes the x-axis from 0 to max speed with 10000 spaces
        factor = 4 * np.pi * (self.m / (2 * np.pi * self.k * self.T))**(3/2)  # constants in the equation
        exponent = np.exp(-self.m * speeds**2 / (2 * self.k * self.T))  # the exp-part
        mb_distribution = factor * speeds**2 * exponent  # Maxwell-Boltzmann distribution gives expected number per velocity

        plt.figure(figsize=(8, 6), dpi=240)  # We set the size to 8,6 as this best fits in our latex-file and resolution to 240 as this looked good

        plt.hist(all_v_norms.flatten(), bins=50, density=True, color='gray', label='Simulated Speeds')  # Makes histogram with 50 blocks from our v_norms we use .flatten as we want it to be a 1D array

        plt.plot(speeds, mb_distribution, color='black', linewidth=2, linestyle='--', label='Maxwell-Boltzmann Distribution')

        plt.xlabel('Speed (m/s)', fontsize=14)
        plt.ylabel('Probability Density', fontsize=14)
        plt.title('Simulated Speeds vs Maxwell-Boltzmann Distribution', fontsize=18)

        plt.grid(True, color='gray', linestyle='--')

        v_median = np.sqrt(2 * self.k * self.T / self.m)  # the expected mean speed
        plt.axvline(x=v_median, color='black', linestyle=':', linewidth=2)  # makes a dotted line at the mean speed
        plt.text(v_median, max(mb_distribution), 'Most Probable Speed', color='black', fontsize=12)  # makes text at the dotted line

        plt.legend(fontsize=12, loc='upper right')

        plt.tight_layout()  # makes it look better we think. To be honest we just learned to do this no matter what
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
        t_list = np.arange(0, self.engine_simulation_time, self.dt)
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


def interpolate(times, positions_over_time, planet_idx, t, velocities_over_time=None):
    """
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


def interpolate_vectorized(times, positions_over_time, new_times, velocities_over_time=None):
    """
    Interpolate planet positions and optionally velocities to new time list new_ttimes

    Parameters:
    times (array): time list for the positions_over_time
    positions_over_time (array): array of size (N,num_of_planets,2) with original positions
    new_times (array): array containing new times
    velocities_over_time (array): If wanted we can also interpolate this the same as positions_over_time
    """

    x_positions = positions_over_time[:, :, 0].T  # (num_of_planets, N)
    y_positions = positions_over_time[:, :, 1].T  # (num_of_planets, N)

    interpolated_x = np.array([np.interp(new_times, times, x) for x in x_positions])
    interpolated_y = np.array([np.interp(new_times, times, y) for y in y_positions])

    interpolated_x = interpolated_x.T
    interpolated_y = interpolated_y.T

    interpolated_positions = np.stack([interpolated_x, interpolated_y], axis=-1)  # (N, num_of_planets, 2)

    if velocities_over_time is not None:
        vx_over_time = velocities_over_time[:, :, 0].T
        vy_over_time = velocities_over_time[:, :, 1].T

        interpolated_vx = np.array([np.interp(new_times, times, vx) for vx in vx_over_time])
        interpolated_vy = np.array([np.interp(new_times, times, vy) for vy in vy_over_time])

        interpolated_vx = interpolated_vx.T
        interpolated_vy = interpolated_vy.T

        interpolated_velocities = np.stack([interpolated_vx, interpolated_vy], axis=-1)  # (N, num_of_planets, 2)

        return interpolated_positions, interpolated_velocities
    else:
        return interpolated_positions


def rotate_vector(vec, angle):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)]])
    return rotation_matrix @ vec


class Rocket:
    def __init__(self, seed, F, consumption, fuel_mass, number_of_engines, rocket_duration, dt, planet_idx, t_launch=0.0, angle_launch=0.0):
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
        self.planet_mass = system.masses[planet_idx] * constants.m_sun  # turns into kg
        self.planet_radius = system.radii[planet_idx] * 1000  # turns into m
        self.planet_rotation_period = system.rotational_periods[planet_idx] * constants.day

        pos_data = np.load('planet_positions.npz')
        self.positions_over_time = pos_data['positions_over_time']
        self.times = pos_data['times']

        self.t_launch = t_launch * constants.yr  # [s]
        self.t_launch_years = t_launch  # [yrs]

        self.planet_pos = interpolate(self.times, self.positions_over_time, planet_idx, t_launch) * constants.AU  # [m]

        planet_angle = np.arctan2(self.planet_pos[1], self.planet_pos[0])
        total_angle = planet_angle + angle_launch

        self.planet_direction = np.array([np.cos(total_angle), np.sin(total_angle)])

        self.rocket_pos = self.planet_pos + self.planet_radius * self.planet_direction

        planet_rotation_speed = 2 * np.pi * self.planet_radius / self.planet_rotation_period

        v_rot_direction = np.array([-self.planet_direction[1], self.planet_direction[0]])  # (-y, x)
        v_rot = planet_rotation_speed * v_rot_direction

        self.rocket_v = v_rot

        self.m_rocket = self.fuel_mass + mission.spacecraft_mass

        self.rocket_initial_pos = np.copy(self.rocket_pos)

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
            if self.fuel_mass > 0.01 * self.initial_fuel_mass and E_tot <= 0:
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
        mission.launch_rocket(time_step=self.dt)
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
            diff = np.sum((img - ref_img) ** 2)  # sums up all differences in rgb values and squares it

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

    def begin_interplanetary(self):
        mission.begin_interplanetary_travel()


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

        t_launch_years = rocket.t_launch_years  # [yr]
        rocket_duration_years = simulation_time  # [yr]
        t_end = t_launch_years + rocket_duration_years  # [yr]
        t_reached_escape_vel = t_launch_years + rocket.reached_escape_vel / constants.yr
        self.N = int(simulation_time // self.dt)
        self.new_times = np.linspace(t_launch_years, t_end, self.N)

        # Pos and vel at end and start of reaching escape vel
        planet_pos_start, planet_vel_start = interpolate(self.times, self.positions_over_time, planet_idx, t_launch_years,velocities_over_time=self.velocities_over_time)
        planet_pos_escape, planet_vel_escape = interpolate(self.times, self.positions_over_time, planet_idx,t_reached_escape_vel,velocities_over_time=self.velocities_over_time)
        angle_start = np.arctan2(planet_pos_start[1], planet_pos_start[0])
        angle_end = np.arctan2(planet_pos_escape[1], planet_pos_escape[0])
        d_angle = angle_end - angle_start
        print(f"angle :{d_angle}")

        # Pos and vel relative to planet
        self.rocket_pos = rocket.rocket_pos / constants.AU + planet_vel_start * (rocket.reached_escape_vel / constants.yr)
        self.rocket_initial_pos = self.rocket_pos.copy()

        rocket_relative_vel = rocket.rocket_v * constants.yr / constants.AU  # [AU/yr]
        print(rocket_relative_vel)
        rocket_relative_vel = rotate_vector(rocket_relative_vel, d_angle)  # rotate to star's reference system
        print(rocket_relative_vel)
        self.rocket_vel = rocket_relative_vel + planet_vel_start  # [AU/yr]
        print(planet_vel_start, 'planet vel start')
        self.rocket_initial_vel = self.rocket_vel.copy()
        print(f"init vel in system: {self.rocket_initial_vel}")

        # make new times and velocities
        self.interpolated_positions, self.interpolated_velocities = interpolate_vectorized(self.times, self.positions_over_time, 
                                                                                           self.new_times,  velocities_over_time=self.velocities_over_time)  # (N, num_of_planets, 2)

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
        for i in range(system.number_of_planets):
            r_planet = rocket_pos - planet_positions[i]
            r_planet_norm = np.linalg.norm(r_planet)
            a_planet = -self.G * planet_masses[i] * r_planet / r_planet_norm**3
            a_planets += a_planet
        a_total = a_star + a_planets
        return a_total

    def calc_orbit(self, position, velocity):
        mu = self.G * self.star_mass

        r = np.linalg.norm(position)
        v = np.linalg.norm(velocity)

        h_vec = np.cross(position, velocity)
        h = np.linalg.norm(h_vec)
        energy = (v**2 / 2) - (mu / r)
        a = -mu / (2 * energy)

        e = np.linalg.norm((1 / mu) * ((v**2 - mu / r) * position - np.dot(position, velocity) * velocity))
        return a, e

    def Hohmann(self):
        """
        Perform a Hohmann transfer from the current orbit to the target orbit.
        This method now calculates the ideal launch time (t_launch) and launch angle (launch_angle)
        relative to the planet's motion for the Hohmann transfer.

        Returns:
            dv_1: Delta-v for the first burn
            dv_2: Delta-v for the second burn
            t_launch: Ideal launch time (in years)
            launch_angle: Required launch angle relative to the planet's motion (in radians)
            t_12: Time of flight for the transfer
            t_1: Time until the first burn (when velocity is perpendicular to position)
        """
        # Step 1: Calculate current orbital parameters
        a1, e1 = self.calc_orbit(self.rocket_initial_pos, self.rocket_initial_vel)
        mu = self.G * self.star_mass

        # Orbital radii of the planets
        r1 = a1  # AU (assuming semi-major axis equals current position for circular orbit)
        r2 = np.mean([np.linalg.norm(pos) for pos in self.interpolated_positions[:, self.destination_idx]])  # AU

        # Compute the semi-major axis of the transfer orbit
        a_trans = (r1 + r2) / 2

        # Compute the time of flight for the transfer orbit
        t_transit = np.pi * np.sqrt(a_trans**3 / mu)  # in years

        # Compute angular velocities
        omega1 = np.sqrt(mu / r1**3)
        omega2 = np.sqrt(mu / r2**3)

        # Compute required phase angle
        delta_omega = omega2 - omega1
        phase_angle = np.pi - delta_omega * t_transit  # in radians

        # Adjust phase angle to be between 0 and 2π
        phase_angle = phase_angle % (2 * np.pi)

        # Now, find the time(s) when the phase angle between planet1 and planet2 equals the required phase angle

        # Get positions of the two planets over time
        positions1 = self.interpolated_positions[:, self.rocket.planet_idx, :]  # shape (N, 2)
        positions2 = self.interpolated_positions[:, self.destination_idx, :]    # shape (N, 2)

        # Compute the angle of each planet with respect to x-axis (in radians)
        angles1 = np.arctan2(positions1[:, 1], positions1[:, 0]) % (2 * np.pi)
        angles2 = np.arctan2(positions2[:, 1], positions2[:, 0]) % (2 * np.pi)

        # Compute the phase angle between the two planets over time
        phase_angles_over_time = (angles2 - angles1) % (2 * np.pi)

        # Compute the difference between the required phase angle and the actual phase angle
        diff_phase = (phase_angles_over_time - phase_angle + np.pi) % (2 * np.pi) - np.pi  # Shift to (-π, π)

        # Find zero crossings in diff_phase
        sign_diff_phase = np.sign(diff_phase)
        zero_crossings = np.where(np.diff(sign_diff_phase))[0]

        # For each zero crossing, interpolate the time when the phase angle equals the required phase angle
        t_launch_list = []

        for idx in zero_crossings:
            t1 = self.times[idx]
            t2 = self.times[idx + 1]
            phase1 = diff_phase[idx]
            phase2 = diff_phase[idx + 1]

            # Linear interpolation to find t_launch where diff_phase crosses zero
            if phase2 - phase1 != 0:
                t_launch = t1 - phase1 * (t2 - t1) / (phase2 - phase1)
                t_launch_list.append(t_launch)

        if len(t_launch_list) == 0:
            print("No suitable launch windows found in the available data.")
            t_launch = None
            launch_angle = None
        else:
            t_launch = t_launch_list[0]  # Earliest launch window

            # Compute the required launch angle relative to the planet's motion at t_launch
            idx_launch = np.searchsorted(self.times, t_launch)
            if idx_launch >= len(self.times):
                idx_launch = len(self.times) - 1

            # Get the position and approximate velocity of the starting planet at t_launch
            planet_pos_launch = self.interpolated_positions[idx_launch, self.rocket.planet_idx, :]
            planet_vel_launch = self.interpolated_velocities[idx_launch, self.rocket.planet_idx, :]

            # Normalize the planet's velocity vector
            planet_vel_unit = planet_vel_launch / np.linalg.norm(planet_vel_launch)

            # Compute the orbital speed of the planet
            v_circ = np.sqrt(mu / r1)

            # Compute the required speed at perigee of transfer orbit
            v_trans = np.sqrt(mu * (2 / r1 - 1 / a_trans))

            delta_v = v_trans - v_circ

            if delta_v >= 0:
                launch_angle = 0.0  # Along planet's motion
            else:
                launch_angle = np.pi  # Opposite to planet's motion

            print(f"Calculated t_launch: {t_launch} years")
            print(f"Calculated launch_angle: {np.degrees(launch_angle)} degrees")

        # Update the rocket's launch time and angle if calculated
        if t_launch is not None and launch_angle is not None:
            self.rocket.t_launch_years = t_launch
            self.rocket.t_launch = t_launch * constants.yr
            self.rocket.planet_pos = interpolate(self.times, self.positions_over_time, self.rocket.planet_idx, t_launch) * constants.AU

            # Update rocket's direction based on the launch angle
            planet_angle = np.arctan2(self.rocket.planet_pos[1], self.rocket.planet_pos[0])
            total_angle = planet_angle + launch_angle

            self.rocket.planet_direction = np.array([np.cos(total_angle), np.sin(total_angle)])
            self.rocket.rocket_pos = self.rocket.planet_pos + self.rocket.planet_radius * self.rocket.planet_direction

            # Update rocket's velocity based on the launch angle
            planet_rotation_speed = 2 * np.pi * self.rocket.planet_radius / self.rocket.planet_rotation_period
            v_rot_direction = np.array([-self.rocket.planet_direction[1], self.rocket.planet_direction[0]])  # (-y, x)
            v_rot = planet_rotation_speed * v_rot_direction
            self.rocket.rocket_v = v_rot

        # Compute Hohmann transfer delta-v's
        v1 = np.sqrt(mu * (2 / r1 - 1 / a_trans))
        v2 = np.sqrt(mu * (2 / r2 - 1 / a_trans))
        v_circ_dest = np.sqrt(mu / r2)

        dv_1_magnitude = v1 - np.sqrt(mu / r1)  # Delta-v for first burn
        dv_2_magnitude = v_circ_dest - v2  # Delta-v for second burn

        # Assuming the burns are in the direction of velocity
        dv_1 = dv_1_magnitude * (planet_vel_launch / np.linalg.norm(planet_vel_launch))
        dv_2 = dv_2_magnitude * (planet_vel_launch / np.linalg.norm(planet_vel_launch))

        # Time of flight for the transfer
        t_12 = t_transit  # years

        # Time until the first burn
        t_1 = t_launch  # years

        print(f"Delta-v1: {dv_1} AU/yr")
        print(f"Delta-v2: {dv_2} AU/yr")
        print(f"Time to transfer (t_12): {t_12} years")
        print(f"Time of first burn (t_1): {t_1} years")

        return dv_1, dv_2, t_launch, launch_angle, t_12, t_1

    def U(self, rocket_pos, planet_positions, planet_masses, star_mass):
        r_star = np.linalg.norm(rocket_pos)
        U_star = -self.G * star_mass * (self.rocket.m_rocket / constants.m_sun) / r_star

        r_planets = np.linalg.norm(rocket_pos - planet_positions, axis=1)
        U_planets = -self.G * planet_masses * (self.rocket.m_rocket / constants.m_sun) / r_planets

        U_total = U_star + np.sum(U_planets)
        return U_total

    def run(self, boosts=False, calc_energy=False):
        dt = self.dt
        N = self.N
        masses = self.masses
        star_mass = self.star_mass

        rocket_pos = self.rocket_pos
        rocket_vel = self.rocket_vel

        rocket_pos_list = np.zeros((N, 2))
        rocket_pos_list[0] = rocket_pos
        rocket_vel_list = np.zeros((N, 2))
        rocket_vel_list[0] = rocket_vel

        planet_positions = self.interpolated_positions[0]
        a_i = self.acceleration(rocket_pos, planet_positions, masses, star_mass)

        energies = []

        # Perform Hohmann transfer to get delta-v's and launch parameters
        dv1, dv2, t_launch, launch_angle, t_12, t_1 = self.Hohmann()

        # Convert delta-v from AU/yr to m/s
        delta_v1_m_s = dv1 * constants.AU / constants.yr
        delta_v2_m_s = dv2 * constants.AU / constants.yr

        # Calculate the time steps for the burns
        dt_boost1 = abs(delta_v1_m_s) / self.rocket.F  # seconds
        dt_boost2 = abs(delta_v2_m_s) / self.rocket.F  # seconds

        print(dt_boost1, dt_boost2)

        fuel_consumed = 0

        t = 0
        t_list = np.linspace(0, self.simulation_time, N)
        idx_boost1 = np.argmin(abs(t_list - t_1))
        idx_boost2 = np.argmin(abs(t_list - t_12))
        l_destination_list = np.zeros(N - 1)
        for i in range(N - 1):
            if boosts:
                if i == idx_boost1:
                    rocket_vel += dv1
                    fuel_consumed += self.rocket.consumption * dt_boost1
                    print(f"performing boost 1: boosted (AU/yr): {dv1}")
                if i == idx_boost2:
                    rocket_vel += dv2
                    fuel_consumed += self.rocket.consumption * dt_boost2
                    print(f"performing boost 2: boosted (AU/yr): {dv2}")

            t += dt
            rocket_pos += rocket_vel * dt + 0.5 * a_i * dt**2
            rocket_pos_list[i + 1] = rocket_pos

            # update acceleration
            planet_positions = self.interpolated_positions[i + 1]
            a_iplus1 = self.acceleration(rocket_pos, planet_positions, masses, star_mass)

            # update velocity
            rocket_vel += 0.5 * (a_iplus1 + a_i) * dt
            rocket_vel_list[i + 1] = rocket_vel

            # ready for next
            a_i = a_iplus1

            # add energy to list each 100 time step
            if i % 100 == 0 and calc_energy:
                T = 0.5 * (self.rocket.m_rocket / constants.m_sun) * np.linalg.norm(rocket_vel)**2
                U = self.U(rocket_pos, planet_positions, masses, star_mass)
                energies.append(T + U)

            # check if we are within orbit of new planet
            l_destination = np.linalg.norm(rocket_pos - planet_positions[self.destination_idx])
            proximity_length = np.linalg.norm(rocket_pos) * np.sqrt(masses[self.destination_idx]) / (10 * star_mass)
            l_destination_list[i] = l_destination - proximity_length
            if l_destination <= proximity_length:
                print(f"reached prox length at time : {t} years")
                break

        self.rocket_pos = rocket_pos
        self.rocket_pos_list = rocket_pos_list
        self.rocket_vel = rocket_vel
        self.rocket_vel_list = rocket_vel_list
        self.l_destination_list = l_destination_list
        print(f"deviated by {min((l_destination_list))} AU from the destination")
        print(f"fuel consumed {fuel_consumed * constants.m_sun}(kg)")

        self.energies = np.array(energies)

    def plot_combined(self):
        plt.figure()
        for i in range(system.number_of_planets):
            x = self.interpolated_positions[:, i, 0]
            y = self.interpolated_positions[:, i, 1]
            planet_name = f"Planet {i + 1}"
            plt.plot(x, y, alpha=0.8, label=planet_name, color=colors[i])

        star_color = np.array(system.star_color) / 255
        plt.plot(0, 0, "o", color=star_color, label='Star', markersize=10)

        initial_p1 = self.interpolated_positions[0, self.rocket.planet_idx, :]
        initial_p2 = self.interpolated_positions[0, self.destination_idx, :]
        plt.plot([initial_p1[0], initial_p2[0]], [initial_p1[1], initial_p2[1]], 'ro', markersize=1)

        rocket_x = self.rocket_pos_list[:, 0]
        rocket_y = self.rocket_pos_list[:, 1]
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

    def plot_energy(self, boosts=False):
        """
        Plots the deviation of the system's energy from the mean
        """
        plt.figure()
        energy = np.array(self.energies) * constants.m_sun * (constants.AU / constants.yr)**2 / 1000  # into kJ
        energy_dt = self.dt
        times_energies = np.arange(len(energy)) * energy_dt / constants.yr

        plt.plot(times_energies, energy, label=f"Energy over time kJ", alpha=0.6)
        plt.xlabel("Time (years)")
        plt.ylabel(f"Energy (kJ)")
        plt.title("Energy over time kJ")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        if boosts == False:
            plt.savefig('energy_plot_part5.png')
        else:
            plt.savefig('energy_plot_part5_boosts.png')
        plt.show()

    def plot_l_distances(self):
        plt.figure()
        l_destination_list = self.l_destination_list
        times = self.new_times[:-1]
        plt.plot(times, l_destination_list, label=f"L_destination", alpha=0.6)
        plt.xlabel("Time (years)")
        plt.ylabel(f"L_destination (AU)")
        plt.title("L_destination Over Time")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('L_destination.png')
        plt.show()


def main():
    seed = 4042
    N = 10**5
    dt_engine = 10**-12
    dt_rocket = 1e-3
    L = 1e-6
    T = 6000
    number_of_engines = int(1e15)
    engine_simulation_time = 10**-9
    rocket_duration = 20 * 60
    fuel_mass = 100000
    planet_idx = 0
    destination_idx = 1

    # Load positions and times
    pos_data = np.load('planet_positions.npz')
    positions_over_time = pos_data['positions_over_time']
    times = pos_data['times']

    # Initialize the RocketEngine
    engine = RocketEngine(seed, N, engine_simulation_time, dt_engine, L, T)
    F, consumption = engine.run_engine()

    # Initialize the Rocket with default launch parameters
    # These will be updated by the RocketSystem's Hohmann method
    rocket = Rocket(seed, F, consumption, fuel_mass, number_of_engines, rocket_duration, dt_rocket, planet_idx, t_launch = 0.39)
    rocket.initiate_launch()
    rocket_xy, rocket_v, total_fuel_consumed, launch_duration = rocket.run()

    print(f"Final Position: {rocket_xy} m")
    print(f"Final Velocity: {rocket_v} m/s")
    print(f"Total Fuel Consumed: {total_fuel_consumed} kg")

    # Initialize and run the rocket system simulation after launch
    simulation_time = 20  # Total simulation time in years

    dt_simulation = 1e-4  # Time step in years

    rocket_system = RocketSystem(rocket, simulation_time, dt_simulation, destination_idx)

    # Perform Hohmann transfer to calculate launch window and angles
    dv1, dv2, t_launch, launch_angle, t_12, t_1 = rocket_system.Hohmann()

    # Verify launch
    rocket.verify_launch(rocket_system.rocket_initial_pos)

    # Find and verify orientation
    # rocket.take_picture()
    # orientation = rocket.find_phi()
    # print(orientation, "orientation")
    boosts = False
    # print(rocket_system.calculate_launch_window())  # Now integrated into Hohmann
    rocket_system.run(boosts=boosts, calc_energy=True)
    rocket_system.plot_combined()
    # rocket.verify_orientation(rocket_system.rocket_initial_pos, rocket_system.rocket_initial_vel, orientation)
    rocket_system.plot_energy(boosts)
    rocket_system.plot_l_distances()


if __name__ == "__main__":
    main()
