"""
This code is written without the skeleton-code and has taken waaay to much time.
"""
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools import constants as constants
import numpy as np
from matplotlib import pyplot as plt

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

        self.plot_maxwell_boltzmann_comparison(all_v_norms)
        self.validate_mean_speed(v_means)
        avg_F = F / len(t_list)
        consumption = consumed / self.engine_simulation_time
        return avg_F, consumption   


class Rocket():
    def __init__(self, seed, F, consumption, fuel_mass, number_of_engines, rocket_duration, dt):
        self.seed = seed
        self.dt = dt
        self.initial_fuel_mass = fuel_mass
        self.fuel_mass = fuel_mass
        self.rocket_duration = rocket_duration
        self.F = F * number_of_engines
        self.consumption = consumption * number_of_engines
        self.system = SolarSystem(seed)
        self.mission = SpaceMission(seed)
        self.G = constants.G
        self.total_fuel_consumed = 0
        self.total_force = 0

        self.planet_mass = self.system.masses[0] * 1.98847e30   # turns into kg from solar mass
        self.planet_radius = self.system.radii[0] * 10**3 # from km to m
        self.planet_rotation_sec = self.system.rotational_periods[0] * 24 * 60 * 60 # days to seconds
        self.planet_initial_pos = [self.system.initial_positions[0][0], self.system.initial_positions[1][0]]

        self.rocket_x_initial = self.planet_radius
        self.rocket_y_initial = 0

        self.rocket_x = self.rocket_x_initial
        self.rocket_y = self.rocket_y_initial
        self.rocket_r = np.linalg.norm([self.rocket_x, self.rocket_y])

        self.rocket_v_x = 0
        self.rocket_v_y = np.pi * 2 * self.planet_radius / self.planet_rotation_sec
        self.rocket_v = np.linalg.norm([self.rocket_v_x, self.rocket_v_y])

        self.m_rocket = self.fuel_mass + self.mission.spacecraft_mass

        np.random.seed(seed)

    def gravity(self): 
        """
        Calculates the gravity force on the rocket

        Returns:
        a_g (float) : the gravity force on the rocket [m/s^2]
        """
        a_g = -(self.G * self.planet_mass) / (self.rocket_r)**2
        return a_g
    
    def U_r(self):
        """
        Calculates the potential energy of the rocket 

        Returns:
        The potential energy of the rocket (float) [J] 
        """
        return - (constants.G * self.planet_mass * self.m_rocket) / (self.rocket_r)
    
    def T(self, v, m_rocket):
        """
        Calculates the kinetic energy of the rocket 

        Parameters:
        m_rocket (float) : the mass of the rocket [kg]

        Returns:
        The kinetic energy (float) [J]
        """
        return 0.5 * m_rocket * v**2

    def calculate_astronomical_pos(self):
        """
        Calculates the astronomical position of the rocket

        Returns:
        Astronomical position (array) : The x and y coordinate of the rocket [AU]
        """
        planet_pos_x = self.planet_initial_pos[0]
        planet_pos_y = self.planet_initial_pos[1]
   
        m_to_AU = 1 / 149597870700

        astronomical_rocket_x = planet_pos_x + (self.rocket_x_initial * m_to_AU) 
        astronomical_rocket_y = planet_pos_y + (self.rocket_y_initial * m_to_AU) 

        return [astronomical_rocket_x, astronomical_rocket_y]

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
        E_list = []
        fuel_complete_time = None
        reached_escape_vel = None
        for t in time_list:
            a_g = self.gravity()
            E_tot = self.U_r() + self.T(self.rocket_v, self.m_rocket)
            if self.fuel_mass > 0.01*self.initial_fuel_mass and E_tot <= 0:
                self.fuel_mass += -fuel_consumed    
                self.total_fuel_consumed += fuel_consumed
                self.m_rocket = self.mission.spacecraft_mass + self.fuel_mass
                a_r = self.F / self.m_rocket
            else:
                a_r = 0
                if fuel_complete_time == None and self.fuel_mass < 0.01 * self.initial_fuel_mass:
                    fuel_complete_time = t
                    print(f"fuel complete after {fuel_complete_time}")
                if E_tot >= 0 and reached_escape_vel == None:
                    reached_escape_vel = t
                    print(f"remaining fuel: {self.fuel_mass} kg")
                    print(f"reached escape velocity at {reached_escape_vel}")
            a = a_r + a_g
            if a < 0 and self.rocket_r <= self.planet_radius:
                a = 0
            speed_boost = a * self.dt
            self.rocket_v_x += speed_boost
            self.rocket_v = np.linalg.norm([self.rocket_v_x, self.rocket_v_y])

            self.rocket_x += self.rocket_v_x * self.dt
            self.rocket_y += self.rocket_v_y * self.dt
            self.rocket_r = np.linalg.norm([self.rocket_x, self.rocket_y])
            E_list.append(E_tot)
        print(f"max Energy {max(E_list)}")
        return [self.rocket_x, self.rocket_y], self.rocket_v, self.total_fuel_consumed
    
    def initiate_launch(self):
        """
        Use the ASt2000tools package to launch
        """
        launch_position = self.calculate_astronomical_pos()
        time_of_launch = 0
        self.mission.set_launch_parameters(self.F, self.consumption, self.initial_fuel_mass, self.rocket_duration, launch_position, time_of_launch)
        self.mission.launch_rocket(time_step = self.dt)
        return
    
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

    engine = RocketEngine(seed, N, engine_simulation_time, dt_engine, L, T)
    F, consumption = engine.run_engine() 

    rocket = Rocket(seed, F, consumption, fuel_mass, number_of_engines, rocket_duration, dt_rocket)
    rocket_xy, rocket_v, total_fuel_consumed = rocket.run()

    print(f"Final Position: {rocket_xy} m")
    print(f"Final Velocity: {rocket_v} m/s")
    print(f"Total Fuel Consumed: {total_fuel_consumed} kg")
    print(rocket.gravity())
    print(rocket.calculate_astronomical_pos())
    rocket.initiate_launch()

if __name__ == "__main__":
    main()

