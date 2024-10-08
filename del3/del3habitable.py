
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


seed = 4042
system = SolarSystem(seed)

sigma = constants.sigma

R_star = system.star_radius * 1e3 # to m
T_star = system.star_temperature
L_star = 4 * np.pi * sigma * R_star **2 * T_star**4
number_of_planets = system.number_of_planets
semi_major_axes = system.semi_major_axes.copy() 

def calculate_planet_temps():
    distances = semi_major_axes * constants.AU
    temps = (L_star / (16 * np.pi * sigma * (distances**2)))**(1/4)
    return temps

def habitable_planets(temps):
    T_upper = 390
    T_lower = 260

    habitable_idx = np.where((temps >= T_lower) & (temps <= T_upper))[0]

    return habitable_idx

def habitable_zone(L_star):
    T_upper = 390
    T_lower = 260

    inner = np.sqrt(L_star / (16 * np.pi * sigma * T_upper**4)) / constants.AU # to AU
    outer = np.sqrt(L_star / (16 * np.pi * sigma * T_lower**4)) / constants.AU # to AU
    return inner,outer

positions_over_time = np.load('planet_positions.npz')['positions_over_time']

def plot_combined(positions_over_time):
    """
    Plots the numerical and analytical orbits of the planets, along with the habitable zone.

    Parameters:
    positions_over_time (array): Planet positions over time in AU.

    Returns:
    None
    """
    plt.figure()
    for i in range(system.number_of_planets):
        x = positions_over_time[:, i, 0]  # takes all time steps of planet i at coordinate 0 (x-axis)
        y = positions_over_time[:, i, 1]  # takes all time steps of planet i at coordinate 1 (y-axis)
        plt.plot(x, y, alpha=0.8)

    # Habitable Zone
    inner, outer = habitable_zone(L_star)
    
    theta = np.linspace(0, 2 * np.pi, 1000)
    inner_x = inner * np.cos(theta)
    inner_y = inner * np.sin(theta)
    outer_x = outer * np.cos(theta)
    outer_y = outer * np.sin(theta)

    plt.fill(outer_x, outer_y, 'green', alpha=0.2, label="Habitable Zone")
    plt.fill(inner_x, inner_y, 'white', alpha=1.0)  


    plt.plot(0, 0, label="Numerical Orbits", alpha=0.8) 
    
    star_color = np.array(system.star_color) / 255
    plt.scatter(0, 0, color=star_color, label='Star', s=100, zorder=5)

    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')
    plt.title('Habitable Zone of the Solar System')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('combined_plot_with_habitable_zone.png')

temps = calculate_planet_temps()
print(f"Temperatures: {temps}")
print(f"Habitable planets: {habitable_planets(temps)}")
print(f"Habitable zone: {habitable_zone(L_star)}")
plot_combined(positions_over_time)
plt.show()
