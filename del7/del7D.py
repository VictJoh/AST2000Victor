"""
This code is written without the skeleton-code. And, yes I know using global variables is not ideal, but I could not be bothered to manually output every variable. Sorry
"""
from PIL import Image
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission, LandingSequence
from ast2000tools.shortcuts import SpaceMissionShortcuts
from ast2000tools.star_population import StarPopulation

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


sigma = constants.sigma

star_radius = system.star_radius * 1000 # [m]
star_temp = system.star_temperature # [K]
star_L = 4 * np.pi * star_radius**2 * sigma * star_temp**4 / constants.L_sun # [L_sun]

print(star_L)
stars = StarPopulation(seed = seed)
T = stars.temperatures # [K]
L = stars.luminosities # [L_sun]
r = stars.radii        # [R_sun]


c = stars.colors
s = np.maximum(1e3*(r - r.min())/(r.max() - r.min()), 1.0) # Make point areas proportional to star radii

fig, ax = plt.subplots()
ax.scatter(T, L, c=c, s=s, alpha=0.8, edgecolor='k', linewidth=0.05)
ax.scatter(star_temp, star_L, color='black', s=100, marker='*', label='My Star', linewidth=0.5)

ax.set_xlabel('Temperature [K]')
ax.invert_xaxis()
ax.set_xscale('log')
ax.set_xticks([35000, 18000, 10000, 6000, 4000, 3000])
ax.set_xticklabels(list(map(str, ax.get_xticks())))
ax.set_xlim(40000, 2000)
ax.minorticks_off()
plt.legend()

ax.set_ylabel(r'Luminosity [$L_\odot$]')
ax.set_yscale('log')
ax.set_ylim(1e-4, 1e6)

plt.savefig('Del7/HR_diagram.png')
# plt.show()


def core_temp():
    X = 0.7
    Y = 0.2
    Z = 0.1

    mu = 4 / (6*X + Y + 2)
    T0 = system.star_temperature # K
    G = constants.G 
    R = system.star_radius * 1e3 # m
    V = (4/3) * np.pi * R**3 
    star_mass = system.star_mass * constants.m_sun
    m_H = constants.m_p
    k = constants.k_B
    rho0 = system.star_mass / V

    T_c = T0 + (G * (2 * np.pi / 3) * rho0 * (mu * m_H) / k) * R**2
    
    return T_c


Tc = core_temp()
print(f"T_c = {Tc:.2e} K")


