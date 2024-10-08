import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem as system
from ast2000tools.space_mission import SpaceMission as mission
from ast2000tools import constants as constants
import numpy as np
from matplotlib import pyplot as plt
from numba import jit

seed = 4042
system = system(seed) 

def solve_analytic_r(a, e, f):
    p = a * (1-e**2)
    r = p / (1 + e* np.cos(f))
    return r

def xy_pos(f, r):
    r_x = r * np.cos(f)
    r_y = r * np.sin(f)
    return r_x, r_y

f_vals = np.linspace(0, 2*np.pi, 360)

r_list = np.zeros((system.number_of_planets, 360))

r_x_list = np.zeros((system.number_of_planets, 360))
r_y_list = np.zeros((system.number_of_planets, 360))

plt.figure(figsize=(8,8))
for i in range(system.number_of_planets):
    e = system.eccentricities[i]
    a = system.semi_major_axes[i]
    r = solve_analytic_r(a,e,f_vals)
    r_x, r_y = xy_pos(f_vals, r) 
    r_x_list[i] = r_x
    r_y_list[i] = r_y
    plt.plot(r_x, r_y, label=f'Planet {i+1}')

star_radius_km = system.star_radius  # in kilometers
au_to_km = constants.AU  # 1 AU in kilometers
star_radius_au = star_radius_km / au_to_km  # Convert radius to AU
star_color = np.array(system.star_color) / 255


plt.plot(0, 0,"o", color=star_color, label='Star', markersize = 10) # not to size
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.title('Elliptical Orbits of Planets')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
