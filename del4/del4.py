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





def inverse_stereographical_projection(rX, rY):
    D = rX**2 + rY**2 + 1

    x = (2*rX) / D
    y = (2*rY) / D
    z = (D-2) / D
    return x, y, z

def cartesian_to_spherical(x, y, z):
    theta = np.mod(np.arccos(z), 2*np.pi) # polar in [0,2pi]
    phi = np.arctan2(y,x) # azimuth

    return theta, phi

def map_to_pixel(theta, phi, pixels):
    height = len(pixels[:, 0])
    width = len(pixels[0, :])
    # make angles range from [0,1]
    theta_01 = theta_norm = theta / np.pi  
    phi_01 = phi_norm = phi / (2 * np.pi)  

    row = (1 - theta_norm) * (height - 1)  
    col = phi_norm * (width - 1)
    
    return row, col


def construct_image(himmelkule, pixels, theta, phi):
    height = len(pixels[:, 0])
    width = len(pixels[0, :])
    row, col = map_to_pixel(theta, phi, pixels)

    image = np.zeros_like(pixels)
    for _,_,r,g,b in himmelkule()

        for i in range(height):
            for j in range(width):
                row = int(

    return image



def main():
    required_files = ['planet_positions.npz', 'planet_velocities.npz', 'himmelkule.npy', 'sample0000.png']
    if all(os.path.exists(file) for file in required_files):
        positions_over_time = np.load('planet_positions.npz')['positions_over_time']
        times = np.load('planet_positions.npz')['times']
        velocities_over_time = np.load('planet_velocities.npz')['velocities_over_time']
        himmelkule = np.load('himmelkule.npy')
        img = Image.open("sample0000.png") 
    else:
        print("files are missing")
        quit()


    img = Image.open("sample0000.png") # Open existing png
    pixels = np.array(img) # png into numpy array
    height = len(pixels[:, 0])
    width = len(pixels[0, :])


    X = np.linspace(-1, 1, width)
    Y = np.linspace(-1,1, height)

    rX, rY = np.meshgrid(X, Y)
    x, y, z = inverse_stereographical_projection(rX, rY)
    theta, phi = cartesian_to_spherical(x, y, z)

    image = construct_image(himmelkule, pixels, theta, phi)
    actual_image = Image.fromarray(image)
    actual_image.save('himmelkule_image.png')

    polar_angle = 0
    azimuth_angle = 0
    SpaceMission.get_sky_image_pixel(polar_angle,azimuth_angle)
    print("finished running")


if __name__ == "__main__":
    main()
