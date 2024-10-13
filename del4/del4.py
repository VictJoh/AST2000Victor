"""I have written this code without the use of the skeleton-code"""

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
mission = SpaceMission(seed)


def inverse_stereographical_projection(rX, rY, phi0, theta0 = np.pi/2):
    rho = np.sqrt(rX**2 + rY**2)
    beta = 2 * np.arctan2(rho, 2)

    theta = np.pi/2 -np.arcsin(np.cos(beta) * np.cos(theta0) + (rY * np.sin(beta) * np.sin(theta0) / rho))
    phi = phi0 + np.arctan2(rX * np.sin(beta), rho * np.sin(theta0) * np.cos(beta) - rY * np.cos(theta0) * np.sin(beta))

    # make them continous on the correct domain
    theta = np.mod(theta, np.pi)
    phi = np.mod(phi, np.pi * 2)
    return theta, phi


def construct_image(himmelkule, pixels, theta, phi):
    height = len(pixels[:, 0])
    width = len(pixels[0, :])

    rgb_list = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            pixel_idx = mission.get_sky_image_pixel(theta[i, j], phi[i, j])
            rgb_list[i, j] = himmelkule[pixel_idx][2:]
    return rgb_list

def find_phi(input_image, image_path, num_images=360):
    img = np.array(Image.open(input_image))
    min_diff = np.inf
    best_phi = None
    for i in range(num_images):
        ref_img = np.array(Image.open(f"{image_path}/himmelkule_image{i}.png"))
        diff = np.sum((img - ref_img) ** 2) # sums up all differences in rgb values and squares it

        if diff < min_diff:
            min_diff = diff
            best_phi = i
    return best_phi

def calculate_radial_vel():
    lambda0 = 656.3e-9 # observed wavelength
    doppler1, doppler2 = mission.star_doppler_shift_at_sun  # [nm]
    doppler1 *= 1e-9 # [m]
    doppler2 *= 1e-9 # [m]
    v1 = constants.c * (doppler1/lambda0)
    v2 = constants.c  * (doppler2/lambda0)
    return v1, v2

def calculate_spacecraft_vel(vstar1, vstar2, d_lambda1 = 0, d_lambda2 = 0):
    lambda0 = 656.3e-9 # observed wavelength
    phi1, phi2 = mission.star_direction_angles
    v_measured1 = constants.c * (d_lambda1 / lambda0)
    v_measured2 = constants.c * (d_lambda2 / lambda0)

    A = np.array([
        [np.cos(phi1), np.sin(phi1)],
        [np.cos(phi2), np.sin(phi2)]
    ])
    

    b = np.array([
        vstar1 - v_measured1,
        vstar2 - v_measured2
    ])
    vx, vy = np.linalg.solve(A, b)
    return vx, vy

def test_spacecraft_vel(v_rocket, vstar1, vstar2, d_lambda1, d_lambda2):
    vx, vy = v_rocket

    d_lambda1 = 0
    d_lambda2 = 0

    vx_approx, vy_approx = calculate_spacecraft_vel(vstar1, vstar2, d_lambda1, d_lambda2)    

    tolerance = 1e-6
    assert abs(vx_approx - vx) < tolerance, f"vx deviated by {vx_approx - vx}"
    assert abs(vy_approx - vy) < tolerance, f"vy deviated by {vy_approx - vy}"

    print("measured velocities are as expected")
    return  


def trilaterate(distances, positions):
    x_sun, y_sun = 0, 0
    d_sun = distances[-1]

    A = np.zeros((len(distances) -1, 2))
    b = np.zeros(len(distances)-1)

    for i in range(len(distances)-1):
        x, y = positions[i]
        d = distances[i]
        k = x**2 + y**2

        A[i, 0] = 2 * (x - x_sun)
        A[i, 1] = 2 * (y - y_sun)

        b[i] = -d**2 + k 
        position = np.linalg.least_squares(A, b, rcond=None)[0]
    return position


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

    pixels = np.array(img) # png into numpy array
    height = len(pixels[:, 0])
    width = len(pixels[0, :])

    fov_phi = 70 * (np.pi / 180)  
    fov_theta = fov_phi  

    X_lim = 2 * np.sin(fov_phi / 2) / (1 + np.cos(fov_phi / 2))
    Y_lim = 2 * np.sin(fov_theta / 2) / (1 + np.cos(fov_theta / 2))

    X = np.linspace(-X_lim, X_lim, width)
    Y = np.linspace(-Y_lim, Y_lim, height)
    rX, rY = np.meshgrid(X, Y)
    phi0 = 0
    theta, phi = inverse_stereographical_projection(rX, rY, phi0)

    # image = construct_image(himmelkule, pixels, theta, phi)
    # actual_image = Image.fromarray(image)
    # actual_image.save(f'C:/Users/victo/Documents/GitHub/AST2000-Project/del4/pictures/himmelkule_image{0}.png')

    # for i in range(360):
    #     phi0 = i * (np.pi / 180) 
    #     theta, phi = inverse_stereographical_projection(rX, rY, phi0)
        
    #     image = construct_image(himmelkule, pixels, theta, phi)
    #     actual_image = Image.fromarray(image)
    #     actual_image.save(f'C:/Users/victo/Documents/GitHub/AST2000-Project/del4/pictures/himmelkule_image{i}.png')

    image_path = 'C:/Users/victo/Documents/GitHub/AST2000-Project/del4/pictures'
    input_img = 'C:/Users/victo/Documents/GitHub/AST2000-Project/sample0200.png'
    phi_new = find_phi(input_img, image_path)
    print(f"sample0200 centered at around {phi_new} deg")

    vstar1, vstar2 = calculate_radial_vel()
    vx, vy = calculate_spacecraft_vel(vstar1, vstar2)
    print(vx, vy)
    # test_spacecraft_vel(v_rocket, vstar1, vstar2, d_lambda1, d_lambda2)

    launch_time = 597.06 / constants.yr
    time_idx = int(launch_time//1e-5) # launch time from part 1
    planet_positions = positions_over_time[time_idx]
    distances = mission.measure_distances()
    print(trilaterate(planet_positions, distances))

    print("finished running")
if __name__ == "__main__":
    main()
