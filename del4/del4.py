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


def inverse_stereographical_projection(rX, rY, phi0 = 0, theta0 = np.pi/2):
    """
    Finds the inverse stereographical projection from X,Y to theta, phi

    Parameters:
    rX (array): X-coordinates
    rY (array): Y-coordinates
    phi0 (float): Phi angle of center [rad]
    theta0 (float): Theta angle of center [rad]

    Return:
    theta (array): Theta angles [rad]
    phi (array): Phi angles [rad]
    """

    rho = np.sqrt(rX**2 + rY**2)
    beta = 2 * np.arctan2(rho, 2)

    theta = theta0 - np.arcsin(np.cos(beta) * np.cos(theta0) + (rY * np.sin(beta) * np.sin(theta0) / rho))
    phi = phi0 + np.arctan2(rX * np.sin(beta), rho * np.sin(theta0) * np.cos(beta) - rY * np.cos(theta0) * np.sin(beta))

    # make them continous on the correct domain
    theta = np.mod(theta, np.pi)
    phi = np.mod(phi, np.pi * 2)
    return theta, phi


def construct_image(himmelkule, pixels, theta, phi):
    """
    Construct an RGB image based from spherical coords.

    Parameters:
    himmelkule (array): Array containing RGB-values for each pixel
    pixels (array): Pixel-grid
    theta (array): Theta angles [rad]
    phi (array): Phi angles [rad]

    Return:
    rgb_list (array): 3D array representing the constructed RGB image.
    """
    height = len(pixels[:, 0])
    width = len(pixels[0, :])  

    rgb_list = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            pixel_idx = mission.get_sky_image_pixel(theta[j, i], phi[j, i])
            rgb_list[i, j] = himmelkule[pixel_idx][2:]
    return rgb_list

def find_phi(input_image, image_path, num_images=360):
    """
    Find the azimuthal angle phi that best matches an input image by comparing it to reference images.

    Parameters:
    input_image (str): File path to input image
    image_path (str): File path to all images
    num_images (int): Number of reference images (one for each phi)

    Return:
    best_phi (int): The best phi angle [deg]
    """
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
    """
    Calculate the radial velocities of two stars based on measured Doppler-shifts

    Return:
    v1 (float): Radial velocity of star1 [m/s]
    v2 (float): Radial velocity of star2 [m/s].
    """
    lambda0 = 656.3e-9 # observed wavelength
    doppler1, doppler2 = mission.star_doppler_shifts_at_sun  # [nm]
    doppler1 *= 1e-9 # [m]
    doppler2 *= 1e-9 # [m]
    v1 = constants.c * (doppler1/lambda0) 
    v2 = constants.c  * (doppler2/lambda0)
    return v1, v2

def calculate_spacecraft_vel(vstar1, vstar2, d_lambda1 = 0, d_lambda2 = 0):
    """
    Calculate rockets velocity in x,y direction

    Parameters:
    vstar1 (float): Radial velocity of the first star in [m/s[
    vstar2 (float): Radial velocity of the second star in [m/s]
    d_lambda1 (float): Doppler shift for the first star [nm]
    d_lambda2 (float): Doppler shift for the second star [nm]

    Return:
    vx (float): Rocket-velocity in x [m/s]
    vy (float): Rocket-velocity in y [m/s]
    """
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

def test_spacecraft_vel():
    """
    Test the spacecraft velocity calculation function for correctness.
    """
    vx = 1000 
    vy = 1000  
    v_rocket = (vx, vy)

    # 2. Get star direction angles (in radians)
    phi1, phi2 = mission.star_direction_angles

    # assume no velocity of star
    vstar1 = 0 
    vstar2 = 0

    # calc expected doppler
    lambda0 = 656.3e-9  # Rest wavelength in meters
    c = constants.c      # Speed of light in m/s

    # calc radial velocity towards the stars
    v_radial1 = vx * np.cos(phi1) + vy * np.sin(phi1)
    v_radial2 = vx * np.cos(phi2) + vy * np.sin(phi2)

    # calculate observed dopplershift as vel of stars are 0
    v_measured1 = - v_radial1
    v_measured2 = - v_radial2

    d_lambda1 = (v_measured1 / c) * lambda0  # [m]
    d_lambda2 = (v_measured2 / c) * lambda0  # [m]

    vx_calculated, vy_calculated = calculate_spacecraft_vel(vstar1, vstar2, d_lambda1, d_lambda2)

    # compare
    tolerance = 1e-4  # Tolerance in m/s
    assert abs(vx_calculated - vx) < tolerance, f"vx deviated by {vx_calculated - vx}"
    assert abs(vy_calculated - vy) < tolerance, f"vy deviated by {vy_calculated - vy}"

    print("Measured velocities are as expected")

def trilaterate(distances, positions):  
    """
    Trilaterate rocket's position
    Parameters:
    distances (array): Distances to each planet and sun [AU]
    positions (array): Positions of the planets [AU]

    Returns:
    position (array): Estimated position (x, y) of the spacecraft.
    """

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
        position = np.linalg.lstsq(A, b, rcond=None)[0]
    return position

### TO BE USED FOR TRILATERATE FROM PART 6 SO I DO NOT HAVE TO ADD MUCH CODE UNUSED
def initiate_launch():
    mission.set_launch_parameters(thrust = 1950414.2360053714, mass_loss_rate = 313.3328015733722, initial_fuel_mass = 100000, estimated_launch_duration = 1200, launch_position = [2.79291596, 0.50094891], time_of_launch=6.23095)
    mission.launch_rocket(time_step = 0.001)

def verify_launch():
    mission.verify_launch_result(position_after_launch =  [2.79291222, 0.50100744])



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
    Y = np.linspace(Y_lim, -Y_lim, height) # to avoid rotation of the picture for some reason
    rX, rY = np.meshgrid(X, Y, indexing = 'ij')
    phi0 = 0
    theta, phi = inverse_stereographical_projection(rX, rY, phi0)

    # image = construct_image(himmelkule, pixels, theta, phi)
    # actual_image = Image.fromarray(image)
    # actual_image.save(f'del4/pictures/himmelkule_image{0}.png')

    # for i in range(360):
    #     phi0 = i * (np.pi / 180) 
    #     theta, phi = inverse_stereographical_projection(rX, rY, phi0)
        
    #     image = construct_image(himmelkule, pixels, theta, phi)
    #     actual_image = Image.fromarray(image)
    #     actual_image.save(f'del4/pictures/himmelkule_image{i}.png')

    image_path = 'del4/pictures'
    input_img = 'sample0200.png'
    phi_new = find_phi(input_img, image_path)
    print(f"sample0200 centered at around {phi_new} deg")

    test_spacecraft_vel()

    initiate_launch()
    verify_launch()

    mission.take_picture(filename = "del4/pictures/find_phi_part4.jpeg")
    phi = find_phi(input_image = "find_phi_part4.jpeg", image_path = image_path)
    vstar1, vstar2 = calculate_radial_vel()
    vx, vy = calculate_spacecraft_vel(vstar1, vstar2)
    velocity_after_launch = np.array([vx, vy]) / constants.AU * constants.yr # AU/yr
    print("vel = ",velocity_after_launch)

    launch_time = 6.23095 + 307.846/constants.yr
    time_idx = int(launch_time//1e-5) # launch time from part 1
    planet_positions = positions_over_time[time_idx]
    distances = mission.measure_distances()
    position_after_launch = trilaterate(distances, planet_positions) # the values are a bit off as I have not interpolated, but they worked earlier, but I changed the launch etc. and changed it here aswell without thinking twice
    mission.verify_manual_orientation(position_after_launch = position_after_launch, velocity_after_launch = velocity_after_launch, angle_after_launch = phi)  
if __name__ == "__main__":
    main()
