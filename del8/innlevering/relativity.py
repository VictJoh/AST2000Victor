import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools import constants
from ast2000tools.relativity import RelativityExperiments
import numpy as np
import matplotlib.pyplot as plt


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



seed = 4042
system = SolarSystem(seed)
mission = SpaceMission(seed)
relativity = RelativityExperiments(seed)

planet_idx = 0

relativity.spaceship_duel(planet_idx)
relativity.cosmic_pingpong(planet_idx)
relativity.spaceship_race(planet_idx)
relativity.neutron_decay(planet_idx)
relativity.antimatter_spaceship(planet_idx)

def plot_ty():
    # this took way too long time and then I realised I could have used a mask just for the time we start acceleration...
    v0 = 0.99
    g = -3.33e-10 # [1/s]
    tB = 202 * 31557600 # [s]
    t_tp = -v0/g + tB
    print(t_tp/31557600)
    L0 = 200 * 31557600 # [s]
    gamma = 1/np.sqrt(1-v0**2)

    ty_list = np.linspace(0,t_tp,1000)
    accelerated = (ty_list >= tB)
    v_Y_list = np.ones_like(ty_list) * v0
    v_Y_list[accelerated] = v0 + g * (ty_list[accelerated] - tB)

    gamma_list = 1/(np.sqrt(1-v_Y_list**2))

    xY_list = np.zeros_like(ty_list)

    xY_list[~accelerated] = v0 * ty_list[~accelerated]
    xY_list[accelerated] = L0 + v_Y_list[accelerated] * (ty_list[accelerated] - tB) 
    
    tY_prime_list = np.zeros_like(ty_list)
    tY_prime_list = ty_list - xY_list * (1- 1/gamma_list)

    tY_prime_list /= 31557600
    ty_list /= 31557600

    
    plt.figure()
    plt.plot(ty_list, tY_prime_list, label="tY'(tY)" )
    plt.title("tY' vs tY")
    plt.axvline(x= tB / 31557600 , color='purple', alpha = 0.6, linestyle='--', label='t_B')    
    plt.xlabel("tY (years)")
    plt.ylabel("tY' (years)")
    plt.grid(True)
    plt.legend(loc = "upper left")
    plt.savefig("ty_plot.png")
    plt.show()

plot_ty()
