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

def part1():
    relativity.black_hole_descent(planet_idx=1, consider_light_travel=False)
    relativity.black_hole_descent(planet_idx=1, consider_light_travel=True)

    signal_idx1, times1 = np.loadtxt("black_hole_descent_frame_1.txt")
    signal_idx1_lt, times1_lt = np.loadtxt("black_hole_descent_frame_1_with_light_travel.txt")

    signal_idx2, times2 = np.loadtxt("black_hole_descent_frame_2.txt")
    signal_idx2_lt, times2_lt = np.loadtxt("black_hole_descent_frame_2_with_light_travel.txt")



    time_differences1 = np.diff(times1)
    time_differences1_lt = np.diff(times1_lt)

    time_differences2 = np.diff(times2)
    time_differences2_lt = np.diff(times2_lt)

    plt.figure()
    plt.plot(signal_idx1[:-1], time_differences1, label='Without Light Travel Time', marker='o')
    plt.plot(signal_idx1_lt[:-1], time_differences1_lt, label='With Light Travel Time', marker='o')

    plt.title('Time Differences Between Signals (Planet System)')
    plt.xlabel('Signal Number')
    plt.ylabel('Time Difference (s)')
    plt.legend()
    plt.savefig("TimeDiff_Planet.png")
    plt.show()


    plt.figure()
    plt.plot(signal_idx2[:-1], time_differences2, label='Without Light Travel Time', marker='o')
    plt.plot(signal_idx2_lt[:-1], time_differences2_lt, label='With Light Travel Time', marker='o')

    plt.title('Time Differences Between Signals (Space Ship System)')
    plt.xlabel('Signal Number')
    plt.ylabel('Time Difference (s)')
    plt.legend()
    plt.savefig("TimeDiff_Ship.png")
    plt.show()

def part2():
    relativity.gps(planet_idx = 1)
    print(f"c_km_pr_s: {constants.c_km_pr_s}, G_SI: {constants.G}")

    c = constants.c_km_pr_s
    G = constants.G

    M = 6.3911604335564e24  # [kg]
    R = 6618.0645243 # [km]
    h = 10636.72035186296 # [km}

    

    def solve(relativistic = False, first = True):
        if first == True: # this is horrible code, I know, but I did not read the entire task before starting
            x_sat1, y_sat1 = 14805.887, -8860.774
            x_sat2, y_sat2 = 8391.887, -15076.599
            tau = 450.1735168

            t1, t2 = 450.1022909, 450.1118486 
  
        else:
            x_sat1, y_sat1 = -7574.279 , 15503.480
            x_sat2, y_sat2 = 1192.222, 17213.547 
            tau = 13355.1476662

            t1, t2 =  13355.0849584, 13355.0756715

        def t_rel(t, tau):
            G_km = G * (1e-3)**3
            M_nat = G_km*M/(c**2)  # to convert into km
            v = 4.972078160029129 / c
            t_r = tau - np.sqrt((1- 2*M_nat/R)/(1 - (2*M_nat)/(R+h)-v**2))* (tau - t)
            return t_r 

        t1_r, t2_r = t_rel(t1,tau), t_rel(t2,tau)

        if relativistic:  # I do not need to change unit because the unit out is km meaning that [s] dissapears
            C1 = 0.5 * ((R + h) ** 2 + R ** 2 - c ** 2 * (tau - t1_r) ** 2)
            C2 = 0.5 * ((R + h) ** 2 + R ** 2 - c ** 2 * (tau - t2_r) ** 2)
        else:
            C1 = 0.5 * ((R + h) ** 2 + R ** 2 - c ** 2 * (tau - t1) ** 2)
            C2 = 0.5 * ((R + h) ** 2 + R ** 2 - c ** 2 * (tau - t2) ** 2)


        a1, b1 = x_sat1, y_sat1
        a2, b2 = x_sat2, y_sat2

        D = a1 * b2 - a2 * b1
        Dx = C1 * b2 - C2 * b1
        Dy = a1 * C2 - a2 * C1

        x = Dx / D
        y = Dy / D

        return x,y

    first = False
    x,y = solve(first = first)
    x_r, y_r = solve(relativistic = True, first = first)
    

    print("not-relativistic ", "first: ", first, x,y, np.sqrt(x**2 + y**2))
    print("relativistic ", "first: ", first, x_r, y_r, np.sqrt(x_r**2 + y_r**2))

def part5():
    M = 1  
    m = 1 

    r_min = 2.0 * M 
    r_max = 20 * M
    r = np.linspace(r_min, r_max, 1000)

    L_values = np.linspace(0,10,5)

    plt.figure(figsize=(10, 6))

    for L in L_values:
        V_eff = np.sqrt((1-2*M/r)*(1+ ((L/m)**2) / (r**2)))
        plt.plot(r, V_eff, label=f'L = {L}')

    plt.axvline(x=2 * M, color='purple', alpha = 0.6, linestyle='--', label='Schwarzschild Radius')

    plt.xlabel('Radial Distance r (M)')
    plt.ylabel('Effective Potential')
    plt.title('Effective Potential')
    plt.grid(True)
    plt.legend()
    plt.savefig('potential.png')
    plt.show()

    M = 1  
    m = 1 

    r_min = 2.0 * M 
    r_max = 20 * M
    r = np.linspace(r_min, r_max, 1000)
    R = 20 * M
    theta = 167 
    v = 0.993 

    gamma = 1 / (np.sqrt(1-v**2))
    L = m*(R*gamma*v*np.sin(np.radians(theta)))

    V_eff = np.sqrt((1-2*M/r)*(1+ ((L/m)**2) / (r**2)))
    V_eff_max = np.max(V_eff)  
    r_max = r[np.argmax(V_eff)]

    E_m = np.sqrt(1 - 2 * M / R) * gamma

    plt.figure(figsize=(10, 6))


    plt.plot(r, V_eff, label=f'L = {L}')

    plt.plot(r_max, V_eff_max, 'ro', label=f'Max V_eff: {V_eff_max} ')
    plt.axhline(y=E_m, color='green', linestyle='--', label=f'E/m: {E_m}')
    plt.axvline(x=2 * M, color='purple', alpha = 0.6, linestyle='--', label='Schwarzschild Radius')
    

    plt.xlabel('Radial Distance r (M)')
    plt.ylabel('Effective Potential')
    plt.title('Effective Potential')
    plt.grid(True)
    plt.legend()
    plt.savefig('actual_potential.png')
    plt.show()

if __name__ == "__main__":
    # part1()
    # part2()
    part5()



