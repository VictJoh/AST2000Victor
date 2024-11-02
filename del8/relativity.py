import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission
from ast2000tools import constants
from ast2000tools.relativity import RelativityExperiments
import numpy as np

seed = 4042
system = SolarSystem(seed)
mission = SpaceMission(seed)
relativity = RelativityExperiments(seed)

planet_idx = 0

relativity.spaceship_duel(planet_idx)
relativity.cosmic_pingpong(planet_idx)