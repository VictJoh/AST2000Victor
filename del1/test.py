import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem


seed = 4042
system = SolarSystem(seed) 

with open('planet_table.txt', 'w') as file:
      file.write(f"{'Number':<10} {'Type':<15} {'Radius [km]':<20}{'Mass [Solar mass]':<25}{'Semi_major_axis [AU]':<30}{'Rotational period [Days]':<35}\n")
      file.write("="*125 + "\n")
      file.write(f"{'Sun':<10} {'star':<15} {system.star_radius:<20.2f} {system.star_mass:<25.2f} {0:<30} {0:<35}\n")
      for planet_idx in range(system.number_of_planets):
            planet_number = f"{planet_idx:<10}"
            planet_type = f"{system.types[planet_idx]:<15}"
            planet_radius = f"{system.radii[planet_idx]:<20.2f}"
            planet_mass = f"{system.masses[planet_idx]:<25.2e}"
            semi_major_axis = f"{system.semi_major_axes[planet_idx]:<30.2f}"
            rotational_period = f"{system.rotational_periods[planet_idx]:<35.2f}"
            
            file.write(f"{planet_number}{planet_type}{planet_radius}{planet_mass}{semi_major_axis}{rotational_period}\n")
