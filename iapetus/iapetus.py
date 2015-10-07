'''
iapetus.py
----------
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

BigG = 6.67408e-11

ice_density = 935.
chondrite_density = 3500.


# Create a class for Iapetus that will do the heavy lifting.
class Iapetus(object):

    def __init__(self, n_slices):
	#The constructor takes the number of depth slices which will
	#be used for the calculation.  More slices will generate more
	#accurate profiles, but it will take longer.

	self.cmb = 285.e3  # Guess for the radius of the core-mantle-boundary
	self.outer_radius = 735.6e3 # Outer radius of the planet

	self.radii = np.linspace(0.e3, self.outer_radius, n_slices) # Radius list
	self.pressures = np.linspace(1.0e8, 0.0, n_slices) # initial guess at pressure profile

    def generate_profiles( self ):
	self.densities = self._evaluate_densities(self.radii)
	self.gravity = self._compute_gravity(self.densities, self.radii)
	self.pressures = self._compute_pressure(self.densities, self.gravity, self.radii)

	self.mass = self._compute_mass(self.densities, self.radii)

    def _evaluate_densities(self, radii):
	#Evaluates the equation of state for each radius slice of the model.
	#Returns density, bulk sound speed, and shear speed.

	rho = np.empty_like(radii)    
	bulk_sound_speed = np.empty_like(radii)    
	shear_velocity = np.empty_like(radii)    

	for i,r in enumerate(radii):
	    if r > self.cmb:
                rho[i] = ice_density
	    else:
                rho[i] = chondrite_density

	return rho

    def _compute_gravity(self, density, radii):
	#Calculate the gravity of the planet, based on a density profile.  This integrates
	#Poisson's equation in radius, under the assumption that the planet is laterally
	#homogeneous. 
 
	#Create a spline fit of density as a function of radius
	rhofunc = UnivariateSpline(radii, density )

	#Numerically integrate Poisson's equation
	poisson = lambda p, x : 4.0 * np.pi * BigG * rhofunc(x) * x * x
	grav = np.ravel(odeint( poisson, 0.0, radii ))
	grav[1:] = grav[1:]/radii[1:]/radii[1:]
	grav[0] = 0.0 #Set it to zero a the center, since radius = 0 there we cannot divide by r^2
	return grav

    def _compute_pressure(self, density, gravity, radii):
	#Calculate the pressure profile based on density and gravity.  This integrates
	#the equation for hydrostatic equilibrium  P = rho g z.

	#convert radii to depths
	depth = radii[-1]-radii

	#Make a spline fit of density as a function of depth
	#rhofunc = UnivariateSpline( depth[::-1], density[::-1] )
	#Make a spline fit of gravity as a function of depth
	#gfunc = UnivariateSpline( depth[::-1], gravity[::-1] )
	rhofunc = interp1d( depth[::-1], density[::-1] , bounds_error=False)
	gfunc = interp1d( depth[::-1], gravity[::-1] , bounds_error=False)
       

	#integrate the hydrostatic equation
	pressure = np.ravel(odeint( (lambda p, x : gfunc(x)* rhofunc(x)), 0.0,depth[::-1]))
	return pressure[::-1]

    def _compute_mass( self, density, radii):
	rhofunc = UnivariateSpline(radii, density )
	mass = quad( lambda r : 4*np.pi*rhofunc(r)*r*r, 
				 radii[0], radii[-1] )[0]
	return mass



n_slices = 300
iapetus = Iapetus(n_slices)
iapetus.generate_profiles()

observed_mass = 1.805635e21

print ("Total mass of the planet: %.2e, or %.0f%% of the observed mass" % (iapetus.mass, iapetus.mass/observed_mass*100. ) )

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = '\usepackage{relsize}'
plt.rc('font', family='sanserif')

#Come up with axes for the final plot
figure = plt.figure( figsize = (12,10) )
ax1 = plt.subplot2grid( (2,2) , (0,0), colspan=3, rowspan=1)
ax2 = plt.subplot2grid( (2,2) , (1,0), colspan=3, rowspan=1)

#Make a subplot showing the calculated pressure profile
ax1.plot( iapetus.radii/1.e3, iapetus.pressures/1.e6, 'k', linewidth=2.)
ax1.set_ylabel("Pressure (MPa)")

#Make a subplot showing the calculated gravity profile
ax2.plot( iapetus.radii/1.e3, iapetus.gravity, 'k', linewidth=2.)
ax2.set_ylabel("Gravity (m/s$^2)$")
ax2.set_xlabel("Radius (km)")

plt.show()
