import numpy as np

from scipy.integrate import odeint
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.constants import G

import matplotlib.pyplot as plt

import burnman
from burnman import minerals


#Request prettification of matplotlib defaults
try:
  plt.style.use('ian')
except:
  print("Cannot use requested style sheet")

#useful constants
radius_earth = 6371.e3
radius_cmb = 3480.e3 



# A light wrapper around UnivariateSpline 
# which allows for repeated values at 
# boundaries between layers
class PiecewiseSpline:
    def __init__(self, x, y):
        assert( len(x) == len(y) )
        self.splines = []
        left = 0

        for i in range( len(x)-1 ):
            if x[i] == x[i+1]:
                self.splines.append( ((x[left], x[i]), UnivariateSpline(x[left:i], y[left:i])) )
                left = i+1
        self.splines.append( ((x[left], x[-1]), UnivariateSpline(x[left:-1], y[left:-1])) )

    def __call__(self, x):
        if x < self.splines[0][0][0]:
            return self.splines[0][1](x)
        for interval, func in self.splines:
            if x <= interval[1] and x >= interval[0]:
                return func(x)
        if x > self.splines[-1][0][1]:
            return self.splines[-1][1](x)
        raise Exception("Cannot find value "+str(x) )


#Layered rock model for Earth's mantle
class MantleRock(burnman.Composite):
    def __init__(self):
        #Olivine and opx upper mantle
        amount_olivine = 0.8
        fe_ol = 0.1
        fe_opx = 0.3
        self.ol = minerals.SLB_2011.mg_fe_olivine()
        self.opx = minerals.SLB_2011.orthopyroxene()
        self.ol.set_composition([1.-fe_ol, fe_ol] )
        self.opx.set_composition([1.-fe_opx, fe_opx, 0.0, 0.0] )
        self.upper_mantle = burnman.Composite( [amount_olivine, 1.0-amount_olivine], [self.ol,self.opx] )

        #Wadsleyite and garnet transition zone
        amount_wadsleyite = 0.8
        fe_wa = 0.1
        fe_gt = 0.3
        self.wa = minerals.SLB_2011.mg_fe_wadsleyite()
        self.gt = minerals.SLB_2011.garnet()
        self.wa.set_composition([1.-fe_wa, fe_wa] )
        self.gt.set_composition([1.-fe_gt, fe_gt, 0.0, 0.0, 0.0] )
        self.transition_zone = burnman.Composite( [amount_wadsleyite, 1.0-amount_wadsleyite], [self.wa,self.gt] )
       
        #perovskite and ferropericlase lower mantle
        fe_pv=0.05
        fe_pc=0.3
        amount_perovskite = 0.8
        self.pv=minerals.SLB_2011.mg_fe_perovskite()
        self.pc=minerals.SLB_2011.ferropericlase()
        self.pv.set_composition([1.-fe_pv,fe_pv,0.])
        self.pc.set_composition([1.-fe_pc,fe_pc])
        self.lower_mantle = burnman.Composite( [amount_perovskite, 1.0-amount_perovskite], [self.pv,self.pc] )

    def set_state(self, pressure, temperature):
        #Depending on the pressure, select one of the rock types
        if pressure < 14.e9 : #Upper mantle
          self.children = self.upper_mantle.children
        elif pressure < 23.5e9 and pressure > 14.e9: #Transition zone
          self.children = self.transition_zone.children 
        else: #Lower mantle
          self.children = self.lower_mantle.children
        try: #Set state for the rock
          super(MantleRock, self).set_state(pressure, temperature)
        except ValueError:
          print "Cannot evaluate mantle EOS at P,T", pressure, temperature

#Light wrapper around the burnman.Model class 
#to add a method for heat capacity per density, 
#rather than molar heat capacity.  Is this an 
#oversight in the original Model? Perhaps.
class MantleModel( burnman.Model ):
    def __init__(self, rock, p, T):
        super(MantleModel,self).__init__(rock,p,T,burnman.averaging_schemes.HashinShtrikmanAverage() )
        self.c_p_rho = None
    def cp_per_density(self):
        if self.c_p_rho is None:
            self.calc_moduli_()
            self.c_p_rho = np.zeros(len(self.p))
            for idx in range(len(self.p)):
                mass_per_mole_molecules = np.sum( [ m['rho']*m['V'] for m in self.moduli[idx] ] )
                specific_heat_per_mole_molecules = np.sum( [ m['c_p']*m['fraction'] for m in self.moduli[idx] ] )
                self.c_p_rho[idx] =  specific_heat_per_mole_molecules/mass_per_mole_molecules
        return self.c_p_rho

def compute_gravity(density, radii):
    #Calculate the gravity of the planet, based on a density profile.  This integrates
    #Poisson's equation in radius, under the assumption that the planet is laterally
    #homogeneous. 
     
    #Create a spline fit of density as a function of radius
    rhofunc = PiecewiseSpline(radii, density )
 
    #Numerically integrate Poisson's equation
    poisson = lambda p, x : 4.0 * np.pi * G * rhofunc(x) * x * x
    grav = np.ravel(odeint( poisson, 0.0, radii ))
    grav[1:] = grav[1:]/radii[1:]/radii[1:]
    grav[0] = 0.0 #Set it to zero a the center, since radius = 0 there we cannot divide by r^2
    return grav

def compute_pressure(density, gravity, radii):
    #Calculate the pressure profile based on density and gravity.  This integrates
    #the equation for hydrostatic equilibrium  P = rho g z.

    #convert radii to depths
    depth = radii[-1]-radii

    #Make a spline fit of density as a function of depth
    rhofunc = PiecewiseSpline( depth[::-1], density[::-1] )
    #Make a spline fit of gravity as a function of depth
    gfunc = PiecewiseSpline( depth[::-1], gravity[::-1] )

    #integrate the hydrostatic equation
    pressure = np.ravel(odeint( (lambda p, x : gfunc(x)* rhofunc(x)), 0.0,depth[::-1]))
    return pressure[::-1]


#Load in the reference data
ref_data = np.loadtxt("SEMUCB-WM1/data/model.ref", skiprows=3)
ref_radius = ref_data[:,0]
ref_density = ref_data[:,1]
ref_vp = ref_data[:,2]
ref_vs = (ref_data[:,3]+ref_data[:,7])/2.

#compute the gravity and pressure profile from the reference data
ref_gravity = compute_gravity( ref_density, ref_radius)
ref_pressure = compute_pressure( ref_density, ref_gravity, ref_radius)


#Restrict the reference profiles to just the mantle portion now
mantle_radius = ref_radius[np.where( np.logical_and(ref_radius < radius_earth , ref_radius > radius_cmb) )]
mantle_density = ref_density[np.where( np.logical_and(ref_radius < radius_earth , ref_radius > radius_cmb) )]
mantle_pressure = ref_pressure[np.where( np.logical_and(ref_radius < radius_earth , ref_radius > radius_cmb) )]
mantle_gravity = ref_gravity[np.where( np.logical_and(ref_radius < radius_earth , ref_radius > radius_cmb) )]
mantle_vs = ref_vs[np.where( np.logical_and(ref_radius < radius_earth , ref_radius > radius_cmb) )]
mantle_vp = ref_vp[np.where( np.logical_and(ref_radius < radius_earth , ref_radius > radius_cmb) )]



#Make an instance of the MantleRock
mantle_rock = MantleRock()

#Calculate an adiabat with the MantleRock
T0=1600. #Mantle potential temperature
model_temperature = burnman.geotherm.adiabatic(mantle_pressure[::-1], T0, mantle_rock)[::-1]

#Make the mantle model
mantle_model = MantleModel( mantle_rock, mantle_pressure, model_temperature)

#Evaluate a ton of stuff about the model
model_density = mantle_model.density()
model_vp = mantle_model.v_p()
model_vs = mantle_model.v_s()
model_cp = mantle_model.cp_per_density()
model_expansivity = mantle_model.thermal_expansivity()
model_bulk_modulus = mantle_model.K()
model_shear_modulus = mantle_model.G()
#Do a finite difference approximation of dGdT.  Kind of ugly and expensive, but works
dT = 10.0 
mantle_model_hotter = MantleModel( mantle_rock, mantle_pressure, model_temperature+dT)
mantle_model_cooler = MantleModel( mantle_rock, mantle_pressure, model_temperature-dT)
model_dGdT = (mantle_model_hotter.G() - mantle_model_cooler.G())/(2.*dT)

#Generate a figure of the model, comparing it with the SEMUCB model
plt.figure(figsize=(24,16) )
plt.subplot(231)

plt.plot(mantle_radius/1.e3, model_temperature, label=r"Adiabatic profile, $T_p=%2.0f K$"%(T0))
tmp = np.linspace(8.5e9, 135.e9, 100)
plt.plot( mantle_radius[np.where(np.logical_and(mantle_pressure < 135.e9, mantle_pressure > 8.5e9) )]/1.e3, burnman.geotherm.brown_shankland( mantle_pressure[np.where(np.logical_and(mantle_pressure < 135.e9, mantle_pressure > 8.5e9) )] ), label=r"Brown and Shankland, 1981" )
plt.xlabel(r"Radius (km)")
plt.ylabel(r"Temperature (K)")
plt.xlim( radius_cmb/1.e3, radius_earth/1.e3 )
plt.legend(loc='lower left')

plt.subplot(232)

plt.plot(mantle_radius/1.e3, mantle_vp/1.e3, 'k--')
plt.plot(mantle_radius/1.e3, model_vp/1.e3, label=r"Model $V_S$")

plt.plot(mantle_radius/1.e3, mantle_vs/1.e3, 'k--')
plt.plot(mantle_radius/1.e3, model_vs/1.e3, label=r"Model $V_P$")

plt.plot(mantle_radius/1.e3, model_density/1.e3, label=r"Model density")
plt.plot(mantle_radius/1.e3, mantle_density/1.e3, 'k--', label=r"SEMUCB-WM1")

plt.xlabel(r"Radius (km)")
plt.ylabel(r"Wave speed (km/s) or density (kg/m$^3$)")
plt.ylim(2,14)
plt.xlim( radius_cmb/1.e3, radius_earth/1.e3 )
plt.legend(loc='lower left')

plt.subplot(233)
plt.plot(mantle_radius/1.e3, model_expansivity )
plt.xlabel(r"Radius (km)")
plt.ylabel(r"Thermal expansivity (1/K)")
plt.ylim(0, 5.e-5)
plt.xlim( radius_cmb/1.e3, radius_earth/1.e3 )

plt.subplot(234)
plt.plot(mantle_radius/1.e3, model_dGdT )
plt.xlabel(r"Radius (km)")
plt.ylabel(r"$dG/dT$ (Pa/K)")
plt.ylim(-2.5e7, 0)
plt.xlim( radius_cmb/1.e3, radius_earth/1.e3 )

plt.subplot(235)
plt.plot(mantle_radius/1.e3, model_cp )
plt.xlabel(r"Radius (km)")
plt.ylabel(r"$C_p$ (J/kg/K)")
plt.ylim(0, 1500)
plt.xlim( radius_cmb/1.e3, radius_earth/1.e3 )

plt.subplot(236)
plt.plot(mantle_radius/1.e3, 0.5*( model_dGdT/model_shear_modulus + model_expansivity)*100. )
plt.xlabel(r"Radius (km)")
plt.ylabel(r"$d\ln{v_S}/dT \times 100$ (J/kg/K)")
plt.xlim( radius_cmb/1.e3, radius_earth/1.e3 )

plt.savefig("mantle_model.pdf")
#plt.show()

np.savetxt( "mantle_model.txt", \
            zip( mantle_radius, mantle_gravity, mantle_pressure, model_temperature, \
                 model_density, model_vp, model_vs, model_bulk_modulus, \
                 model_shear_modulus, model_expansivity, model_cp, model_dGdT) ,\
            fmt="%12.6e",
            header = "Radius     Gravity      Pressure     Temperature  Density      Vp           Vs           K            G            Expansivity  Cp            dGdT         ")
