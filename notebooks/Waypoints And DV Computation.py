# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%reset
%pylab inline
%pdb off

# <headingcell level=3>

# Import libraries and define derivative function(s) for ODE's

# <codecell>


import numpy as np
from pycse import odelay
from IPython.html.widgets import interact, interactive
from IPython.display import clear_output, display, HTML

from thesis_functions.initialconditions import InputDataDictionary, SetInitialConditions
from thesis_functions.visualization import PlotGrid
from thesis_functions.astro import FindOrbitCenter, ComputeLibrationPoints, stop_yEquals0, stop_zEquals0
from thesis_functions.astro import linearDerivativesFunction, nonlinearDerivativesFunction, nonlinearDerivsWithLinearRelmoSTM, nonlinearDerivsWithLinearRelmo
from thesis_functions.astro import PropagateSatellite, ComputeOffsets, ConvertOffsets, BuildRICFrame, BuildVNBFrame

import scipy.integrate as integrate

# <codecell>


# First satellite 

ICs = InputDataDictionary()

mu, timespan, initialstate1 = SetInitialConditions(ICs, ICset = 'Barbee', ICtestcase = 0, numPoints = 200)

# In nondimensional units, r12 = 1, M = 1, timeConst = Period/(2pi) = 1, G = 1
m1 = 5.97219e24;      # Earth
m2 = 7.34767309e22;   # Moon
M = m1 + m2;
G = 6.67384e-11/1e9;  # m3/(kg*s^2) >> converted to km3
r12 = 384400.0;
timeConst = r12**(1.5)/(G*M)**(0.5)
print timeConst
print 36000.0*24.0/timeConst
timeConst = r12**(1.5)/mu**(0.5)
print timeConst
print 36000.0*24.0/timeConst

T = 2.0*np.pi*r12**(1.5)/(G*M)**(0.5)   # Period in seconds

print 'T', T

Waypoints = dict();
Waypoints[0] = {'t': 0.0,
                'r': [20.0/r12, 20.0/r12, 0.0]};
Waypoints[1] = {'t': 36000.0*24.0/timeConst,
                'r': [15.0/r12, -10.0/r12, 0.0]};  # in 10 hours, move 10 km
#Waypoints[0] = {'t': 0.0,
#                'r': [0.002, 0.002, 0.0]};
#Waypoints[1] = {'t': timespan[20],
#                'r': [0.001, 0.001, 0.0]};

print 'Waypoints', Waypoints

#timespanTest = np.array([Waypoints[0]['t'], Waypoints[1]['t']])
timespanTest = np.linspace(Waypoints[0]['t'], Waypoints[1]['t'], 50)

print 'timespanTest', timespanTest
    
X1, X2, L1, L2, L3, L4, L5 = ComputeLibrationPoints(mu)

I6 = np.eye(6);

initialstateForSTM = np.concatenate((initialstate1, I6.reshape(1,36)[0]))

# integrate first satellite and STM from t1 to t2
statesOverTime1 = integrate.odeint(nonlinearDerivsWithLinearRelmoSTM, initialstateForSTM, timespanTest, (mu,))  # "extra arguments must be given in a tuple"

print len(statesOverTime1), len(statesOverTime1[0])

statesOverTime1 = statesOverTime1.T

print len(statesOverTime1), len(statesOverTime1[0])

# rows 7-42
Phi = statesOverTime1[6:42]

print len(Phi), len(Phi[0])

# last column, converted into 6x6
Phi = Phi[:,len(Phi[0])-1].reshape(6,6)
print 'Phi', Phi

# top left corner and top right corner
Phi11 = Phi[:3, :3]
Phi12 = Phi[:3, 3:6]

print 'Phi11', Phi11
print 'Phi12', Phi12

Phi12I = np.linalg.inv(Phi12)

print 'Phi12Inverse', Phi12I

# Compute required velocity at point 0 to take us to point 1 within time (t1-t0)
Waypoints[0]['v'] = np.dot(Phi12I, Waypoints[1]['r'] - np.dot(Phi11, Waypoints[0]['r']))

print 'initial chaser relative velocity', Waypoints[0]['v']

# integrate first and second satellites and STM from t1 to t2
timespanTest = np.linspace(Waypoints[0]['t'], Waypoints[1]['t'], 50)
initialstateForRelmo = np.concatenate((initialstate1, Waypoints[0]['r'], Waypoints[0]['v']))

print 'initialstateForRelmo', initialstateForRelmo

statesOverTime1 = integrate.odeint(nonlinearDerivsWithLinearRelmo, initialstateForRelmo, timespanTest, (mu,))  # "extra arguments must be given in a tuple"

print len(statesOverTime1), len(statesOverTime1[0])

statesOverTime1 = statesOverTime1.T

print len(statesOverTime1), len(statesOverTime1[0])

# target satellite position and velocity over time in RLP frame
x1, y1, z1, xdot1, ydot1, zdot1 = statesOverTime1[0:6]  # rows 1-6

# offset between target and chaser satellite over time in RLP frame
dx, dy, dz, dxdot, dydot, dzdot = statesOverTime1[6:12] # rows 7-12

print 'dx', dx
print 'dy', dy

plot(dx)

# create empty dictionaries
dataoffsetRLP = {};
dataoffsetRLP['offsetRLP'] = {'x':dx, 'y':dy, 'z':dz}

# Plot offset (relative motion) between satellites 1 and 2 in RLP
points = {'zero': [0,0,0]}
PlotGrid('Offset between Satellites 1 and 2 in RLP Frame', 'X', 'Y', 'Z', dataoffsetRLP, points, 'auto')
 

# Plot satellite 1 in RLP frame
#data = {'sat1': {'x':x1, 'y':y1, 'z':z1}}
#points = {'L1': L1}
#PlotGrid('Satellite 1 in RLP Frame', 'X', 'Y', 'Z', data, points, 'equal')
    

# <codecell>


