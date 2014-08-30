# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%reset
%pylab
%pdb off

# Can also do %pylab inline

# <headingcell level=3>

# Import libraries and define derivative function(s) for ODE's

# <codecell>


import numpy as np
from pycse import odelay
from IPython.html.widgets import interact, interactive
from IPython.display import clear_output, display, HTML

from thesis_functions.initialconditions import InputDataDictionary, SetInitialConditions
from thesis_functions.visualization import CreatePlotGrid, SetPlotGridData
from thesis_functions.astro import FindOrbitCenter, ComputeLibrationPoints, stop_yEquals0, stop_zEquals0
from thesis_functions.astro import ComputeNonlinearDerivs, ComputeRelmoDynamicsMatrix
from thesis_functions.astro import odeintNonlinearDerivs, odeintNonlinearDerivsWithLinearRelmoSTM, odeintNonlinearDerivsWithLinearRelmo
from thesis_functions.astro import ComputeRequiredVelocity, PropagateSatelliteAndChaser
from thesis_functions.astro import PropagateSatellite, ComputeOffsets, ConvertOffsets, BuildRICFrame, BuildVNBFrame

import scipy.integrate as integrate

# <headingcell level=3>

# Initial Conditions and Waypoints

# <codecell>

# First satellite 

ICs = InputDataDictionary()

mu, timespan, initialstate1 = SetInitialConditions(ICs, ICset = 'Barbee', ICtestcase = 0, numPoints = 200)

# In nondimensional units, r12 = 1, M = 1, timeConst = Period/(2pi) = 1, G = 1
m1 = 5.97219e24;      # Earth  # kg
m2 = 7.34767309e22;   # Moon   # kg
M = m1 + m2;
G = 6.67384e-11/1e9;  # m3/(kg*s^2) >> converted to km3
r12 = 384400.0;       # km

timeConst = r12**(1.5)/(G*M)**(0.5)  # units are seconds  # this is how you convert between dimensional time (seconds) and non-dimensional time
print 'timeconst', timeConst

T = 2.0*np.pi*r12**(1.5)/(G*M)**(0.5)   # Period in seconds of Moon around Earth
print 'Period of Moon around Earth in seconds', T


# TODO: input waypoints in RIC or VNB frame
# TODO: visualize results in RIC and VNB frames
# TODO: get decent test cases in the Sun-Earth-Moon frame

Waypoints = dict();
Waypoints[0] = {'t': 0.0,
                'r': [0.0, 1000.0/r12, 0.0]};
Waypoints[0] = {'t': 86400.0*2.88/timeConst,      # 2.88 days
                'r': [0.0, 275.0/r12, 0.0]};  # move 725 km  # 400% errors
Waypoints[0] = {'t': 86400.0*4.70/timeConst,      # 1.82 days
                'r': [0.0, 180.0/r12, 0.0]};  # move 95 km  # 400% errors
Waypoints[0] = {'t': 86400.0*5.31/timeConst,
                'r': [0.0, 100.0/r12, 0.0]};  # 40% errors
Waypoints[1] = {'t': 86400.0*5.67/timeConst,
                'r': [0.0, 15.0/r12, 0.0]};  # 8% errors
Waypoints[2] = {'t': 86400.0*6.03/timeConst,
                'r': [0.0, 5.0/r12, 0.0]};  # 10% errors
Waypoints[3] = {'t': 86400.0*6.64/timeConst,
                'r': [0.0, 1.0/r12, 0.0]};  # 
Waypoints[4] = {'t': 86400.0*7.0/timeConst,
                'r': [0.0, 0.030/r12, 0.0]};  # 
Waypoints[5] = {'t': 86400.0*7.26/timeConst,
                'r': [0.0, 0.0/r12, 0.0]};

#Waypoints[0] = {'t': 0.0,
#                'r': [0.002, 0.002, 0.0]};
#Waypoints[1] = {'t': timespan[20],
#                'r': [0.001, 0.001, 0.0]};

S = [0, 1, 2, 3, 4]
for currentPoint in S:
    nextPoint = currentPoint + 1;    
    print currentPoint, Waypoints[currentPoint],'percentage of orbit covered getting to next point:', (Waypoints[nextPoint]['t'] - Waypoints[currentPoint]['t'])/np.max(timespan)*100.0

# Cheat sheet:
# np.array([v1, v2])
# np.linspace(v1, v2, numPoints)
# np.concatenate(( a1, a2 ))

X1, X2, L1, L2, L3, L4, L5 = ComputeLibrationPoints(mu)

# <codecell>


# Create plots
fig1 = plt.figure()
fig2 = plt.figure()
ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)
ax1.set_title('dx_LINEAR vs timespan')
ax2.set_title('Difference between LINEAR and NONLINEAR: dy vs dx')
axXZ, axYZ, axXY, ax3D = CreatePlotGrid('Offset between Satellites 1 and 2 in RLP Frame', 'X', 'Y', 'Z', 'auto')
points = {'zero': [0,0,0]}
dataoffsetRLP = {}
SetPlotGridData(axXZ, axYZ, axXY, ax3D, dataoffsetRLP, points)
for w in Waypoints:
    points = {w: np.array(Waypoints[w]['r'])*r12}
    SetPlotGridData(axXZ, axYZ, axXY, ax3D, dataoffsetRLP, points)
points = {}
    
# Travel between waypoints
for currentPoint in S:

    nextPoint = currentPoint+1;
    
    ## Compute required velocity to travel between waypoints

    # Compute required velocity at point 1 to take us to point 2 within time (t2-t1)
    # This is from Lian et al.
    # Method signature:
    # initialRelativeVelocity = ComputeRequiredVelocity(initialstate1, initialRelativePosition, initialTime, targetRelativePosition, targetTime)
    Waypoints[currentPoint]['v'] = ComputeRequiredVelocity(initialstate1, Waypoints[currentPoint]['r'], Waypoints[currentPoint]['t'], Waypoints[nextPoint]['r'], Waypoints[nextPoint]['t'], mu)

    #print 'initial chaser relative velocity', Waypoints[currentPoint]['v']

    initialRelativeState = np.concatenate(( Waypoints[currentPoint]['r'], Waypoints[currentPoint]['v'] ))

    ## Integrate first satellite with full nonlinear dynamics and second satellite with linear relmo dynamics

    # array of time points
    timespan = np.linspace(Waypoints[currentPoint]['t'], Waypoints[nextPoint]['t'], 500)

    # target satellite position and velocity over time in RLP frame from integrating initial state with full nonlinear dynamics
    # offset between target and chaser satellite over time in RLP frame from integrating initial offset with linearized relmo dynamics
    x1, y1, z1, xdot1, ydot1, zdot1, dx_LINEAR, dy_LINEAR, dz_LINEAR, dxdot_LINEAR, dydot_LINEAR, dzdot_LINEAR = PropagateSatelliteAndChaser(mu, timespan, initialstate1, initialRelativeState)

    ##  Integrate second satellite with full nonlinear dynamics

    # initial state of second satellite in absolute RLP coordinates (not relative to first satellite)
    initialstate2 = np.array(initialstate1) - np.array(initialRelativeState)

    # chaser satellite position and velocity over time in RLP frame from integrating initial state with full nonlinear dynamics
    x2, y2, z2, xdot2, ydot2, zdot2 = PropagateSatellite(mu, timespan, initialstate2);

    # Compute offsets in RLP frame based on nonlinear motion
    dx_NONLIN, dy_NONLIN, dz_NONLIN = ComputeOffsets(timespan, x1, y1, z1, xdot1, ydot1, zdot1, x2, y2, z2, xdot2, ydot2, zdot2);
    

    # Record updated primary satellite initial state and updated chaser satellite waypoint for next iteration
    
    initialstate1 = np.array([ x1[-1], y1[-1], z1[-1], xdot1[-1], ydot1[-1], zdot1[-1] ])
    Waypoints[nextPoint]['r'] = np.array([ dx_NONLIN[-1], dy_NONLIN[-1], dz_NONLIN[-1] ])
    
    ## VISUALIZATIONS

    ax1.plot(timespan, dx_LINEAR*r12)

    #print 'initialState1', initialstate1
    #print 'relative initial state of 2nd wrt 1st', initialRelativeState
    #print 'initialState2', initialstate2
    
    # Compare linear relmo propagation to nonlinear dynamics
    #print np.amax(np.absolute(dx_LINEAR)), np.amax(np.absolute(dy_LINEAR))

    ax2.plot((dx_NONLIN - dx_LINEAR)/np.amax(np.absolute(dx_LINEAR))*100.0, (dy_NONLIN - dy_LINEAR)/np.amax(np.absolute(dy_LINEAR))*100.0)
    
    # create empty dictionaries
    dataoffsetRLP = {};
    dataoffsetRLP['offsetRLPFromLinearRelmo'] = {'x':dx_LINEAR*r12, 'y':dy_LINEAR*r12, 'z':dz_LINEAR*r12}
    dataoffsetRLP['offsetRLPFromNonlinearDynamics'] = {'x':dx_NONLIN*r12, 'y':dy_NONLIN*r12, 'z':dz_NONLIN*r12}
    
    points = {nextPoint: np.array(Waypoints[nextPoint]['r'])*r12}

    # Plot offset (relative motion) between satellites 1 and 2 in RLP
    SetPlotGridData(axXZ, axYZ, axXY, ax3D, dataoffsetRLP, points)


    # Plot satellite 1 in RLP frame
    #data = {'sat1': {'x':x1, 'y':y1, 'z':z1}}
    #points = {'L1': L1}
    #PlotGrid('Satellite 1 in RLP Frame', 'X', 'Y', 'Z', data, points, 'equal')

    

# <codecell>


