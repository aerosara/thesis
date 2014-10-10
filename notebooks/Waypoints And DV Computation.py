# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%reset
%pylab
%pdb off

# Can do "%pylab" or "%pylab inline"

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
from thesis_functions.astro import PropagateSatellite, ComputeOffsets, ConvertOffsets, ConvertOffset, BuildRICFrame, BuildVNBFrame

import scipy.integrate as integrate

# <headingcell level=3>

# Initial Conditions

# <codecell>

# First satellite 

ICs = InputDataDictionary()

mu, timespan, initialstate1 = SetInitialConditions(ICs, ICset = 'Barbee', ICtestcase = 0, numPoints = 200)

X1, X2, L1, L2, L3, L4, L5 = ComputeLibrationPoints(mu)

center = L1

# Build instantaneous RIC and VNB frames
#x1, y1, z1, xdot1, ydot1, zdot1 = initialstate1
#x1 = np.array([initialstate1[0]])
#y1 = np.array([initialstate1[1]])
#z1 = np.array([initialstate1[2]])
#xdot1 = np.array([initialstate1[3]])
#ydot1 = np.array([initialstate1[4]])
#zdot1 = np.array([initialstate1[5]])
#rVec, iVec, cVec = BuildRICFrame(x1, y1, z1, xdot1, ydot1, zdot1, center)
#vVec, nVec, bVec = BuildVNBFrame(x1, y1, z1, xdot1, ydot1, zdot1, center)

# Cheat sheet:
# np.array([v1, v2])
# np.array([rVec[0], iVec[0], cVec[0]])  # makes a 3x3 matrix
# np.linspace(v1, v2, numPoints)
# np.concatenate(( a1, a2 ))

# <headingcell level=3>

# Define Waypoints

# <codecell>

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
# TODO: get decent test cases in the Sun-Earth-Moon frame

Waypoints = dict();
Waypoints[0] = {'t'    : 0.0,
                'r_RIC': [0.0, 1000.0/r12, 0.0]};
Waypoints[0] = {'t'    : 86400.0*2.88/timeConst,      # 2.88 days
                'r_RIC': [0.0, 275.0/r12, 0.0]};  # move 725 km  # 400% errors
Waypoints[0] = {'t'    : 86400.0*4.70/timeConst,      # 1.82 days
                'r_RIC': [0.0, 180.0/r12, 0.0]};  # move 95 km  # 400% errors
Waypoints[0] = {'t'    : 86400.0*5.31/timeConst,
                'r_RIC': [0.0, 100.0/r12, 0.0]};  # 40% errors
Waypoints[1] = {'t'    : 86400.0*5.67/timeConst,
                'r_RIC': [0.0, 15.0/r12, 0.0]};  # 8% errors
Waypoints[2] = {'t'    : 86400.0*6.03/timeConst,
                'r_RIC': [0.0, 5.0/r12, 0.0]};  # 10% errors
Waypoints[3] = {'t'    : 86400.0*6.64/timeConst,
                'r_RIC': [0.0, 1.0/r12, 0.0]};  # 
Waypoints[4] = {'t'    : 86400.0*7.0/timeConst,
                'r_RIC': [0.0, 0.030/r12, 0.0]};  # 
Waypoints[5] = {'t'    : 86400.0*7.26/timeConst,
                'r_RIC': [0.0, 0.0/r12, 0.0]};


# Build RIC-to-RLP frame (inverse of RLP-to-RIC frame)  (TODO)
# TODO: Have to do this once we've propagated to each waypoint time

initialState1ForSegment = initialstate1

S = [0, 1, 2, 3, 4]
#S = [0]
for currentPoint in S:
    nextPoint = currentPoint + 1;    
    print currentPoint, Waypoints[currentPoint], 'percentage of orbit covered getting to next point:', (Waypoints[nextPoint]['t'] - Waypoints[currentPoint]['t'])/np.max(timespan)*100.0
    
    # array of time points
    timespan = np.linspace(Waypoints[currentPoint]['t'], Waypoints[nextPoint]['t'], 500)

    # chaser satellite position and velocity over time in RLP frame from integrating initial state with full nonlinear dynamics
    x1, y1, z1, xdot1, ydot1, zdot1 = PropagateSatellite(mu, timespan, initialState1ForSegment);

    # Build RIC and VNB frames
    rVec, iVec, cVec = BuildRICFrame(x1, y1, z1, xdot1, ydot1, zdot1, center)
    
    if (currentPoint == S[0]):
        
        # this matrix converts from RLP coordinates to the RIC frame at the timestamp of the current (first) point
        RLPtoRIC = np.array([rVec[0], iVec[0], cVec[0]])

        # this matrix converts from RIC to RLP at the timestamp of the current (first) point
        RICtoRLP = np.linalg.inv(RLPtoRIC)
        RLPxVec = RICtoRLP[:,0]
        RLPyVec = RICtoRLP[:,1]
        RLPzVec = RICtoRLP[:,2]
        
        # current point
        drW = Waypoints[currentPoint]['r_RIC'][0]
        diW = Waypoints[currentPoint]['r_RIC'][1]
        dcW = Waypoints[currentPoint]['r_RIC'][2]

        # Convert current waypoint from RIC frame to RLP frame at the timestamp of the current (first) point
        dxW, dyW, dzW = ConvertOffset(drW, diW, dcW, RLPxVec, RLPyVec, RLPzVec);
        Waypoints[currentPoint]['r_RLP'] = [dxW, dyW, dzW]


    # this matrix converts from RLP coordinates to the RIC frame at the timestamp of the next point
    RLPtoRIC = np.array([rVec[-1], iVec[-1], cVec[-1]])
    
    # this matrix converts from RIC to RLP at the timestamp of the next point
    RICtoRLP = np.linalg.inv(RLPtoRIC)
    RLPxVec = RICtoRLP[:,0]
    RLPyVec = RICtoRLP[:,1]
    RLPzVec = RICtoRLP[:,2]
        
    # next point
    drW = Waypoints[nextPoint]['r_RIC'][0]
    diW = Waypoints[nextPoint]['r_RIC'][1]
    dcW = Waypoints[nextPoint]['r_RIC'][2]

    # Convert next waypoint from RIC frame to RLP frame at the timestamp of the next point
    dxW, dyW, dzW = ConvertOffset(drW, diW, dcW, RLPxVec, RLPyVec, RLPzVec);
    Waypoints[nextPoint]['r_RLP'] = [dxW, dyW, dzW]
    
    # Record updated primary satellite initial state for next segment
    initialState1ForSegment = np.array([ x1[-1], y1[-1], z1[-1], xdot1[-1], ydot1[-1], zdot1[-1] ])
    

# <headingcell level=3>

# Travel between waypoints

# <codecell>


# Create plots
#fig1 = plt.figure()
#fig2 = plt.figure()
#ax1 = fig1.add_subplot(111)
#ax2 = fig2.add_subplot(111)
#ax1.set_title('dx_LINEAR vs timespan')
#ax2.set_title('Difference between LINEAR and NONLINEAR: dy vs dx')

# Plots of offset in RLP frame
axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP = CreatePlotGrid('Offset between Satellites 1 and 2 in RLP Frame', 'X', 'Y', 'Z', 'auto')
axXZ_RIC, axYZ_RIC, axXY_RIC, ax3D_RIC = CreatePlotGrid('Offset between Satellites 1 and 2 in RIC Frame', 'R', 'I', 'C', 'auto')
axXZ_VNB, axYZ_VNB, axXY_VNB, ax3D_VNB = CreatePlotGrid('Offset between Satellites 1 and 2 in VNB Frame', 'V', 'N', 'B', 'auto')

# add zero point to plots
points = {'zero': [0,0,0]}
data = {}
SetPlotGridData(axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP, data, points)
SetPlotGridData(axXZ_RIC, axYZ_RIC, axXY_RIC, ax3D_RIC, data, points)
SetPlotGridData(axXZ_VNB, axYZ_VNB, axXY_VNB, ax3D_VNB, data, points)

# add all waypoints to RLP plot
for w in Waypoints:
    points = {w: np.array(Waypoints[w]['r_RLP'])*r12}
    SetPlotGridData(axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP, data, points)

points = {}

initialState1ForSegment = initialstate1

# Travel between waypoints
for currentPoint in S:

    nextPoint = currentPoint + 1;
    
    ## Compute required velocity to travel between waypoints

    # Compute required velocity at point 1 to take us to point 2 within time (t2-t1)
    # This is from Lian et al.
    # Method signature:
    # initialRelativeVelocity = ComputeRequiredVelocity(initialState1ForSegment, initialRelativePosition, initialTime, targetRelativePosition, targetTime)
    Waypoints[currentPoint]['v_RLP'] = ComputeRequiredVelocity(initialState1ForSegment, Waypoints[currentPoint]['r_RLP'], Waypoints[currentPoint]['t'], Waypoints[nextPoint]['r_RLP'], Waypoints[nextPoint]['t'], mu)

    #print 'initial chaser relative velocity', Waypoints[currentPoint]['v']

    initialRelativeState = np.concatenate(( Waypoints[currentPoint]['r_RLP'], Waypoints[currentPoint]['v_RLP'] ))

    ## Integrate first satellite with full nonlinear dynamics and second satellite with linear relmo dynamics

    # array of time points
    timespan = np.linspace(Waypoints[currentPoint]['t'], Waypoints[nextPoint]['t'], 500)

    # target satellite position and velocity over time in RLP frame from integrating initial state with full nonlinear dynamics
    # offset between target and chaser satellite over time in RLP frame from integrating initial offset with linearized relmo dynamics
    x1, y1, z1, xdot1, ydot1, zdot1, dx_LINEAR, dy_LINEAR, dz_LINEAR, dxdot_LINEAR, dydot_LINEAR, dzdot_LINEAR = PropagateSatelliteAndChaser(mu, timespan, initialState1ForSegment, initialRelativeState)

    ##  Integrate second satellite with full nonlinear dynamics

    # initial state of second satellite in absolute RLP coordinates (not relative to first satellite)
    initialstate2 = np.array(initialState1ForSegment) - np.array(initialRelativeState)

    # chaser satellite position and velocity over time in RLP frame from integrating initial state with full nonlinear dynamics
    x2, y2, z2, xdot2, ydot2, zdot2 = PropagateSatellite(mu, timespan, initialstate2);

    # Compute offsets in RLP frame based on nonlinear motion
    dx_NONLIN, dy_NONLIN, dz_NONLIN = ComputeOffsets(timespan, x1, y1, z1, xdot1, ydot1, zdot1, x2, y2, z2, xdot2, ydot2, zdot2);
    
    ## Offsets in RIC and VNB
    
    # Build RIC and VNB frames
    rVec, iVec, cVec = BuildRICFrame(x1, y1, z1, xdot1, ydot1, zdot1, center)
    vVec, nVec, bVec = BuildVNBFrame(x1, y1, z1, xdot1, ydot1, zdot1, center)

    # Compute offsets in RIC frame
    dr_LINEAR, di_LINEAR, dc_LINEAR = ConvertOffsets(dx_LINEAR, dy_LINEAR, dz_LINEAR, rVec, iVec, cVec);
    dr_NONLIN, di_NONLIN, dc_NONLIN = ConvertOffsets(dx_NONLIN, dy_NONLIN, dz_NONLIN, rVec, iVec, cVec);

    # Compute offsets in VNB frame
    dv_LINEAR, dn_LINEAR, db_LINEAR = ConvertOffsets(dx_LINEAR, dy_LINEAR, dz_LINEAR, vVec, nVec, bVec);
    dv_NONLIN, dn_NONLIN, db_NONLIN = ConvertOffsets(dx_NONLIN, dy_NONLIN, dz_NONLIN, vVec, nVec, bVec);   
    
    
    ## Compute waypoint locations in RIC and VNB

    # current point
    dxW = Waypoints[currentPoint]['r_RLP'][0]
    dyW = Waypoints[currentPoint]['r_RLP'][1]
    dzW = Waypoints[currentPoint]['r_RLP'][2]
    
    # Convert waypoint to RIC frame
    drW, diW, dcW = ConvertOffset(dxW, dyW, dzW, rVec[0], iVec[0], cVec[0]);

    # Convert waypoint to VNB frame
    dvW, dnW, dbW = ConvertOffset(dxW, dyW, dzW, vVec[0], nVec[0], bVec[0]);
    
    Waypoints[currentPoint]['r_RIC'] = [drW, diW, dcW]
    Waypoints[currentPoint]['r_VNB'] = [dvW, dnW, dbW]

    # next point
    dxW = Waypoints[nextPoint]['r_RLP'][0]
    dyW = Waypoints[nextPoint]['r_RLP'][1]
    dzW = Waypoints[nextPoint]['r_RLP'][2]
    
    # Convert waypoint to RIC frame
    drW, diW, dcW = ConvertOffset(dxW, dyW, dzW, rVec[-1], iVec[-1], cVec[-1]);

    # Convert waypoint to VNB frame
    dvW, dnW, dbW = ConvertOffset(dxW, dyW, dzW, vVec[-1], nVec[-1], bVec[-1]);
    
    Waypoints[nextPoint]['r_RIC'] = [drW, diW, dcW]
    Waypoints[nextPoint]['r_VNB'] = [dvW, dnW, dbW]
    
    
    ## Output that gets fed into next iteration/segment
    
    # Record updated primary satellite initial state and updated chaser satellite waypoint for next segment
    initialState1ForSegment = np.array([ x1[-1], y1[-1], z1[-1], xdot1[-1], ydot1[-1], zdot1[-1] ])
    Waypoints[nextPoint]['r_RLP'] = np.array([ dx_NONLIN[-1], dy_NONLIN[-1], dz_NONLIN[-1] ])
    
    ## VISUALIZATIONS

    #ax1.plot(timespan, dx_LINEAR*r12)

    # Compare linear relmo propagation to nonlinear dynamics
    #ax2.plot((dx_NONLIN - dx_LINEAR)/np.amax(np.absolute(dx_LINEAR))*100.0, (dy_NONLIN - dy_LINEAR)/np.amax(np.absolute(dy_LINEAR))*100.0)
    
    # create empty dictionaries
    dataoffsetRLP = {};
    dataoffsetRLP['linear_' + str(currentPoint) + '_' + str(nextPoint)] = {'x':dx_LINEAR*r12, 'y':dy_LINEAR*r12, 'z':dz_LINEAR*r12}
    dataoffsetRLP['nonlin_' + str(currentPoint) + '_' + str(nextPoint)] = {'x':dx_NONLIN*r12, 'y':dy_NONLIN*r12, 'z':dz_NONLIN*r12}
    
    dataoffsetRIC = {};
    dataoffsetRIC['linear_' + str(currentPoint) + '_' + str(nextPoint)] = {'x':dr_LINEAR*r12, 'y':di_LINEAR*r12, 'z':dc_LINEAR*r12}
    dataoffsetRIC['nonlin_' + str(currentPoint) + '_' + str(nextPoint)] = {'x':dr_NONLIN*r12, 'y':di_NONLIN*r12, 'z':dc_NONLIN*r12}
    
    dataoffsetVNB = {};
    dataoffsetVNB['linear_' + str(currentPoint) + '_' + str(nextPoint)] = {'x':dv_LINEAR*r12, 'y':dn_LINEAR*r12, 'z':db_LINEAR*r12}
    dataoffsetVNB['nonlin_' + str(currentPoint) + '_' + str(nextPoint)] = {'x':dv_NONLIN*r12, 'y':dn_NONLIN*r12, 'z':db_NONLIN*r12}

    # Plot offset (relative motion) between satellites 1 and 2 in RLP, RIC, and VNB frames
    points = {nextPoint: np.array(Waypoints[nextPoint]['r_RLP'])*r12}
    SetPlotGridData(axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP, dataoffsetRLP, points)
    
    points = {currentPoint: np.array(Waypoints[currentPoint]['r_RIC'])*r12,
              nextPoint: np.array(Waypoints[nextPoint]['r_RIC'])*r12}
    SetPlotGridData(axXZ_RIC, axYZ_RIC, axXY_RIC, ax3D_RIC, dataoffsetRIC, points)
    
    points = {currentPoint: np.array(Waypoints[currentPoint]['r_VNB'])*r12,
              nextPoint: np.array(Waypoints[nextPoint]['r_VNB'])*r12}
    SetPlotGridData(axXZ_VNB, axYZ_VNB, axXY_VNB, ax3D_VNB, dataoffsetVNB, points)
    
    points = {}

    

# <codecell>


# <codecell>


